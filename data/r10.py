"""R10: field-level writes on the second domain, fresh streams (registration in PREREG.md, round 10).

Streams: D.make_pair(2026720000 + 256*10000 + i, 256), frozen R5-4 wording, i < 100, true events in typed form.
Held fixed from R8-2 and R9: writers (Qwen2.5-14B, phi-4), SPEC, GLOSS_D2, the FMT_D2 record grammar, the full prior state
in the user turn, greedy decoding with at most 1,500 tokens, renderer v2, queried-shipment selection, reading by the writer
with likelihood margins, all arms of a writer read in one batch.

Arms
  RULE                 the rule (CPU)
  ONE_K, FREE_K        keyed protocol of R9 (the writer outputs the whole record of the update's shipment), teacher forced
                       and on its own merged state; code now rejects every other shipment
  ONE_F, FREE_F        field protocol: the writer outputs only the fields the update may change (touched()), code keeps every
                       other field and record and rejects anything else
  INJ_K, INJ_F         CPU: ONE_K's and ONE_F's merged single-step differences written into the rule's run
  INJ_KT               CPU: only ONE_K's errors on the allowed fields written into the rule's run (protocol check)
Phases: one_key, free_key, one_field, free_field (GPU, --reader), inj (CPU), read (GPU, --reader), analyze (CPU).
Smoke: R10_SMOKE=1 uses 2 pairs of 2026820000 + 256*10000 + i, 12 updates each, and writes smoke_* files.
"""
import argparse, gzip, json, os, random, re, sys
from collections import defaultdict
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "domain2"))
import r8  # noqa: E402
import r9  # noqa: E402

SMOKE = bool(os.environ.get("R10_SMOKE"))
r8.BASE2, r8.N_PAIRS2, r8.SMOKE_STEPS = (2026820000, 2, 12) if SMOKE else (2026720000, 100, None)
PFX = "smoke_" if SMOKE else ""
READERS, DOM, W_MAX, T = r8.READERS, 2, r8.W_MAX, r8.T
FIELD_NAME = {"clearance": "clearance", "remembered": "remembered", "paid": "payment", "holds": "holds"}
FIELD_RX = {"clearance": r"clearance=(none|no_limit|through:\d+)", "remembered": r"remembered=(none|no_limit|through:\d+)",
            "paid": r"payment=(confirmed|not_confirmed)", "holds": r"holds=([A-Za-z+_]+)"}
FIELD_FMT = ("Output only the fields listed under Fields to output, on one line that starts with shipment=<id> and gives each "
             "listed field as field=value in the record format, as they stand after the update, between a line CHANGES and a "
             "line END. Write a listed field unchanged if the update does not change it. Code keeps every field and record "
             "you do not output, and ignores any other field or record.")
DEFAULT_REC = {"clearance": None, "remembered": None, "paid": False, "holds": []}


def out_path(name):
    return os.path.join(HERE, PFX + name)


def streams():
    return r8.r82_streams(DOM)


def allowed(u):
    return sorted(f for _, f in r8.touched(DOM, u))


# ------------------------------------------------------------------ prompts
def sys_field():
    s = r8.writer_sys(DOM)
    assert s.count(r9.D2_OLD) == 1
    return s.replace(r9.D2_OLD, FIELD_FMT)


def prompt(tok, mode, prev_n, prior_lines, n, u):
    if mode == "key":
        return r9.writer_prompt_diff(tok, DOM, prev_n, prior_lines, n, u, "key")
    user = (f"Current state after Event {prev_n:03d}:\nSTATE\n" + ("\n".join(prior_lines) or "(no records)") + f"\nEND\n\n"
            f"Update:\n{r8.typed2(n, u)}\n\nFields to output:\nshipment={u['shipment']} "
            + " ".join(FIELD_NAME[f] for f in allowed(u)) + f"\n\nOutput these fields as they stand after Event {n:03d}.")
    return tok.apply_chat_template([{"role": "system", "content": sys_field()}, {"role": "user", "content": user}],
                                   tokenize=False, add_generation_prompt=True)


# ------------------------------------------------------------------ parsing with a hard filter
def block(text):
    lines = text.splitlines()
    idx = [i for i, l in enumerate(lines) if r9._tag(l) == "CHANGES"]
    if not idx:
        return lines
    out = []
    for l in lines[idx[-1] + 1:]:
        if r9._tag(l) == "END":
            break
        out.append(l)
    return out


def _val(f, x):
    x = x.lower()
    if f in ("clearance", "remembered"):
        return None if x == "none" else {"through": None} if x == "no_limit" else {"through": int(x.split(":")[1])}
    if f == "paid":
        return x == "confirmed"
    return [] if x == "none" else sorted({h for h in x.split("+") if h in ("compliance", "floor")})


def parse_key(text, u, n):
    """Keyed: the whole record of the update's shipment. -> (record or None, rejected other shipments)."""
    d = r8.parse2("\n".join(block(text)), n)
    sh = (d or {}).get("shipments") or {}
    return sh.get(u["shipment"]), sorted(s for s in sh if s != u["shipment"])


def parse_field(text, u, n):
    """Field: the allowed fields on the line naming the update's shipment. -> (fields found, rejected, all present)."""
    sid, want = u["shipment"], allowed(u)
    found, rejected = {}, []
    for line in block(text):
        m = re.search(r"shipment=(\S+)", line)
        if not m:
            continue
        if m.group(1) != sid:
            rejected.append("shipment=" + m.group(1))
            continue
        for f, rx in FIELD_RX.items():
            v = re.search(r"\b" + rx, line, re.I)
            if v is None:
                continue
            if f in want:
                found[f] = _val(f, v.group(1))
            else:
                rejected.append(FIELD_NAME[f])
    return found, rejected, all(f in found for f in want)


def merge(mode, prior, out, u, n):
    st = {"last_event": n, "shipments": {k: dict(v) for k, v in prior["shipments"].items()}}
    sid = u["shipment"]
    if mode == "key":
        if out is not None:
            st["shipments"][sid] = dict(out)
        return st
    rec = dict(st["shipments"].get(sid, DEFAULT_REC))
    rec.update(out)
    st["shipments"][sid] = rec
    return st


def generate(llm, mode, jobs):
    """jobs: [(prompt, u, n)] -> [dict], one retry with CHANGES prefilled when the named record or a listed field is missing."""
    from vllm import SamplingParams
    sp = SamplingParams(temperature=0.0, max_tokens=W_MAX)

    def parse(text, u, n):
        if mode == "key":
            rec, rej = parse_key(text, u, n)
            return rec, rej, rec is not None
        return parse_field(text, u, n)
    res = llm.generate([p for p, _, _ in jobs], sp, use_tqdm=False)
    out = []
    for (p, u, n), r in zip(jobs, res):
        o, rej, ok = parse(r.outputs[0].text, u, n)
        out.append({"first": r.outputs[0].text, "retry": None, "out": o, "rejected": rej, "ok": ok,
                    "tokens": len(r.prompt_token_ids) + len(r.outputs[0].token_ids), "finish": r.outputs[0].finish_reason})
    bad = [i for i, o in enumerate(out) if not o["ok"]]
    if bad:
        rr = llm.generate([jobs[i][0] + "CHANGES\n" for i in bad], sp, use_tqdm=False)
        for i, r in zip(bad, rr):
            o, rej, ok = parse("CHANGES\n" + r.outputs[0].text, jobs[i][1], jobs[i][2])
            if mode == "field" and not ok:                       # keep what was found; missing fields keep the prior
                o = {**out[i]["out"], **o}
            out[i].update(retry=r.outputs[0].text, out=o if (ok or mode == "field") else out[i]["out"],
                          rejected=out[i]["rejected"] + rej, ok=ok)
            out[i]["tokens"] += len(r.prompt_token_ids) + len(r.outputs[0].token_ids)
    return out


def _out_json(mode, o):
    if o is None:
        return None
    if mode == "key":
        return r8.lines2({"shipments": {"x": o}})[0].split(" ", 1)[1]
    return {FIELD_NAME[f]: (r8.lim(v) if f in ("clearance", "remembered") else v) for f, v in o.items()}


def _load_out(mode, rec):
    """Rebuild the parsed output of a stored step."""
    if rec["out"] is None:
        return None
    if mode == "key":
        return r8.parse2("shipment=x " + rec["out"], rec["n"])["shipments"]["x"]
    inv = {v: k for k, v in FIELD_NAME.items()}
    return {inv[k]: (_val(inv[k], v) if inv[k] in ("clearance", "remembered") else v) for k, v in rec["out"].items()}


# ------------------------------------------------------------------ writer phases
def phase_one(reader, mode):
    import domain2 as D
    llm = r8.vllm(reader)
    tok = llm.get_tokenizer()
    meta, jobs = [], []
    for key, pair, ups, _ in streams():
        st, prev = D.new_state(), 0
        for k, (n, u) in enumerate(ups):
            jobs.append((prompt(tok, mode, prev, r8.to_lines(DOM, st), n, u), u, n))
            meta.append((key, k, n))
            st, prev = r8.step2(st, n, u), n
    res = generate(llm, mode, jobs)
    with gzip.open(out_path(f"r10_one{mode}_{reader}.jsonl.gz"), "wt") as f:
        for (key, k, n), o in zip(meta, res):
            f.write(json.dumps({"stream": key, "k": k, "n": n, **dict(o, out=_out_json(mode, o["out"]))}) + "\n")
    print(reader, f"ONE {mode} steps", len(jobs), "failed after retry", sum(not o["ok"] for o in res),
          "rejected", sum(len(o["rejected"]) for o in res), flush=True)


def phase_free(reader, mode):
    import domain2 as D
    llm = r8.vllm(reader)
    tok = llm.get_tokenizer()
    S = streams()
    cur = {key: D.new_state() for key, _, _, _ in S}
    prev = {key: 0 for key, _, _, _ in S}
    f = gzip.open(out_path(f"r10_free{mode}_{reader}.jsonl.gz"), "wt")
    bad = 0
    for k in range(max(len(ups) for _, _, ups, _ in S)):
        act = [(key, ups[k]) for key, _, ups, _ in S if k < len(ups)]
        jobs = [(prompt(tok, mode, prev[key], r8.to_lines(DOM, cur[key]), n, u), u, n) for key, (n, u) in act]
        for (key, (n, u)), o in zip(act, generate(llm, mode, jobs)):
            bad += not o["ok"]
            cur[key], prev[key] = merge(mode, cur[key], o["out"], u, n), n
            f.write(json.dumps({"stream": key, "k": k, "n": n, **dict(o, out=_out_json(mode, o["out"])),
                                "state": r8.to_lines(DOM, cur[key])}) + "\n")
        if k % 10 == 0:
            print(reader, f"FREE {mode} step", k, flush=True)
    f.close()
    with open(out_path(f"r10_free{mode}_final_{reader}.jsonl"), "w") as g:
        for key in cur:
            g.write(json.dumps({"stream": key, "state": r8.to_lines(DOM, cur[key])}) + "\n")
    print(reader, f"FREE {mode} done, failed after retry", bad, flush=True)


# ------------------------------------------------------------------ injection and per-step errors (CPU)
def phase_inj():
    import domain2 as D
    out, stats = [], []
    S = streams()
    for reader in READERS:
        one = {m: defaultdict(dict) for m in ("key", "field")}
        for m in one:
            for rec in map(json.loads, gzip.open(out_path(f"r10_one{m}_{reader}.jsonl.gz"), "rt")):
                one[m][rec["stream"]][rec["k"]] = rec
        z = defaultdict(int)
        for key, pair, ups, _ in S:
            rule = D.new_state()
            inj = {"INJ_K": rule, "INJ_F": rule, "INJ_KT": rule}
            for k, (n, u) in enumerate(ups):
                prior = rule
                rule = r8.step2(rule, n, u)
                inj = {a: r8.step2(x, n, u) for a, x in inj.items()}
                tc, rc, pc = r8.touched(DOM, u), r8.comps(DOM, rule), r8.comps(DOM, prior)
                z["steps"] += 1
                z["touched"] += len(tc)
                for m, X in (("key", "K"), ("field", "F")):
                    rec = one[m][key][k]
                    z[f"{m}_failed"] += not rec["ok"]
                    z[f"{m}_rejected"] += len(rec["rejected"])
                    mc = r8.comps(DOM, merge(m, prior, _load_out(m, rec), u, n))
                    for c in set(rc) | set(mc):
                        if rc.get(c) == mc.get(c):
                            continue
                        eff = r9.effective(DOM, c, rc.get(c), mc.get(c))
                        if c in tc:
                            z[f"{m}_touched_wrong"] += eff
                            z[f"{m}_touched_wrong:{u['op']}"] += eff
                        elif c[0] == u["shipment"]:
                            z[f"{m}_samerecord_wrong"] += eff
                        else:
                            z[f"{m}_other_wrong"] += eff
                        inj[f"INJ_{X}"] = r8.set_comp(DOM, inj[f"INJ_{X}"], c, mc.get(c))
                        if m == "key" and c in tc:
                            inj["INJ_KT"] = r8.set_comp(DOM, inj["INJ_KT"], c, mc.get(c))
            for a, st in inj.items():
                out.append({"writer": reader, "stream": key, "arm": a, "state": r8.to_lines(DOM, st)})
        stats.append(dict(z, writer=reader, streams=len(S)))
    with open(out_path("r10_inj_final.jsonl"), "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    json.dump(stats, open(out_path("r10_step_stats.json"), "w"), indent=1)
    print(json.dumps(stats, indent=1))


# ------------------------------------------------------------------ reading
ARMS10 = ("RULE", "INJ_K", "FREE_K", "INJ_F", "FREE_F", "INJ_KT")


def rows(reader):
    inj = {(r["stream"], r["arm"]): r["state"] for r in map(json.loads, open(out_path("r10_inj_final.jsonl"))) if r["writer"] == reader}
    fr = {m: {r["stream"]: r["state"] for r in map(json.loads, open(out_path(f"r10_free{m}_final_{reader}.jsonl")))}
          for m in ("key", "field")}
    out = []
    for key, pair, ups, s in streams():
        rule = r8.run2(ups)
        sts = {"RULE": rule, "FREE_K": r8.from_lines(DOM, fr["key"][key]), "FREE_F": r8.from_lines(DOM, fr["field"][key])}
        sts.update({a: r8.from_lines(DOM, inj[(key, a)]) for a in ("INJ_K", "INJ_F", "INJ_KT")})
        for arm in ARMS10:
            st = sts[arm]
            out.append({"domain": 2, "truth": s["truth"], "family": s["family"], "query": s["query"], "arm": arm, "writer": reader,
                        "pair": pair, "stream": key, "memory": "\n".join(r8.query_rows2(st, s["entity"], False)),
                        "state_answer": r8.answer2(st, s["entity"]),
                        "query_exact": r8.query_comps2(st, s["entity"]) == r8.query_comps2(rule, s["entity"])})
    return out


def phase_read(reader):
    from transformers import AutoTokenizer
    from x3_recursive_memory import READERS as RP
    tok = AutoTokenizer.from_pretrained(RP[reader][0], local_files_only=True)
    items = []
    for r in rows(reader):
        p, cue = r8.prompt_parts(tok, dict(r, arm="Enat"))
        items.append((r, p + cue))
    r8.read_items(reader, items, out_path(f"r10_read_{reader}.jsonl"))


# ------------------------------------------------------------------ analysis
def exact_ba(rs):
    by = defaultdict(list)
    for r in rs:
        by[r["truth"]].append(r["decision"] == r["truth"])
    return sum(Fraction(sum(v), len(v)) for v in by.values()) / len(by)


def contrast(rs, w):
    """(exact estimate as Fraction, lo, hi, one-sided p) of sum_w w * BA(arm), pair bootstrap 2000, seed 0."""
    est, lo, hi, p = r8.boot_combo(rs, w)
    g = defaultdict(list)
    for r in rs:
        g[r["arm"]].append(r)
    return sum(c * exact_ba(g[a]) for a, c in w.items()), lo, hi, p


def f2(x):
    from fmt import r2
    return r2(float(x))


def show(c):
    return f"{'+' if c[0] >= 0 else ''}{f2(c[0])} [{f2(c[1])}, {f2(c[2])}] p={'< 0.0005' if c[3] == 0 else f'{c[3]:.4f}'}"


def phase_analyze():
    M = Fraction(1, 20)
    R = {rd: [json.loads(l) for l in open(out_path(f"r10_read_{rd}.jsonl"))] for rd in READERS}
    E = {"reader": R, "state": {rd: [dict(r, decision=r["state_answer"]) for r in R[rd]] for rd in READERS}}
    out = ["R10 field-level writes, second domain, fresh streams. Balanced accuracy read by the writer (exact state accuracy in brackets)"]
    for rd in READERS:
        out.append(f"  {rd}: " + ", ".join(f"{a} {f2(exact_ba([r for r in E['reader'][rd] if r['arm'] == a]))} "
                                           f"({f2(exact_ba([r for r in E['state'][rd] if r['arm'] == a]))})" for a in ARMS10))
    res = {}
    for lab, w in (("H10-1", {"RULE": 1, "FREE_F": -1}), ("H10-2", {"FREE_F": 1, "FREE_K": -1})):
        for ep in ("reader", "state"):
            cs = {rd: contrast(E[ep][rd], w) for rd in READERS}
            rej = r8.holm([cs[rd][3] for rd in READERS])
            for rd, h in zip(READERS, rej):
                res[(lab, ep, rd)] = (cs[rd], h and cs[rd][0] >= M and cs[rd][1] > 0)
    for rd in READERS:
        h1 = res[("H10-1", "reader", rd)][1] and res[("H10-1", "state", rd)][1]
        h2 = res[("H10-2", "reader", rd)][1] and res[("H10-2", "state", rd)][1]
        dis = (res[("H10-1", "reader", rd)][1] != res[("H10-1", "state", rd)][1] or
               res[("H10-2", "reader", rd)][1] != res[("H10-2", "state", rd)][1])
        e10 = res[("H10-1", "reader", rd)][0][2] < 0.05 and res[("H10-1", "state", rd)][0][2] < 0.05
        pc = contrast(E["state"][rd], {"INJ_F": 1, "INJ_KT": -1})
        pok = pc[1] >= -0.05 and pc[2] <= 0.05
        case = ("E, unresolved (reader and state disagree)" if dis else "B, both" if h1 and h2 and pok else
                "B', both, protocol moved" if h1 and h2 else "A, computation" if h1 else "C, copying" if h2 and e10 else
                "D, partly resolved" if h2 else "E, unresolved")
        for lab in ("H10-1", "H10-2"):
            for ep in ("reader", "state"):
                c, ok = res[(lab, ep, rd)]
                out.append(f"  {rd} {lab} {ep}: {show(c)} -> {'holds' if ok else 'does not hold'}")
        out.append(f"  {rd} E10 {e10}; protocol check INJ_F - INJ_KT (state) {show(pc)} -> {'within 0.05' if pok else 'outside'}; case {case}")
    out.append("secondary, report only")
    for lab, w in (("RULE - INJ_F", {"RULE": 1, "INJ_F": -1}), ("INJ_F - FREE_F", {"INJ_F": 1, "FREE_F": -1}),
                   ("RULE - FREE_K", {"RULE": 1, "FREE_K": -1}), ("RULE - INJ_K", {"RULE": 1, "INJ_K": -1}),
                   ("INJ_K - FREE_K", {"INJ_K": 1, "FREE_K": -1}), ("RULE - INJ_KT", {"RULE": 1, "INJ_KT": -1})):
        for rd in READERS:
            out.append(f"  {lab} {rd}: reader {show(contrast(E['reader'][rd], w))}; state {show(contrast(E['state'][rd], w))}")
    stats = {s["writer"]: s for s in json.load(open(out_path("r10_step_stats.json")))}
    for rd in READERS:
        s = stats[rd]
        rs = R[rd]
        dec = {(r["arm"], r["stream"]): r["decision"] for r in rs}
        keys = {r["stream"] for r in rs}
        disc = lambda a, b: sum(dec[(a, k)] != dec[(b, k)] for k in keys)
        line = f"  steps {rd}: {s['steps']} updates, {s['touched']} allowed fields."
        for m, X in (("key", "K"), ("field", "F")):
            line += (f" ONE_{X}: allowed fields wrong {s.get(f'{m}_touched_wrong', 0)} ({s.get(f'{m}_touched_wrong', 0) / s['steps']:.3f} per update), "
                     f"other fields of the shipment wrong {s.get(f'{m}_samerecord_wrong', 0)}, other records wrong {s.get(f'{m}_other_wrong', 0)}, "
                     f"rejected writes {s.get(f'{m}_rejected', 0)}, failed {s.get(f'{m}_failed', 0)}.")
        out.append(line + f" Decisions differ INJ_K/FREE_K {disc('INJ_K', 'FREE_K')}, INJ_F/FREE_F {disc('INJ_F', 'FREE_F')} of {len(keys)}")
        ops = sorted({k.split(":")[1] for k in s if ":" in k})
        out.append(f"  allowed-field errors by op {rd}: " + ", ".join(
            f"{op} K {s.get('key_touched_wrong:' + op, 0)} F {s.get('field_touched_wrong:' + op, 0)}" for op in ops))
    open(out_path("r10_analyze.out"), "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True)
    ap.add_argument("--reader")
    a = ap.parse_args()
    {"one_key": lambda: phase_one(a.reader, "key"), "free_key": lambda: phase_free(a.reader, "key"),
     "one_field": lambda: phase_one(a.reader, "field"), "free_field": lambda: phase_free(a.reader, "field"),
     "inj": phase_inj, "read": lambda: phase_read(a.reader), "analyze": phase_analyze}[a.phase]()
