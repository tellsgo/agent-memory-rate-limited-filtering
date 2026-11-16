#!/usr/bin/env python3
"""
R9 (PREREG, third fresh blind review, independent referee REFEREE_fresh3): separate computing an update from keeping
the state, and rerun StateMem with its published prompts and a parser that leaves strings intact.

R9-1, local edits on the R8-2 streams (first 100 pairs per domain, true events, null events skipped), with the R8-2
policy, glossary, record grammar, writers, greedy decoding, renderer v2, selection and reader held fixed. The writer
sees the rule's or its own complete prior state and the update, as in R8-2, and outputs only some records between
CHANGES and END. Code writes each output record over the prior state and keeps every other record.
  keyed (registered)    code lists the record or records the update concerns (named by the update itself); the writer
                        writes them as they stand after the update, unchanged if the update does not change them
  open diff (secondary) the writer outputs the records the update creates or changes, NONE if none (the smoke run showed
                        that this format leads the writers to answer NONE for updates that create a record)
  ONE_K, ONE_D   teacher forced from the rule's prior state; FREE_K, FREE_D recursive from the empty state on the
                 writer's own merged state; INJ_K, INJ_D (CPU) the rule's run with ONE_K's or ONE_D's merged
                 differences from the rule written in at every step
R9-0 (report only): INJ_T and INJ_U, R8-2's full-rewrite errors (ONE) injected into the rule's run only on the
  components the update touches, or only on the others.
Every final state (RULE, INJ, FREE, INJ_T, INJ_U, INJ_K, FREE_K, INJ_D, FREE_D) is rendered by renderer v2, selected,
and read by the writer model, all arms of a writer in one batch.
  H9-1 BA(FREE_K) - BA(FREE) >= 0.05, lower limit > 0, Holm over the 4 writer x domain cells
  H9-2 BA(RULE) - BA(FREE_K) >= 0.05, lower limit > 0, Holm over the 4 cells
  E9   the upper 95% limit of BA(RULE) - BA(FREE_K) is below 0.05
  A cell's claim also needs the reader-free state answers to point the same way. The same tests on FREE_D are reported.
R9-3, StateMem with the published App. E.1 and E.2 prompts (statemem_published_prompts.json), on the R7-2 streams (first
100 pairs of R7-1 domain 1), encoder Qwen2.5-14B, greedy:
  SM_PUB   units from the E.2 PolicyEncoder, then the E.1 turn encoder with a task context for this log
  SM_FULL  the same units, and the full policy in the task context of every turn
  Parser: json.loads on the raw block first; only after a failure a repair outside string literals (bare unit ids
  quoted, trailing commas dropped); then the one retry. Gate: phase parser_test.
  Reading: App. E.3 layout as in R8-1c, whole and projected; both readers.
  Registered: E - SM_FULL (projected) >= 0.10, lower limit > 0, Holm over the 2 readers (E = R7-1 resolved rows).
Phases: parser_test | one_key, free_key, one_diff, free_diff --reader R --domain D | inj | read --reader R |
        sm_policy | sm_encode --variant pub|full | sm_read --reader R | analyze
"""
import argparse, gzip, json, os, re, sys, time
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "domain2"))
if os.environ.get("R8_SMOKE"):                      # R9 smoke uses the R8 smoke streams and the R7 smoke streams
    os.environ["R7_SMOKE"] = "1"
import r8  # noqa: E402
from r8 import (PFX, READERS, SMOKE_STEPS, T, W_MAX, answer1, answer2, boot_combo, comps, fmt, from_lines, holm,  # noqa: E402
                native_block, parse1, parse2, project, prompt_parts, query_comps1, query_comps2, query_rows1, query_rows2,
                r82_streams, read_items, run1, run2, set_comp, step1, step2, to_lines, touched, truth_d1, typed1, typed2,
                vllm, writer_sys, SM_GUIDE_NATIVE)
from state_machine import OVERRIDE_RULE_V2, POLICY_V2  # noqa: E402

CELLS = [(rd, d) for d in (1, 2) for rd in READERS]
H82_CELLS = [("qwen2.5-14b", 1), ("phi-4", 1), ("qwen2.5-14b", 2)]      # the cells where H8-2 held
D1_OLD = ("Keep a record once it exists, also when it says allowed=no or member=no. Output the complete state after the "
          "update, every record changed or not, between a line STATE and a line END, and nothing else.")
D2_OLD = "Output the complete state after the update, every record changed or not, between a line STATE and a line END, and nothing else."
DIFF_FMT = ("Output only the records that this update creates or changes, each as its complete record line, between a line "
            "CHANGES and a line END. If the update changes nothing, write NONE between them. Every record you do not "
            "output is kept unchanged.")
KEY_FMT = ("Output only the records listed under Records to output, each as its complete record line as it stands after the "
           "update, between a line CHANGES and a line END. Write a listed record unchanged if the update does not change it. "
           "Every record you do not output is kept unchanged.")


def out_path(name):
    return os.path.join(HERE, PFX + name)


# ------------------------------------------------------------------ R9-1 diff writer
def writer_sys_diff(dom, mode="diff"):
    s, old = writer_sys(dom), (D1_OLD if dom == 1 else D2_OLD)
    assert s.count(old) == 1
    return s.replace(old, DIFF_FMT if mode == "diff" else KEY_FMT)


def record_keys(dom, u):
    """The record lines an update concerns, without values (keyed protocol)."""
    if dom == 2:
        return [f"shipment={u['shipment']}"]
    out = []
    for c in sorted(touched(1, u)):
        out.append({"access": lambda: f"access service={c[1]} resource={c[2]}", "member": lambda: f"member service={c[1]} group={c[2]}",
                    "gg": lambda: f"group_grant group={c[1]} resource={c[2]}", "window": lambda: f"window resource={c[1]}"}[c[0]]())
    return out


def writer_prompt_diff(tok, dom, prev_n, prior_lines, n, u, mode="diff"):
    upd = typed1(n, u) if dom == 1 else typed2(n, u)
    user = f"Current state after Event {prev_n:03d}:\nSTATE\n" + ("\n".join(prior_lines) or "(no records)") + f"\nEND\n\nUpdate:\n{upd}\n\n"
    if mode == "diff":
        user += f"Output the records that the update of Event {n:03d} creates or changes."
    else:
        user += "Records to output:\n" + "\n".join(record_keys(dom, u)) + f"\n\nOutput these records as they stand after Event {n:03d}."
    return tok.apply_chat_template([{"role": "system", "content": writer_sys_diff(dom, mode)}, {"role": "user", "content": user}],
                                   tokenize=False, add_generation_prompt=True)


def _tag(line):
    return line.strip().strip("`*# ").upper()


R_ABS = (re.compile(r"access\s+service=(\S+)\s+resource=(\S+)\b.*\b(?:source|record)=none\b", re.I),
         re.compile(r"member\s+service=(\S+)\s+group=(\S+)\b.*\brecord=none\b", re.I),
         re.compile(r"group_grant\s+group=(\S+)\s+resource=(\S+)\b.*\brecord=none\b", re.I),
         re.compile(r"window\s+resource=(\S+)\b.*\brecord=none\b", re.I))


def absent1(lines):
    """First-domain components a local-edit output declares absent (record=none, or an access line with source=none)."""
    out = set()
    for line in lines:
        line = line.strip().lstrip("-* ").strip()
        for kind, rx in zip(("access", "member", "gg", "window"), R_ABS):
            m = rx.match(line)
            if m:
                out.add((kind,) + m.groups())
                break
    return out


def parse_diff(dom, text, n):
    """-> (status, diff state or None, absent components). status "ok" | "none" | None. The last CHANGES block, else the
    whole text."""
    lines = text.splitlines()
    idx = [i for i, l in enumerate(lines) if _tag(l) == "CHANGES"]
    body = lines
    if idx:
        body = []
        for l in lines[idx[-1] + 1:]:
            if _tag(l) == "END":
                break
            body.append(l)
    ab = absent1(body) if dom == 1 else set()
    d = parse1("\n".join(body)) if dom == 1 else parse2("\n".join(body), n)
    if d is not None:
        d = {c: v for c, v in d.items() if c not in ab} if dom == 1 else d
    if d or ab:
        return "ok", (d or None), ab
    if any(_tag(l) == "NONE" for l in body):
        return "none", None, set()
    return None, None, set()


def merge(dom, prior, d, n, absent=()):
    """Code keeps every record the diff does not name; a second-domain line replaces its shipment's record; a
    first-domain record declared absent is removed."""
    if dom == 1:
        st = dict(prior)
        st.update(d or {})
        for c in absent:
            st.pop(tuple(c), None)
        return st
    st = {"last_event": n, "shipments": {k: dict(v) for k, v in prior["shipments"].items()}}
    for s, v in ((d or {}).get("shipments") or {}).items():
        st["shipments"][s] = dict(v)
    return st


def gen_diffs(llm, dom, jobs):
    """jobs: [(prompt, n)] -> [[text, retry text, status, diff, tokens, finish]], one retry with CHANGES prefilled."""
    from vllm import SamplingParams
    sp = SamplingParams(temperature=0.0, max_tokens=W_MAX)
    res = llm.generate([p for p, _ in jobs], sp, use_tqdm=False)
    out = []
    for (p, n), r in zip(jobs, res):
        k, d, ab = parse_diff(dom, r.outputs[0].text, n)
        out.append([r.outputs[0].text, None, k, d, len(r.prompt_token_ids) + len(r.outputs[0].token_ids), r.outputs[0].finish_reason, ab])
    bad = [i for i, o in enumerate(out) if o[2] is None]
    if bad:
        rr = llm.generate([jobs[i][0] + "CHANGES\n" for i in bad], sp, use_tqdm=False)
        for i, r in zip(bad, rr):
            out[i][1] = r.outputs[0].text
            out[i][2], out[i][3], out[i][6] = parse_diff(dom, "CHANGES\n" + r.outputs[0].text, jobs[i][1])
            out[i][4] += len(r.prompt_token_ids) + len(r.outputs[0].token_ids)
    return out


def _rec(key, k, n, o):
    t1, t2, st, d, ntk, fin, ab = o
    return {"stream": key, "k": k, "n": n, "first": t1, "retry": t2, "status": st or "failed",
            "diff": None if d is None else (to_lines(1, d) if "shipments" not in d else to_lines(2, d)),
            "absent": sorted(list(c) for c in ab), "tokens": ntk, "finish": fin}


def phase_one_diff(reader, dom, mode="diff"):
    """Teacher forced: every step from the rule's prior state."""
    import domain2 as D
    llm = vllm(reader)
    tok = llm.get_tokenizer()
    step = step1 if dom == 1 else step2
    meta, jobs = [], []
    for key, pair, ups, _ in r82_streams(dom):
        st, prev = ({} if dom == 1 else D.new_state()), 0
        for k, (n, u) in enumerate(ups):
            jobs.append((writer_prompt_diff(tok, dom, prev, to_lines(dom, st), n, u, mode), n))
            meta.append((key, k, n))
            st, prev = step(st, n, u), n
    res = gen_diffs(llm, dom, jobs)
    with gzip.open(out_path(f"r9_one{mode}_{reader}_d{dom}.jsonl.gz"), "wt") as f:
        for (key, k, n), o in zip(meta, res):
            f.write(json.dumps(_rec(key, k, n, o)) + "\n")
    print(reader, f"d{dom}", f"ONE {mode} steps", len(jobs), "failed after retry", sum(o[2] is None for o in res), flush=True)


def phase_free_diff(reader, dom, mode="diff"):
    """Recursive: every step from the writer's own merged state, from the empty state."""
    import domain2 as D
    llm = vllm(reader)
    tok = llm.get_tokenizer()
    S = r82_streams(dom)
    cur = {key: ({} if dom == 1 else D.new_state()) for key, _, _, _ in S}
    prev = {key: 0 for key, _, _, _ in S}
    f = gzip.open(out_path(f"r9_free{mode}_{reader}_d{dom}.jsonl.gz"), "wt")
    bad = 0
    for k in range(max(len(ups) for _, _, ups, _ in S)):
        act = [(key, ups[k]) for key, _, ups, _ in S if k < len(ups)]
        jobs = [(writer_prompt_diff(tok, dom, prev[key], to_lines(dom, cur[key]), n, u, mode), n) for key, (n, u) in act]
        res = gen_diffs(llm, dom, jobs)
        for (key, (n, u)), o in zip(act, res):
            bad += o[2] is None
            nxt = merge(dom, cur[key], o[3], n, o[6])
            f.write(json.dumps(dict(_rec(key, k, n, o), state=to_lines(dom, nxt))) + "\n")
            cur[key], prev[key] = nxt, n
        if k % 10 == 0:
            print(reader, f"d{dom}", f"FREE {mode} step", k, flush=True)
    f.close()
    with open(out_path(f"r9_free{mode}_final_{reader}_d{dom}.jsonl"), "w") as g:
        for key in cur:
            g.write(json.dumps({"stream": key, "state": to_lines(dom, cur[key])}) + "\n")
    print(reader, f"d{dom}", f"FREE {mode} done, failed after retry", bad, flush=True)


MODES = ("diff", "key")
DEFAULT2 = {"clearance": "none", "remembered": "none", "paid": False, "holds": ()}     # comps() encodes limits with lim()


def effective(dom, c, a, b):
    """True if two values of component c differ in meaning (an absent record equals its default)."""
    if dom == 1:
        dflt = {"member": False, "window": (False, False)}.get(c[0], "absent")
        a = dflt if a is None and dflt != "absent" else a
        b = dflt if b is None and dflt != "absent" else b
        return a != b
    a = DEFAULT2[c[1]] if a is None else a
    b = DEFAULT2[c[1]] if b is None else b
    return a != b


def phase_inj():
    """CPU: INJ (all), INJ_T, INJ_U from R8-2's ONE outputs; INJ_D and INJ_K from the local-edit outputs; per-step errors."""
    import domain2 as D
    out, stats = [], []
    r8inj = {(r["domain"], r["writer"], r["stream"]): r["state"] for r in map(json.loads, open(out_path("r8_inj_final.jsonl")))}
    for dom in (1, 2):
        step = step1 if dom == 1 else step2
        S = r82_streams(dom)
        for reader in READERS:
            full, loc = defaultdict(dict), {m: defaultdict(dict) for m in MODES}
            for rec in map(json.loads, gzip.open(out_path(f"r8_one_{reader}_d{dom}.jsonl.gz"), "rt")):
                full[rec["stream"]][rec["k"]] = rec
            for m in MODES:
                for rec in map(json.loads, gzip.open(out_path(f"r9_one{m}_{reader}_d{dom}.jsonl.gz"), "rt")):
                    loc[m][rec["stream"]][rec["k"]] = rec
            z = defaultdict(int)
            same_all = 0
            for key, pair, ups, _ in S:
                init = {} if dom == 1 else D.new_state()
                rule, ia, it, iu = init, init, init, init
                il = {m: init for m in MODES}
                for k, (n, u) in enumerate(ups):
                    prior = rule
                    rule, ia, it, iu = (step(x, n, u) for x in (rule, ia, it, iu))
                    il = {m: step(x, n, u) for m, x in il.items()}
                    tc = touched(dom, u)
                    rc, pc = comps(dom, rule), comps(dom, prior)
                    z["steps"] += 1
                    z["touched"] += len(tc)
                    rec = full[key][k]
                    if rec["ok"]:
                        o = comps(dom, from_lines(dom, rec["state"], n))
                        for c in set(rc) | set(o):
                            if rc.get(c) != o.get(c):
                                ia = set_comp(dom, ia, c, o.get(c))
                                if c in tc:
                                    it = set_comp(dom, it, c, o.get(c))
                                else:
                                    iu = set_comp(dom, iu, c, o.get(c))
                            eff = effective(dom, c, rc.get(c), o.get(c))
                            z["full_touched_wrong" if c in tc else "full_untouched_wrong"] += eff
                            z["full_untouched_dropped"] += eff and c not in tc and o.get(c) is None
                    for m in MODES:
                        drec = loc[m][key][k]
                        z[f"{m}_" + drec["status"]] += 1
                        dd = from_lines(dom, drec["diff"], n) if drec["status"] == "ok" and drec["diff"] else None
                        if dom == 1 and dd == {}:
                            dd = None
                        ab = {tuple(c) for c in drec.get("absent") or []}
                        merged = merge(dom, prior, dd, n, ab)
                        mc = comps(dom, merged)
                        named = (set(comps(dom, dd)) if dd else set()) | ab
                        for c in set(rc) | set(mc):
                            if rc.get(c) != mc.get(c):
                                il[m] = set_comp(dom, il[m], c, mc.get(c))
                            if c in tc:
                                z[f"{m}_touched_wrong"] += effective(dom, c, rc.get(c), mc.get(c))
                            elif c in named:
                                if effective(dom, c, pc.get(c), mc.get(c)):
                                    z[f"{m}_spurious"] += 1
                                else:
                                    z[f"{m}_reemitted"] += 1
                for arm, st in (("INJ", ia), ("INJ_T", it), ("INJ_U", iu), ("INJ_D", il["diff"]), ("INJ_K", il["key"])):
                    out.append({"domain": dom, "writer": reader, "stream": key, "arm": arm, "state": to_lines(dom, st)})
                same_all += to_lines(dom, ia) == r8inj.get((dom, reader, key))
            stats.append(dict(z, domain=dom, writer=reader, streams=len(S), inj_equals_r8=same_all))
    with open(out_path("r9_inj_final.jsonl"), "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    json.dump(stats, open(out_path("r9_step_stats.json"), "w"), indent=1)
    print(json.dumps(stats, indent=1))


ARMS9 = ("RULE", "INJ", "FREE", "INJ_T", "INJ_U", "INJ_K", "FREE_K", "INJ_D", "FREE_D")


def r9_rows(reader):
    from x6_parse_filter import query_at
    inj = {(r["domain"], r["stream"], r["arm"]): r["state"] for r in map(json.loads, open(out_path("r9_inj_final.jsonl")))
           if r["writer"] == reader}
    rows = []
    for dom in (1, 2):
        free = {r["stream"]: r["state"] for r in map(json.loads, open(out_path(f"r8_free_final_{reader}_d{dom}.jsonl")))}
        fl = {m: {r["stream"]: r["state"] for r in map(json.loads, open(out_path(f"r9_free{m}_final_{reader}_d{dom}.jsonl")))}
              for m in MODES}
        for key, pair, ups, s in r82_streams(dom):
            rule = run1(ups) if dom == 1 else run2(ups)
            sts = {"RULE": rule, "FREE": from_lines(dom, free[key]), "FREE_D": from_lines(dom, fl["diff"][key]),
                   "FREE_K": from_lines(dom, fl["key"][key])}
            sts.update({a: from_lines(dom, inj[(dom, key, a)]) for a in ("INJ", "INJ_T", "INJ_U", "INJ_D", "INJ_K")})
            for arm in ARMS9:
                st = sts[arm]
                if dom == 1:
                    mem, qc, qr = query_rows1(st, s["p"], s["r"], False), query_comps1(st, s["p"], s["r"]), query_comps1(rule, s["p"], s["r"])
                    row = {"domain": 1, "truth": truth_d1(s), "family": s.get("family"), "query": query_at(s, T),
                           "state_answer": answer1(st, s["p"], s["r"])}
                else:
                    mem, qc, qr = query_rows2(st, s["entity"], False), query_comps2(st, s["entity"]), query_comps2(rule, s["entity"])
                    row = {"domain": 2, "truth": s["truth"], "family": s["family"], "query": s["query"],
                           "state_answer": answer2(st, s["entity"])}
                rows.append(dict(row, arm=arm, writer=reader, pair=pair, stream=key, memory="\n".join(mem), query_exact=qc == qr,
                                 n_components=len(comps(dom, st)), n_components_rule=len(comps(dom, rule))))
    return rows


def phase_read(reader):
    from transformers import AutoTokenizer
    from x3_recursive_memory import READERS as RP
    tok = AutoTokenizer.from_pretrained(RP[reader][0], local_files_only=True)
    items = []
    for r in r9_rows(reader):
        p, cue = prompt_parts(tok, dict(r, arm="Enat"))
        items.append((r, p + cue))
    read_items(reader, items, out_path(f"r9_read_{reader}.jsonl"))


# ------------------------------------------------------------------ R9-3 StateMem, published prompts
PROMPTS = json.load(open(os.path.join(HERE, "statemem_published_prompts.json")))
TASK_CTX = ("## Task context:\nYou are tracking state for an operations log of an access-governance system for service "
            "accounts, read event by event. Later events may supersede or update earlier ones. Track:\n -- hard constraints\n"
            " -- soft preferences\n -- entity status (active / closed / superseded)\n -- cross-unit dependencies\n")
FULL_POLICY = POLICY_V2 + OVERRIDE_RULE_V2
TASK_CTX_FULL = TASK_CTX + "The governance policy that decides access is:\n" + FULL_POLICY + "\n"
SM_TURNS = SMOKE_STEPS or 256


def scan_fix(t):
    """Outside string literals only: quote bare unit ids (U12) and drop commas before ] or }."""
    out, i, ins, esc = [], 0, False, False
    while i < len(t):
        ch = t[i]
        if ins:
            out.append(ch)
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                ins = False
            i += 1
            continue
        if ch == '"':
            ins, esc = True, False
            out.append(ch)
            i += 1
            continue
        m = re.match(r"U\d+", t[i:])
        if m and (i == 0 or not (t[i - 1].isalnum() or t[i - 1] == "_")):
            j = i + len(m.group(0))
            if j >= len(t) or not (t[j].isalnum() or t[j] == "_"):
                out.append('"' + m.group(0) + '"')
                i = j
                continue
        if ch == ",":
            j = i + 1
            while j < len(t) and t[j] in " \t\r\n":
                j += 1
            if j < len(t) and t[j] in "]}":
                i += 1
                continue
        out.append(ch)
        i += 1
    return "".join(out)


def parse_fixed(text, kind="{"):
    """-> (object or None, "raw" | "decode" | "repair" | "fail"). The raw block first, repair only after a failure."""
    o_, c_, typ = ("{", "}", dict) if kind == "{" else ("[", "]", list)
    a, b = text.find(o_), text.rfind(c_)
    if a < 0:
        return None, "fail"
    if b > a:
        try:
            o = json.loads(text[a:b + 1])
            return (o, "raw") if isinstance(o, typ) else (None, "fail")
        except json.JSONDecodeError:
            pass
    try:
        o, _ = json.JSONDecoder().raw_decode(text[a:])
        if isinstance(o, typ):
            return o, "decode"
    except json.JSONDecodeError:
        pass
    if b > a:
        try:
            o = json.loads(scan_fix(text[a:b + 1]))
            return (o, "repair") if isinstance(o, typ) else (None, "fail")
        except json.JSONDecodeError:
            pass
    return None, "fail"


def phase_parser_test():
    """Gate before any R9-3 run: no output that is valid JSON may parse differently from json.loads."""
    import r5_comparators as R
    import r7_n2
    msgs, bad = [], 0
    for name in ("r7_n2_events.jsonl.gz", "r7_n3_events.jsonl.gz"):
        p = os.path.join(HERE, name)
        if not os.path.exists(p):
            continue
        n = valid = broke_old = broke_new = repaired = 0
        for e in map(json.loads, gzip.open(p, "rt")):
            for txt in (e["first"], None if e["retry"] is None else "{" + e["retry"]):
                if txt is None:
                    continue
                n += 1
                ref = R.parse_json(txt)                     # json.loads of the raw block, no repair
                new, how = parse_fixed(txt)
                valid += ref is not None
                broke_old += ref is not None and r7_n2.parse(txt) is None
                if ref is not None and new != ref:
                    broke_new += 1
                repaired += how == "repair"
        bad += broke_new
        msgs.append(f"{name}: {n} outputs, {valid} valid JSON, broken by the R7 repair {broke_old}, "
                    f"changed by the R9 parser {broke_new}, parsed only after the R9 repair {repaired}")
    ev = {(e["stream"], e["t"]): e for e in map(json.loads, gzip.open(os.path.join(HERE, "r7_n2_events.jsonl.gz"), "rt"))
          if e["stream"] == "2028960014-APPROVE" and e["t"] == 21}
    e = ev.get(("2028960014-APPROVE", 21))
    if e is not None:
        ok = parse_fixed(e["first"])[0] is not None
        msgs.append(f"2028960014-APPROVE t=21 first output parses: {ok}")
        bad += not ok
    t = '{"add": [{"content": "derived from U3, U5", "derived_from": [U3, U5,]}], "supersede": [],}'
    o, how = parse_fixed(t)
    ok = o is not None and o["add"][0]["content"] == "derived from U3, U5" and o["add"][0]["derived_from"] == ["U3", "U5"]
    msgs.append(f"synthetic repair test (bare ids and trailing commas outside strings, ids inside a string kept): {ok}")
    bad += not ok
    msgs.append("PARSER TEST " + ("PASSED" if bad == 0 else "FAILED"))
    open(out_path("r9_parser_test.out"), "w").write("\n".join(msgs) + "\n")
    print("\n".join(msgs))
    assert bad == 0


def fill(template, **kw):
    for k, v in kw.items():
        template = template.replace("{" + k + "}", v)
    return template


def llm_14b():
    import r5_comparators as R
    from vllm import LLM
    return LLM(model=R.snapshot("Qwen/Qwen2.5-14B-Instruct"), max_model_len=32768, seed=0, enforce_eager=True,
               enable_prefix_caching=True, gpu_memory_utilization=float(os.environ.get("VLLM_UTIL", "0.85")))


def user_msg(tok, text):
    """The paper does not say how its prompts split into chat messages: each filled prompt is one user message."""
    return tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)


def has_rank(units):
    return any(all(x in str(u.get("content", "")) for x in ("security_office", "access_admin", "ops_console")) for u in units)


def phase_sm_policy():
    from vllm import SamplingParams
    llm = llm_14b()
    tok = llm.get_tokenizer()
    p = user_msg(tok, fill(PROMPTS["policy_encoder"], task_context=TASK_CTX, policy=FULL_POLICY))
    r = llm.generate([p], SamplingParams(temperature=0.0, max_tokens=2048), use_tqdm=False)[0]
    raw, retry = r.outputs[0].text, None
    o, how = parse_fixed(raw, "[")
    if o is None:
        retry = llm.generate([p + "["], SamplingParams(temperature=0.0, max_tokens=2048), use_tqdm=False)[0].outputs[0].text
        o, how = parse_fixed("[" + retry, "[")
    units = [u for u in (o or []) if isinstance(u, dict) and u.get("content")]
    samples = []
    for sd in (1, 2, 3):                              # report only: three samples at temperature 0.7
        x = llm.generate([p], SamplingParams(temperature=0.7, max_tokens=2048, seed=sd), use_tqdm=False)[0].outputs[0].text
        so = parse_fixed(x, "[")[0] or []
        su = [u for u in so if isinstance(u, dict) and u.get("content")]
        samples.append({"seed": sd, "raw": x, "units": su, "has_rank": has_rank(su)})
    json.dump({"raw": raw, "retry": retry, "parse": how, "units": units, "has_rank": has_rank(units), "samples": samples},
              open(out_path("r9_sm_policy.json"), "w"), indent=1)
    print("policy units", len(units), "rank order stated", has_rank(units), "samples", [s["has_rank"] for s in samples])
    assert units, "PolicyEncoder produced no units"


def store_class():
    import r7_n2

    class PubStore(r7_n2.FullStore):
        """FullStore with the published trigger fields (kind, payload, description)."""
        @staticmethod
        def line(k, v, with_turn=True):
            meta = ", ".join(x for x in (v["priority"], v["type"] and f"type {v['type']}", v["scope"] and f"scope {v['scope']}",
                                         v["source_type"] and f"from {v['source_type']}") if x)
            links = "".join(f"; {n.replace('_', ' ')} {', '.join(v[n])}" for n in ("coupled_with", "derived_from") if v[n])
            trig = "".join(f"; trigger {x.get('kind', '')}: {x.get('description') or json.dumps(x.get('payload'))}" for x in v["triggers"])
            turn = ("policy" if v["source"] < 0 else f"turn {v['source']}") if with_turn else ""
            return f"[{k}] ({meta}{', ' + turn if turn else ''}) {v['content']}{links}{trig}"
    return PubStore


def sm_streams():
    import r7_n2
    return r7_n2.r7_streams()


def phase_sm_encode(variant):
    from vllm import SamplingParams
    Store = store_class()
    units = json.load(open(out_path("r9_sm_policy.json")))["units"]
    ctx = TASK_CTX if variant == "pub" else TASK_CTX_FULL
    jobs = [(s, Store(units)) for s in sm_streams()]
    llm = llm_14b()
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0.0, max_tokens=1536)
    log = gzip.open(out_path(f"r9_sm_events_{variant}.jsonl.gz"), "wt")
    cost, bad, t0 = defaultdict(lambda: [0, 0]), defaultdict(int), time.time()
    for t in range(SM_TURNS):
        prompts = [user_msg(tok, fill(PROMPTS["turn_encoder"], current_state=st.encoder_view(), turn_number=str(t),
                                      latest_turn=s["lines"][t], task_context=ctx, action_watching_instructions=""))
                   for s, st in jobs]
        res = llm.generate(prompts, sp, use_tqdm=False)
        first = [r.outputs[0].text for r in res]
        parsed = [parse_fixed(x) for x in first]
        rtxt = {}
        retry = [i for i, (o, _) in enumerate(parsed) if o is None]
        if retry:
            rr = llm.generate([prompts[i] + "{" for i in retry], sp, use_tqdm=False)
            for i, r in zip(retry, rr):
                rtxt[i] = r.outputs[0].text
                parsed[i] = parse_fixed("{" + r.outputs[0].text)
                cost[jobs[i][0]["key"]][0] += 1
                cost[jobs[i][0]["key"]][1] += len(r.prompt_token_ids) + len(r.outputs[0].token_ids)
        for i, ((s, st), r, (o, how)) in enumerate(zip(jobs, res, parsed)):
            c = cost[s["key"]]
            c[0] += 1
            c[1] += len(r.prompt_token_ids) + len(r.outputs[0].token_ids)
            log.write(json.dumps({"t": t, "variant": variant, "stream": s["key"], "first": first[i], "finish": r.outputs[0].finish_reason,
                                  "retry": rtxt.get(i), "parse": how, "ok": o is not None, "out": o}) + "\n")
            if o is None:
                bad[s["key"]] += 1
                continue
            st.apply(o, t)
        if t % 16 == 15:
            print(f"{variant} t={t + 1} {time.time() - t0:.0f}s failed {sum(bad.values())}", flush=True)
    log.close()
    with open(out_path(f"r9_sm_mem_{variant}.jsonl"), "w") as f:
        for s, st in jobs:
            f.write(json.dumps({"stream": s["key"], "pair": s["pair"], "t": SM_TURNS, "units": st.units, "write_calls": cost[s["key"]][0],
                                "write_tokens": cost[s["key"]][1], "failed_events": bad[s["key"]]}) + "\n")
    print(variant, "done, events without a valid update", sum(bad.values()), flush=True)


def phase_sm_read(reader):
    import r5_comparators as R
    from transformers import AutoTokenizer
    from x3_recursive_memory import PERSONAS, READERS as RP, chat
    from x6_parse_filter import query_at
    tok = AutoTokenizer.from_pretrained(RP[reader][0], local_files_only=True)
    Store = store_class()
    ss = {s["key"]: s for s in sm_streams()}
    items = []
    for variant in ("pub", "full"):
        for m in map(json.loads, open(out_path(f"r9_sm_mem_{variant}.jsonl"))):
            s = ss[m["stream"]]
            st = Store([])
            st.units = m["units"]
            base = {"domain": 1, "pair": s["pair"], "stream": m["stream"], "family": s.get("family"), "truth": R.truth_at(s, m["t"])}
            for view, block in (("whole", native_block(st.units)), ("projected", native_block(st.units, project(st.units, s["p"], s["r"]))),
                                ("r72block", st.block())):
                sys_ = PERSONAS["gov"] + "\n\n" + POLICY_V2 + "\n\n" + block + "\n\n" + SM_GUIDE_NATIVE
                user = "Evaluate the incoming request.\n\n[Request]\n" + query_at(s, m["t"])
                items.append((dict(base, arm=f"SM_{variant.upper()}_{view}", memory=block), chat(tok, sys_, user) + "DECISION:"))
    read_items(reader, items, out_path(f"r9_sm_read_{reader}.jsonl"))


# ------------------------------------------------------------------ analysis
def pfmt(p):
    return "< 0.0005" if p == 0 else f"{p:.4f}"


def phase_analyze():
    from r2_analyze import bal
    out = []
    acc = lambda xs: sum(r["state_answer"] == r["truth"] for r in xs) / len(xs)
    R = {rd: [json.loads(l) for l in open(out_path(f"r9_read_{rd}.jsonl"))] for rd in READERS}
    stats = {(s["writer"], s["domain"]): s for s in json.load(open(out_path("r9_step_stats.json")))}
    out.append("R9-1 and R9-0: balanced accuracy read by the writer (state-answer accuracy, no reader, in brackets)")
    for rd, d in CELLS:
        rs = [r for r in R[rd] if r["domain"] == d]
        out.append(f"  {rd} d{d}: " + ", ".join(f"{a} {bal([r for r in rs if r['arm'] == a]):.2f} ({acc([r for r in rs if r['arm'] == a]):.2f})"
                                               for a in ARMS9))
    for X, role in (("K", "registered"), ("D", "secondary, report only")):
        F, I = f"FREE_{X}", f"INJ_{X}"
        res1 = {c: boot_combo([r for r in R[c[0]] if r["domain"] == c[1]], {F: 1, "FREE": -1}) for c in CELLS}
        res2 = {c: boot_combo([r for r in R[c[0]] if r["domain"] == c[1]], {"RULE": 1, F: -1}) for c in CELLS}
        rej1, rej2 = holm([res1[c][3] for c in CELLS]), holm([res2[c][3] for c in CELLS])
        cases = {}
        out.append(f"{'keyed' if X == 'K' else 'open diff'} local edits ({role}):")
        out.append(f"  H9-1 (keeping by code helps): BA({F}) - BA(FREE) >= 0.05, lower limit > 0, Holm over 4")
        out.append(f"  H9-2 (a loss remains when the model only computes): BA(RULE) - BA({F}) >= 0.05, lower limit > 0, Holm over 4")
        out.append(f"  E9 (within margin): upper limit of BA(RULE) - BA({F}) < 0.05")
        for c, h1, h2 in zip(CELLS, rej1, rej2):
            rs = [r for r in R[c[0]] if r["domain"] == c[1]]
            sa = {a: acc([r for r in rs if r["arm"] == a]) for a in ARMS9}
            ok1 = h1 and res1[c][0] >= 0.05 and res1[c][1] > 0
            ok2 = h2 and res2[c][0] >= 0.05 and res2[c][1] > 0
            e9 = res2[c][2] < 0.05
            dis1, dis2 = ok1 and sa[F] - sa["FREE"] <= 0, ok2 and sa["RULE"] - sa[F] <= 0
            ok1, ok2 = ok1 and not dis1, ok2 and not dis2
            case = ("B, both" if ok1 and ok2 else "A, computation" if ok2 else "C, retention" if ok1 and e9 else
                    "D, partly resolved" if ok1 else "E, unresolved")
            cases[c] = case
            out.append(f"  {c[0]} d{c[1]}: H9-1 {fmt(res1[c])} p={pfmt(res1[c][3])} Holm {h1} -> {'holds' if ok1 else 'does not hold'}"
                       f"{' (reader and state disagree)' if dis1 else ''}; H9-2 {fmt(res2[c])} p={pfmt(res2[c][3])} Holm {h2} -> "
                       f"{'holds' if ok2 else 'does not hold'}{' (reader and state disagree)' if dis2 else ''}; E9 {e9}; case {case}")
        sel = [cases[c][0] for c in H82_CELLS]
        top = [k for k in "ABCDE" if sel.count(k) >= 2]
        out.append(f"  abstract case (at least two of the three H8-2 cells): {top[0] if top else 'none, one sentence per domain'}; "
                   f"first domain {[cases[c][0] for c in CELLS if c[1] == 1]}, second domain {[cases[c][0] for c in CELLS if c[1] == 2]}")
    out.append("secondary, report only")
    for lab, w in (("RULE - INJ_K", {"RULE": 1, "INJ_K": -1}), ("INJ_K - FREE_K", {"INJ_K": 1, "FREE_K": -1}),
                   ("RULE - INJ_D", {"RULE": 1, "INJ_D": -1}), ("INJ_D - FREE_D", {"INJ_D": 1, "FREE_D": -1}),
                   ("RULE - FREE", {"RULE": 1, "FREE": -1}), ("RULE - INJ", {"RULE": 1, "INJ": -1}), ("INJ - FREE", {"INJ": 1, "FREE": -1}),
                   ("RULE - INJ_T", {"RULE": 1, "INJ_T": -1}), ("RULE - INJ_U", {"RULE": 1, "INJ_U": -1})):
        for rd, d in CELLS:
            out.append(f"  {lab} {rd} d{d}: {fmt(boot_combo([r for r in R[rd] if r['domain'] == d], w))}")
    for rd, d in CELLS:
        rs = [r for r in R[rd] if r["domain"] == d]
        dec = {(r["arm"], r["stream"]): r["decision"] for r in rs}
        keys = {r["stream"] for r in rs}
        disc = lambda a, b: sum(dec[(a, k)] != dec[(b, k)] for k in keys)
        out.append(f"  {rd} d{d}: queried components exact " + ", ".join(
            f"{a} {sum(r['query_exact'] for r in rs if r['arm'] == a)}" for a in ARMS9) +
            f" of {len(keys)}; decisions differ INJ/FREE {disc('INJ', 'FREE')}, INJ_K/FREE_K {disc('INJ_K', 'FREE_K')}, "
            f"INJ_D/FREE_D {disc('INJ_D', 'FREE_D')}")
    for rd, d in CELLS:
        s = stats[(rd, d)]
        ft = s.get("full_touched_wrong", 0) / s["touched"]
        line = (f"  steps {rd} d{d}: {s['steps']} updates, {s['touched']} updated components. Full rewrite (R8 ONE): updated wrong "
                f"{s.get('full_touched_wrong', 0)} ({ft:.3f}), untouched wrong {s.get('full_untouched_wrong', 0)} "
                f"({s.get('full_untouched_wrong', 0) / s['steps']:.3f} per update, dropped {s.get('full_untouched_dropped', 0)}).")
        for m, X in (("key", "K"), ("diff", "D")):
            dt = s.get(f"{m}_touched_wrong", 0) / s["touched"]
            ratio = (dt / ft) if ft else (1.0 if dt == 0 else float("inf"))
            line += (f" ONE_{X}: updated wrong {s.get(f'{m}_touched_wrong', 0)} ({dt:.3f}), spurious edits {s.get(f'{m}_spurious', 0)} "
                     f"({s.get(f'{m}_spurious', 0) / s['steps']:.3f} per update), re-emitted unchanged {s.get(f'{m}_reemitted', 0)}, "
                     f"NONE {s.get(f'{m}_none', 0)}, failed {s.get(f'{m}_failed', 0)}, ratio to the full rewrite {ratio:.2f}"
                     f"{' -> protocol changes the computation (H9-1 not read as pure retention)' if not (1 / 1.5 <= ratio <= 1.5) else ''}.")
        out.append(line + f" Recomputed INJ equals R8-2 INJ in {s['inj_equals_r8']} of {s['streams']} streams")
    # R9-3
    out.append("R9-3 StateMem with the published prompts; E = the R7-1 resolved rows on the same pairs")
    pol = json.load(open(out_path("r9_sm_policy.json")))
    out.append(f"  PolicyEncoder: {len(pol['units'])} units, parse {pol['parse']}, rank order stated {pol['has_rank']}; "
               f"samples at temperature 0.7 state it {[s['has_rank'] for s in pol['samples']]}")
    for v in ("pub", "full"):
        c = defaultdict(int)
        for e in map(json.loads, gzip.open(out_path(f"r9_sm_events_{v}.jsonl.gz"), "rt")):
            c["events"] += 1
            c["retried"] += e["retry"] is not None
            c["failed"] += not e["ok"]
            c["parse_" + e["parse"]] += 1
        out.append(f"  encoder {v}: " + ", ".join(f"{k} {n}" for k, n in sorted(c.items())))
    res, labs = {}, []
    for rd in READERS:
        sm = [json.loads(l) for l in open(out_path(f"r9_sm_read_{rd}.jsonl"))]
        e7 = [dict(r, arm="E7") for r in map(json.loads, open(out_path(f"r7_n1_read_{rd}.jsonl"))) if r["domain"] == 1 and r["arm"] == "E"]
        pairs = {r["pair"] for r in sm}
        e7 = [r for r in e7 if r["pair"] in pairs]
        for arm in sorted({r["arm"] for r in sm}):
            x = [r for r in sm if r["arm"] == arm]
            d = boot_combo(e7 + x, {"E7": 1, arm: -1})
            out.append(f"  {rd} {arm}: BA {bal(x):.2f}, mean tokens {sum(r['memory_tokens'] for r in x) / len(x):.0f}, E - StateMem {fmt(d)}")
            if arm == "SM_FULL_projected":
                res[rd] = d
        both = sm
        out.append(f"  {rd} SM_FULL - SM_PUB (projected): {fmt(boot_combo(both, {'SM_FULL_projected': 1, 'SM_PUB_projected': -1}))}")
    rej = holm([res[rd][3] for rd in READERS])
    for rd, h in zip(READERS, rej):
        ok = h and res[rd][0] >= 0.10 and res[rd][1] > 0
        verdict = ("holds: with the full policy and a correct parser it still trails the resolved rows" if ok else
                   "upper limit below 0.10: drop any claim of a large StateMem deficit" if res[rd][2] < 0.10 else "report the gap with its interval")
        out.append(f"  registered E - SM_FULL (projected) {rd}: {fmt(res[rd])} p={pfmt(res[rd][3])} Holm {h} -> {verdict}")
    open(out_path("r9_analyze.out"), "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True)
    ap.add_argument("--reader")
    ap.add_argument("--domain", type=int)
    ap.add_argument("--variant", choices=["pub", "full"])
    a = ap.parse_args()
    {"parser_test": phase_parser_test, "one_diff": lambda: phase_one_diff(a.reader, a.domain),
     "free_diff": lambda: phase_free_diff(a.reader, a.domain), "one_key": lambda: phase_one_diff(a.reader, a.domain, "key"),
     "free_key": lambda: phase_free_diff(a.reader, a.domain, "key"), "inj": phase_inj, "read": lambda: phase_read(a.reader),
     "sm_policy": phase_sm_policy, "sm_encode": lambda: phase_sm_encode(a.variant), "sm_read": lambda: phase_sm_read(a.reader),
     "analyze": phase_analyze}[a.phase]()
