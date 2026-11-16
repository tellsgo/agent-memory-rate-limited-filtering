#!/usr/bin/env python3
"""
R5-4 (PREREG, ACT-4 of reviews/REFEREE_v2.md): frozen replication on the second domain with fresh seeds and fresh
wording. domain2/ is not edited: fresh templates are swapped into its TEMPLATES and NOTES tables at run time (the
generator renders through them), and every new template is checked to keep its placeholders and to parse one way.

  --phase templates  GPU (vLLM). Qwen2.5-72B-Instruct-AWQ, a model used neither as parser nor as reader, writes new
                     phrasings for every event type and note kind. Output r5_d2_templates_raw.json.
  --phase check      CPU. Select three valid templates per type and one per note kind (deterministic), check that
                     generated lines parse back exactly one way, write r5_d2_templates.json.
  --phase parse      GPU (vLLM). Qwen2.5-14B-Instruct parses every distinct line with the frozen R2-C prompt.
  --phase predict    CPU. Bounded store over shipment records (random and LRU, C = round(f · mean shipments)):
                     the factor 1 - 2 P_e of the exact rule on the store, written to r5_d2_pred.json.
  --phase read_hf    GPU. Qwen2.5-14B and phi-4, YES/NO likelihood (R2-C prompts): log, true state, parsed state,
                     bounded store, dependency retriever.
  --phase read_gen   GPU (vLLM). gpt-oss-120b generates (log, parsed state, bounded store) as in R4-1.
Streams: make_pair(2026260000 + T·10000 + i, T), T = 256 with i < 100 and T = 1024 with i < 50.
"""
import argparse, json, math, os, random, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "domain2"))
sys.path.insert(0, HERE)
import domain2 as D  # noqa: E402

ORIG_T, ORIG_N = dict(D.TEMPLATES), dict(D.NOTES)
BASE = 2026260000
SETS = {256: 100, 1024: 50}
F_D2 = (0.25, 0.5, 0.75)
MEANING = {
    "grant_open": "customs gives shipment {s} export clearance with no time limit",
    "grant_through": "customs gives shipment {s} export clearance that is valid through Event {N}",
    "revoke": "customs revokes (withdraws) the export clearance of shipment {s}",
    "reinstate": "customs reinstates the clearance of shipment {s} that it had revoked earlier",
    "pay": "the payment for shipment {s} is confirmed",
    "reverse": "the payment for shipment {s} is reversed (it no longer counts as paid)",
    "hold": "{src} places a hold on shipment {s} ({src} stands for who placed it)",
    "lift": "{src} lifts the hold it is responsible for on shipment {s} ({src} stands for who lifted it)",
}
NOTE_MEANING = {
    "paperwork": "customs paperwork for {s} was submitted (this changes nothing)",
    "invoice": "an invoice for {s} was emailed (this changes nothing)",
    "weigh": "{s} was weighed at {v} kg",
    "relabel": "{s} was relabelled at bay {v}",
    "status_question": "a customer asked about the status of {s}",
    "lift_request": "someone asked {src} to lift the hold on {s} (a request only, the hold stays)",
    "reinstate_request": "someone requested a reinstatement of the clearance for {s} (a request only)",
    "pickup": "a pickup slot for {s} was booked",
    "forklift": "the forklift in aisle {v} was recharged",
    "shift": "a shift change happened at the north dock",
    "truck": "a truck arrived at gate {v}",
    "rain": "rain delayed yard work",
    "doors": "the doors at bay {v} were inspected",
    "coffee": "the dock office ran out of coffee",
}
PH = re.compile(r"\{(\w+)\}")


def placeholders(t):
    return sorted(set(PH.findall(t)) - {"Src"} | ({"src"} if "{Src}" in t else set()))


def use_templates(path=os.path.join(HERE, "r5_d2_templates.json")):
    tm = json.load(open(path))
    D.TEMPLATES.clear(); D.TEMPLATES.update({k: tuple(v) for k, v in tm["TEMPLATES"].items()})
    D.NOTES.clear(); D.NOTES.update(tm["NOTES"])


def episodes():
    out = []
    for T, n in SETS.items():
        for i in range(n):
            seed = BASE + T * 10000 + i
            for e in D.make_pair(seed, T):
                e["pair"] = seed
                out.append(e)
    return out


def snapshot(name):
    import glob
    org, n = name.split("/")
    for h in (os.path.expanduser("~/.cache/huggingface/hub"),):
        s = sorted(glob.glob(f"{h}/models--{org}--{n}/snapshots/*"))
        if s:
            return s[0]
    raise FileNotFoundError(name)


# ------------------------------------------------------------------ fresh wording
def phase_templates():
    from vllm import LLM, SamplingParams
    llm = LLM(model=snapshot("Qwen/Qwen2.5-72B-Instruct-AWQ"), max_model_len=4096, seed=0, enforce_eager=True,
              gpu_memory_utilization=float(os.environ.get("VLLM_UTIL", "0.85")))
    tok = llm.get_tokenizer()
    jobs = [("T", k, m) for k, m in MEANING.items()] + [("N", k, m) for k, m in NOTE_MEANING.items()]
    prompts = []
    for kind, k, m in jobs:
        names = sorted(set(PH.findall(m)))
        ph = ("Use these placeholders literally, each exactly once per line: " + ", ".join("{" + p + "}" for p in names) + "."
              if names else "This line has no placeholders: do not write any braces.")
        if "src" in names:
            ph += " Write {src} where the source goes; if the line starts with it, write {Src} instead (capitalised), never both."
        prompts.append(tok.apply_chat_template([{"role": "user", "content":
            "Write 8 different one-sentence lines for an export dock operations log. Each line must mean exactly: " + m + ".\n"
            + ph + " Do not add reasons, conditions, outcomes or any other facts, and do not write any other numbers, "
            "shipment IDs or event numbers. Vary the wording and sentence structure; write like real dock staff. "
            "Output one line per row, no numbering, no quotes."}], tokenize=False, add_generation_prompt=True))
    res = llm.generate(prompts, SamplingParams(temperature=0.9, top_p=0.95, max_tokens=600, seed=0), use_tqdm=False)
    raw = {f"{kind}:{k}": [x.strip().lstrip("-*0123456789. ").strip() for x in r.outputs[0].text.split("\n") if x.strip()]
           for (kind, k, _), r in zip(jobs, res)}
    json.dump(raw, open(os.path.join(HERE, "r5_d2_templates_raw.json"), "w"), indent=1)
    print({k: len(v) for k, v in raw.items()})


def phase_check():
    raw = json.load(open(os.path.join(HERE, "r5_d2_templates_raw.json")))
    orig_texts = {t for v in ORIG_T.values() for t in v} | set(ORIG_N.values())
    T, N = {}, {}
    for key, cands in raw.items():
        kind, k = key.split(":")
        need = placeholders(ORIG_T[k][0] if kind == "T" else ORIG_N[k])
        ok = []
        for c in cands:
            c = c.strip().strip('"')
            if not c.endswith("."):
                c += "."
            names = [x.lower() for x in PH.findall(c)]
            if (placeholders(c) != need or len(names) != len(set(names)) or re.search(r"\d", PH.sub("", c))
                    or "SH-" in c or c in orig_texts or c in ok or re.search(r"\bnone\b", c, re.I)):
                continue
            if "{Src}" in c and not c.startswith("{Src}"):
                continue
            if c.startswith("{src}"):
                c = "{Src}" + c[5:]
            ok.append(c)
        sel = ok[:3] if kind == "T" else ok[:1]
        if (kind == "T" and len(sel) < 3) or (kind == "N" and not sel):
            raise SystemExit(f"not enough valid templates for {key}: {ok}")
        (T if kind == "T" else N)[k] = sel if kind == "T" else sel[0]
    D.TEMPLATES.clear(); D.TEMPLATES.update({k: tuple(v) for k, v in T.items()})
    D.NOTES.clear(); D.NOTES.update(N)
    parsers = [("grant" if k.startswith("grant") else k, None, D._compile(t)) for k, v in T.items() for t in v]
    parsers += [("note", k, D._compile(t)) for k, t in N.items()]
    bad = 0
    for seed in range(7000, 7300):                       # every generated line must parse back exactly one way
        for ep in D.make_pair(seed, 48):
            for line, ev in zip(ep["lines"], ep["events"]):
                text = line.split(": ", 1)[1]
                hits = [(typ, note) for typ, note, rx in parsers if rx.match(text)]
                want = (ev["type"], ev.get("note") if ev["type"] == "note" else None)
                bad += hits != [want]
    if bad:
        raise SystemExit(f"{bad} generated lines do not parse back exactly one way")
    json.dump({"TEMPLATES": T, "NOTES": N}, open(os.path.join(HERE, "r5_d2_templates.json"), "w"), indent=1)
    print("templates ok:", sum(len(v) for v in T.values()), "event templates,", len(N), "notes; 300 pairs parse back one way")


# ------------------------------------------------------------------ parsing (frozen R2-C prompt)
def phase_parse():
    from vllm import LLM, SamplingParams
    from r2c_domain2 import PARSE_SYS, few_shot
    use_templates()
    llm = LLM(model=snapshot("Qwen/Qwen2.5-14B-Instruct"), max_model_len=4096, seed=0, enforce_eager=True,
              gpu_memory_utilization=float(os.environ.get("VLLM_UTIL", "0.85")))
    tok = llm.get_tokenizer()
    shots = few_shot()
    uniq = sorted({l.split(": ", 1)[1] for e in episodes() for l in e["lines"]})
    prompts = [tok.apply_chat_template([{"role": "system", "content": PARSE_SYS},
                                        {"role": "user", "content": shots + f"\n\nLine: Event 001: {c}\nJSON:"}],
                                       tokenize=False, add_generation_prompt=True) for c in uniq]
    res = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=96), use_tqdm=False)
    with open(os.path.join(HERE, "r5_d2_parse_qwen2.5-14b.jsonl"), "w") as f:
        for c, r in zip(uniq, res):
            m = re.search(r"\{.*\}", r.outputs[0].text, re.S)
            try:
                u = json.loads(m.group(0)) if m else None
            except json.JSONDecodeError:
                u = None
            f.write(json.dumps({"content": c, "parsed": u}) + "\n")
    print("parsed", len(uniq))


def parsed_updates(e, P):
    out = []
    for i, l in enumerate(e["lines"]):
        u = P.get(l.split(": ", 1)[1])
        out.append(dict(u, event=i + 1) if isinstance(u, dict) else {"event": i + 1, "op": "no_change", "shipment": None,
                                                                     "source": None, "through": None})
    return out


def run_updates(updates, C=None, policy="lru", rng=None):
    """Exact author rule (apply_update); with C, a bounded store of shipment records that forgets what it evicts."""
    st, touch = D.new_state(), {}
    for u in updates:
        s = u.get("shipment")
        new = s is not None and u.get("op") != "no_change" and s not in st["shipments"]
        try:
            nxt = D.apply_update(st, u)
        except Exception:
            nxt = dict(st, last_event=u["event"])
        st = nxt
        if s in st["shipments"]:
            touch[s] = u["event"]
        if C is not None and new and len(st["shipments"]) > C:
            pool = [x for x in st["shipments"] if x != s]
            v = rng.choice(sorted(pool)) if policy == "random" else min(pool, key=lambda x: (touch.get(x, -1), x))
            del st["shipments"][v]
            touch.pop(v, None)
    return st


def run_events(events, C, policy, rng):
    """Same store on the exact events, with the author's in-place engine (_step) for speed; equal to run_updates."""
    st, touch = {"last_event": 0, "shipments": {}}, {}
    for ev in sorted(events, key=lambda e: e["t"]):
        st["last_event"] = ev["t"]
        if ev["type"] == "note":
            continue
        s = ev["shipment"]
        new = s not in st["shipments"]
        D._step(st, ev)
        touch[s] = ev["t"]
        if new and len(st["shipments"]) > C:
            pool = [x for x in st["shipments"] if x != s]
            v = rng.choice(sorted(pool)) if policy == "random" else min(pool, key=lambda x: (touch.get(x, -1), x))
            del st["shipments"][v]
            touch.pop(v, None)
    return st


def capacities(eps):
    out = {}
    for T in SETS:
        sizes = [len(D.current_state(e["events"], len(e["lines"]))["shipments"]) for e in eps if len(e["lines"]) == T]
        m = sum(sizes) / len(sizes)
        out[T] = {"mean_shipments": m, "C": {str(f): max(1, int(math.floor(f * m + 0.5))) for f in F_D2}}
    return out


def phase_predict(reps=50):
    eps = episodes()                                      # the exact events: wording does not enter the store
    caps = capacities(eps)
    out = {"capacities": caps, "factor": {}}
    for T in SETS:
        E = [e for e in eps if len(e["lines"]) == T]
        for pol in ("random", "lru"):
            for f in F_D2:
                C = caps[T]["C"][str(f)]
                err = {"YES": [], "NO": []}
                for e in E:
                    n = reps if pol == "random" else 1
                    rng = random.Random(f"{e['pair']}-{e['truth']}-{f}-{pol}")
                    wrong = sum(("YES" if D.may_ship(run_events(e["events"], C, pol, rng), e["entity"]) else "NO") != e["truth"] for _ in range(n))
                    err[e["truth"]].append(wrong / n)
                fac = 1 - (sum(err["YES"]) / len(err["YES"]) + sum(err["NO"]) / len(err["NO"]))
                out["factor"][f"{T},{pol},{f}"] = fac
                print(f"T={T} |S|={caps[T]['mean_shipments']:.1f} {pol:6s} f={f} C={C} factor {fac:.3f}", flush=True)
    json.dump(out, open(os.path.join(HERE, "r5_d2_pred.json"), "w"), indent=1)


# ------------------------------------------------------------------ memories and readers
def memories():
    from r2c2_resolved import render_state_resolved
    use_templates()
    P = {r["content"]: r["parsed"] for r in map(json.loads, open(os.path.join(HERE, "r5_d2_parse_qwen2.5-14b.jsonl")))}
    caps = json.load(open(os.path.join(HERE, "r5_d2_pred.json")))["capacities"]
    rows = []
    for e in episodes():
        T = len(e["lines"])
        ups = parsed_updates(e, P)
        mem = {"log": ("log", "\n".join(e["lines"])),
               "state": ("state", "\n".join(render_state_resolved(D.current_state(e["events"], T)))),
               "parsed": ("state", "\n".join(render_state_resolved(run_updates(ups))))}
        for pol in ("random", "lru"):
            C = caps[str(T)]["C"]["0.75"]
            mem[f"bounded_{pol}"] = ("state", "\n".join(render_state_resolved(
                run_updates(ups, C, pol, random.Random(f"{e['pair']}-{e['truth']}-0.75-{pol}-read")))))
        mem["retriever"] = ("log", "\n".join(l for l in e["lines"] if re.search(rf"\b{re.escape(e['entity'])}\b", l)))
        for name, (kind, text) in mem.items():
            rows.append({"cond": name, "kind": kind, "T": T, "pair": e["pair"], "family": e["family"], "truth": e["truth"],
                         "memory": text, "query": e["query"]})
    return rows


def phase_read_hf(reader):
    from r2c_domain2 import reader_prompt, yes_no_margin
    from x3_recursive_memory import load_reader, ntok
    tok, model, _ = load_reader(reader, "cuda:0")
    w_head = model.lm_head.weight.detach().float()
    its = [r for r in memories() if not (r["cond"] == "log" and r["T"] == 1024 and reader == "qwen2.5-14b" and False)]
    out = os.path.join(HERE, f"r5_d2_read_{reader}.jsonl")
    done = sum(1 for _ in open(out)) if os.path.exists(out) else 0
    its.sort(key=lambda r: (r["cond"] == "log", r["T"]))          # long logs last
    for a in range(done, len(its), 400):
        chunk = its[a:a + 400]
        bs = 2 if any(r["cond"] == "log" and r["T"] == 1024 for r in chunk) else 8 if any(r["cond"] == "log" for r in chunk) else 16
        ms = yes_no_margin(tok, model, w_head, [reader_prompt(tok, r["memory"], r["query"], r["kind"]) for r in chunk], "cuda:0", bs=bs)
        with open(out, "a") as f:
            for r, m in zip(chunk, ms):
                f.write(json.dumps({k: v for k, v in r.items() if k not in ("memory", "query")} |
                                   {"margin": m, "decision": "NO" if m > 0 else "YES", "memory_tokens": ntok(tok, r["memory"])}) + "\n")
        print(f"{reader}: {a + len(chunk)}/{len(its)}", flush=True)


def phase_read_gen():
    from vllm import LLM, SamplingParams
    from r2c_domain2 import READ_SYS
    its = [r for r in memories() if r["cond"] in ("log", "parsed", "bounded_random", "bounded_lru")]
    its.sort(key=lambda r: (r["cond"] != "log", r["T"]))
    out = os.path.join(HERE, "r5_d2_read_gpt-oss-120b.jsonl")
    done = sum(1 for _ in open(out)) if os.path.exists(out) else 0
    llm = LLM(model=snapshot(os.environ["GPT_OSS_MODEL_ID"]), max_model_len=65536, seed=0, enforce_eager=True,
              gpu_memory_utilization=float(os.environ.get("VLLM_UTIL", "0.9")), attention_config={"backend": "TRITON_ATTN"})
    sp = SamplingParams(max_tokens=16384, temperature=1.0, top_p=1.0, seed=0)
    for a in range(done, len(its), 25):
        chunk = its[a:a + 25]
        msgs = [[{"role": "user", "content": READ_SYS + "\n\n" + ("[Log]\n" if r["kind"] == "log" else "[State]\n") + r["memory"]
                  + f"\n\n[Question]\n{r['query']}\nThink it through, then end your answer with a final line that is exactly "
                  "'ANSWER: YES' or 'ANSWER: NO'."}] for r in chunk]
        res = llm.chat(msgs, sp, use_tqdm=False, chat_template_kwargs={"reasoning_effort": "medium"})
        with open(out, "a") as f:
            for r, x in zip(chunk, res):
                o = x.outputs[0]
                m = re.findall(r"ANSWER:\s*\**\s*(YES|NO)", o.text.split("</think>")[-1].upper())
                f.write(json.dumps({k: v for k, v in r.items() if k not in ("memory", "query")} |
                                   {"decision": m[-1] if m else None, "gen_tokens": len(o.token_ids), "finish": o.finish_reason,
                                    "prompt_tokens": len(x.prompt_token_ids)}) + "\n")
        print(f"gpt-oss d2: {a + len(chunk)}/{len(its)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["templates", "check", "parse", "predict", "read_hf", "read_gen", "count"])
    ap.add_argument("--reader", default="")
    a = ap.parse_args()
    if a.phase == "count":
        eps = episodes(); print(len(eps), capacities(eps))
    else:
        {"templates": phase_templates, "check": phase_check, "parse": phase_parse, "predict": phase_predict,
         "read_gen": phase_read_gen}.get(a.phase, lambda: phase_read_hf(a.reader))()
