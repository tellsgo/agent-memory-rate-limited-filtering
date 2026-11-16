#!/usr/bin/env python3
"""
R7-1 (PREREG, fresh blind review, editor's request): what does resolution add once selection is right?
Fresh paired streams that no earlier analysis touched, in both domains, every memory written without the query and
every arm shown the same dependency set of the queried state:
  domain 1  200 pairs, 256 events, full background, seeds make_pair_v2(2026400000 + 256*10000 + i, 256), i < 200;
            dependency set = the lines the R5-3 retriever's rule selects (r5_comparators.dependency_lines)
  domain 2  100 pairs, 256 events, frozen R5-4 wording, seeds D.make_pair(2026410000 + 256*10000 + i, 256), i < 100;
            dependency set = the lines that name the queried shipment (the R5-4 retriever's rule)
Arms (the same lines in every arm, the same policy text in the system prompt):
  A raw      the dependency lines as logged, in time order with their event numbers
  B trace    A, and the reader first writes a state trace of at most 250 words that lists the relevant events in
             order, gives each state component's chain of values and resolves its current value by the policy, then
             decides (answer-time resolution by the model, in the manner of StateMem's wrapper)
  C summary  A, and the reader first writes a summary of at most 250 words of what is relevant to the request, then
             decides (the matched control of that wrapper: same call, same budget, no structure)
  D typed    the parser's typed update for each dependency line, in time order with event numbers, not resolved
  E rows     the parsed filter's state (every line parsed, the exact rule applied) restricted to the queried
             components by the same selection rule (the query rows of R6-2 and R6-2b)
  E' check   CPU only: parse only the dependency lines at query time and apply the rule; reported as agreement of its
             rows with E and as cost, not read
Every arm is scored by the reader's log-probability margin between the two answers after the decision prompt, as
everywhere in the paper; B and C append their generated text before that prompt.
  --phase parse    GPU (vLLM). Qwen2.5-14B-Instruct parses every distinct line of the fresh streams (domain 1 with the
                   X6 prompt, domain 2 with the frozen R2-C prompt), greedy, at most 96 tokens.
  --phase build    CPU. Memories of A, D, E and the E' agreement; writes r7_n1_mem.jsonl and r7_n1_build.out.
  --phase gen      GPU (vLLM). B and C generations for one reader (greedy, at most 450 tokens) -> r7_n1_gen_<reader>.jsonl.
  --phase read     GPU. Margins of A, B, C, D, E for one reader -> r7_n1_read_<reader>.jsonl.
  --phase analyze  CPU (needs the Qwen tokenizer for the resource counts). H1, H2, secondary contrasts, families,
                   resources -> r7_n1_analyze.out.
"""
import argparse, json, os, random, re, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "domain2"))
from state_machine import POLICY_V2, filter_state, make_pair_v2, render_state, replay, replay_events  # noqa: E402

N1, N2 = 200, 100
BASE1, BASE2, T = 2026400000, 2026410000, 256
PFX = ""
if os.environ.get("R7_SMOKE"):                      # smoke test on other seeds, a few pairs, separate files
    N1, N2, BASE1, BASE2, PFX = 3, 3, 2026500000, 2026510000, "smoke_"
READERS = ("qwen2.5-14b", "phi-4")
ARMS = ("A", "B", "C", "D", "E")
WORDS = 250
GEN_MAX = 450
TRACE_D1 = (f"Before deciding, write a state trace of at most {WORDS} words. List the relevant events in time order with "
            "their event numbers. For each part of the state the request depends on (the service's own decision on the "
            "resource, its group memberships, the group grants on the resource and the resource's change window), write "
            "the chain of values it took and resolve its current value by the policy: source ranks, expiry, groups and "
            "windows. Then end with the decision line.")
SUMMARY_D1 = (f"Before deciding, write a summary of at most {WORDS} words of the information in the memory that is relevant "
              "to the request. Then end with the decision line.")
TRACE_D2 = (f"Before answering, write a state trace of at most {WORDS} words. List the relevant events in time order with "
            "their event numbers. For each part of the state the question depends on (the clearance, the payment and each "
            "hold), write the chain of values it took and resolve its current value by the rules, including expiry and "
            "reinstatement. Then end with the answer line.")
SUMMARY_D2 = (f"Before answering, write a summary of at most {WORDS} words of the information in the log that is relevant "
              "to the question. Then end with the answer line.")


# ------------------------------------------------------------------ streams
def streams_d1():
    out = []
    for i in range(N1):
        for s in make_pair_v2(BASE1 + T * 10000 + i, T):
            s["rev"] = replay_events(s)
            s["key"] = f"{s['pair']}-{s['truth']}"
            out.append(s)
    return out


def truth_d1(s):
    return replay([e for e in s["rev"] if e["t"] < T], s["p"], s["r"], T - 1)


def episodes_d2():
    import r5_domain2 as R
    import domain2 as D
    R.use_templates()
    out = []
    for i in range(N2):
        seed = BASE2 + T * 10000 + i
        for e in D.make_pair(seed, T):
            e["pair"] = seed
            out.append(e)
    return out


def dep_idx_d1(s):
    from r5_comparators import dependency_lines
    keep = set(dependency_lines(s, T))
    return [i for i, l in enumerate(s["lines"][:T]) if l in keep]


def dep_idx_d2(e):
    return [i for i, l in enumerate(e["lines"]) if re.search(rf"\b{re.escape(e['entity'])}\b", l)]


def snapshot(name):
    import glob
    org, n = name.split("/")
    for h in (os.path.expanduser("~/.cache/huggingface/hub"),):
        s = sorted(glob.glob(f"{h}/models--{org}--{n}/snapshots/*"))
        if s:
            return s[0]
    raise FileNotFoundError(name)


# ------------------------------------------------------------------ parse
def phase_parse():
    from vllm import LLM, SamplingParams
    import r5_domain2 as R
    from r2c_domain2 import PARSE_SYS as PS2, few_shot
    from x6_parse_filter import PARSE_SYS as PS1, EXAMPLES, content
    llm = LLM(model=snapshot("Qwen/Qwen2.5-14B-Instruct"), max_model_len=4096, seed=0, enforce_eager=True,
              gpu_memory_utilization=float(os.environ.get("VLLM_UTIL", "0.85")))
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0.0, max_tokens=96)
    chat = lambda sys_, user: tok.apply_chat_template([{"role": "system", "content": sys_}, {"role": "user", "content": user}],
                                                      tokenize=False, add_generation_prompt=True)
    shots1 = "\n\n".join(f"Line: {a}\nJSON: {json.dumps(b)}" for a, b in EXAMPLES)
    u1 = sorted({content(l) for s in streams_d1() for l in s["lines"]})
    shots2 = few_shot()
    u2 = sorted({l.split(": ", 1)[1] for e in episodes_d2() for l in e["lines"]})
    for name, uniq, prompts in (("d1", u1, [chat(PS1, shots1 + f"\n\nLine: {c}\nJSON:") for c in u1]),
                                ("d2", u2, [chat(PS2, shots2 + f"\n\nLine: Event 001: {c}\nJSON:") for c in u2])):
        res = llm.generate(prompts, sp, use_tqdm=False)
        with open(os.path.join(HERE, PFX + f"r7_n1_parse_{name}.jsonl"), "w") as f:
            for c, p, r in zip(uniq, prompts, res):
                m = re.search(r"\{.*\}", r.outputs[0].text, re.S)
                try:
                    u = json.loads(m.group(0)) if m else None
                except json.JSONDecodeError:
                    u = None
                f.write(json.dumps({"content": c, "parsed": u, "prompt_tokens": len(r.prompt_token_ids),
                                    "gen_tokens": len(r.outputs[0].token_ids)}) + "\n")
        print(name, "parsed", len(uniq), flush=True)


# ------------------------------------------------------------------ build
def typed_d1(i, tup):
    if tup[0] == "NONE":
        return f"Event {i + 1:03d}: NONE"
    k, src, p, r, g, vt = tup
    parts = [f"type={k}"] + [f"{n}={v}" for n, v in (("source", src), ("service", p), ("resource", r), ("group", g),
                                                    ("valid_through", None if vt is None else f"Event {vt:03d}")) if v is not None]
    return f"Event {i + 1:03d}: " + " ".join(parts)


def typed_d2(i, u):
    if not isinstance(u, dict) or u.get("op") in (None, "no_change"):
        return f"Event {i + 1:03d}: no_change"
    parts = [f"{k}={u[k]}" for k in ("op", "shipment", "source", "through") if u.get(k) is not None]
    return f"Event {i + 1:03d}: " + " ".join(parts)


def phase_build():
    import r5_domain2 as R
    import domain2 as D
    from r2c2_resolved import render_state_resolved
    from r5_comparators import dependency_lines
    from x6_parse_filter import content, parsed_tuple, query_at, to_event
    P1 = {r["content"]: parsed_tuple(r["parsed"]) for r in map(json.loads, open(os.path.join(HERE, PFX + "r7_n1_parse_d1.jsonl")))}
    P2 = {r["content"]: r["parsed"] for r in map(json.loads, open(os.path.join(HERE, PFX + "r7_n1_parse_d2.jsonl")))}
    rows, out = [], []
    exact = agree = n = 0
    for s in streams_d1():
        truth = truth_d1(s)
        idx = dep_idx_d1(s)
        dep = [s["lines"][i] for i in idx]
        exact += replay([e for e in s["rev"] if e["t"] < T and e["t"] in set(idx)], s["p"], s["r"], T - 1) == truth
        evs = [to_event(P1[content(l)], i) for i, l in enumerate(s["lines"][:T])]
        st = render_state(filter_state(evs, T - 1))
        rows_e = dependency_lines({"p": s["p"], "r": s["r"], "lines": st}, len(st))
        evq = [to_event(P1[content(s["lines"][i])], i) if i in set(idx) else {"t": i, "type": "FILLER"} for i in range(T)]
        stq = render_state(filter_state(evq, T - 1))
        agree += dependency_lines({"p": s["p"], "r": s["r"], "lines": stq}, len(stq)) == rows_e
        n += 1
        base = {"domain": 1, "pair": s["pair"], "stream": s["key"], "family": s.get("family"), "truth": truth,
                "query": query_at(s, T), "n_dep": len(dep)}
        rows.append(dict(base, arm="A", memory="\n".join(dep)))
        rows.append(dict(base, arm="D", memory="\n".join(typed_d1(i, P1[content(s["lines"][i])]) for i in idx)))
        rows.append(dict(base, arm="E", memory="\n".join(rows_e)))
    out.append(f"domain 1: {n} streams, replaying the dependency lines alone is exact in {exact}; E' rows equal E rows in {agree}")
    exact = agree = n = 0
    for e in episodes_d2():
        idx = dep_idx_d2(e)
        dep = [e["lines"][i] for i in idx]
        exact += ("YES" if D.may_ship(D.current_state([ev for ev in e["events"] if ev["t"] - 1 in set(idx)], T), e["entity"])
                  else "NO") == e["truth"]
        ups = R.parsed_updates(e, P2)
        st = render_state_resolved(R.run_updates(ups))
        rows_e = [l for l in st if re.search(rf"\b{re.escape(e['entity'])}\b", l)]
        upq = [u if u["event"] - 1 in set(idx) else {"event": u["event"], "op": "no_change", "shipment": None, "source": None,
                                                     "through": None} for u in ups]
        stq = render_state_resolved(R.run_updates(upq))
        agree += [l for l in stq if re.search(rf"\b{re.escape(e['entity'])}\b", l)] == rows_e
        n += 1
        base = {"domain": 2, "pair": e["pair"], "stream": f"{e['pair']}-{e['truth']}", "family": e["family"],
                "truth": e["truth"], "query": e["query"], "n_dep": len(dep)}
        rows.append(dict(base, arm="A", memory="\n".join(dep)))
        rows.append(dict(base, arm="D", memory="\n".join(typed_d2(i, P2.get(e["lines"][i].split(": ", 1)[1])) for i in idx)))
        rows.append(dict(base, arm="E", memory="\n".join(rows_e)))
    out.append(f"domain 2: {n} streams, replaying the shipment's lines alone is exact in {exact}; E' rows equal E rows in {agree}")
    with open(os.path.join(HERE, PFX + "r7_n1_mem.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    open(os.path.join(HERE, PFX + "r7_n1_build.out"), "w").write("\n".join(out) + "\n")
    print("\n".join(out))


# ------------------------------------------------------------------ prompts
def prompt_parts(tok, row, instr=None):
    """(prompt before any generated text, decision cue). Domain 1: the paper's reader prompt; domain 2: the R2-C prompt."""
    from x3_recursive_memory import PERSONAS, chat
    if row["domain"] == 1:
        user = ("Based strictly on the memory below, evaluate the incoming request.\n\n"
                f"[Memory]\n{row['memory'] or '(empty)'}\n\n[Request]\n{row['query']}")
        sys_ = PERSONAS["gov"] + "\n\n" + POLICY_V2
        cue = "DECISION:"
    else:
        from r2c_domain2 import READ_SYS
        tag = "[Current state]" if row["arm"] == "E" else "[Log]"
        user = f"{tag}\n{row['memory']}\n\n[Question]\n{row['query']} Answer YES or NO."
        sys_ = READ_SYS
        cue = "ANSWER:"
    if instr:
        user += "\n\n" + instr
    return chat(tok, sys_, user), cue


def instruction(row, arm):
    return {(1, "B"): TRACE_D1, (1, "C"): SUMMARY_D1, (2, "B"): TRACE_D2, (2, "C"): SUMMARY_D2}[(row["domain"], arm)]


def clip(text):
    m = re.search(r"\b(DECISION|ANSWER)\s*:", text)
    return (text[:m.start()] if m else text).rstrip(" *#_\n\t")      # drop the decision line and any markdown around it


def phase_gen(reader):
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    from x3_recursive_memory import READERS as RP
    path = RP[reader][0]
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    llm = LLM(model=path, max_model_len=8192, seed=0, enforce_eager=True, gpu_memory_utilization=float(os.environ.get("VLLM_UTIL", "0.85")))
    rows = [r for r in map(json.loads, open(os.path.join(HERE, PFX + "r7_n1_mem.jsonl"))) if r["arm"] == "A"]
    jobs = [(r, arm) for r in rows for arm in ("B", "C")]
    prompts = [prompt_parts(tok, r, instruction(r, arm))[0] for r, arm in jobs]
    res = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=GEN_MAX), use_tqdm=False)
    with open(os.path.join(HERE, PFX + f"r7_n1_gen_{reader}.jsonl"), "w") as f:
        for (r, arm), x in zip(jobs, res):
            f.write(json.dumps({"domain": r["domain"], "stream": r["stream"], "arm": arm, "text": x.outputs[0].text,
                                "gen_tokens": len(x.outputs[0].token_ids), "prompt_tokens": len(x.prompt_token_ids),
                                "finish": x.outputs[0].finish_reason}) + "\n")
    print(reader, "generated", len(jobs), flush=True)


def phase_read(reader):
    from x3_recursive_memory import load_reader, margins_batch, ntok
    tok, model, _ = load_reader(reader, "cuda:0")
    w_head = model.lm_head.weight.detach().float()
    mem = [json.loads(l) for l in open(os.path.join(HERE, PFX + "r7_n1_mem.jsonl"))]
    gen = {(g["domain"], g["stream"], g["arm"]): g for g in map(json.loads, open(os.path.join(HERE, PFX + f"r7_n1_gen_{reader}.jsonl")))}
    items = []
    for r in mem:
        p, cue = prompt_parts(tok, r)
        items.append((r, r["arm"], p + cue, ntok(tok, r["memory"])))
        if r["arm"] == "A":
            for arm in ("B", "C"):
                p, cue = prompt_parts(tok, r, instruction(r, arm))
                items.append((r, arm, p + clip(gen[(r["domain"], r["stream"], arm)]["text"]) + "\n" + cue, ntok(tok, r["memory"])))
    from r2c_domain2 import yes_no_margin
    out = []
    bs = int(os.environ.get("R7_READ_BS", "8"))
    for dom in (1, 2):
        its = [x for x in items if x[0]["domain"] == dom]
        if dom == 1:
            ms = margins_batch(tok, model, w_head, {k: tok.encode(v, add_special_tokens=False) for k, v in
                                                    (("APPROVE", " APPROVE"), ("DENY", " DENY"))}, [x[2] for x in its], "cuda:0", bs=bs)
        else:                                       # log p(" NO") - log p(" YES"), label lengths may differ
            ms = yes_no_margin(tok, model, w_head, [x[2] for x in its], "cuda:0", bs=bs)
        for (r, arm, _, mt), m in zip(its, ms):
            neg = "DENY" if dom == 1 else "NO"
            pos = "APPROVE" if dom == 1 else "YES"
            out.append({"domain": dom, "arm": arm, "pair": r["pair"], "stream": r["stream"], "family": r["family"],
                        "truth": r["truth"], "margin": m, "decision": neg if m > 0 else pos, "memory_tokens": mt})
    with open(os.path.join(HERE, PFX + f"r7_n1_read_{reader}.jsonl"), "w") as f:
        for o in out:
            f.write(json.dumps(o) + "\n")
    print(reader, "read", len(out), flush=True)


# ------------------------------------------------------------------ analysis
def phase_analyze():
    from r2_analyze import bal
    from r5_comparators_analyze import boot
    from fmt import r2
    sg = lambda x: ("+" if x >= 0 else "") + r2(x)
    out = [open(os.path.join(HERE, PFX + "r7_n1_build.out")).read().strip()]
    R = {rd: [json.loads(l) for l in open(os.path.join(HERE, PFX + f"r7_n1_read_{rd}.jsonl"))] for rd in READERS}
    get = lambda rd, dom, arm: [r for r in R[rd] if r["domain"] == dom and r["arm"] == arm]
    for rd in READERS:
        for dom in (1, 2):
            out.append(f"{rd} domain {dom}: " + ", ".join(f"{a} {r2(bal(get(rd, dom, a)))}" for a in ARMS))

    def family(name, a, b):
        tests = []
        for rd in READERS:
            for dom in (1, 2):
                d = boot(get(rd, dom, a), get(rd, dom, b))
                tests.append((f"{rd} d{dom}", d))
        order = sorted(range(4), key=lambda i: tests[i][1][3])
        stop, res = False, {}
        for k, i in enumerate(order):
            lab, d = tests[i]
            thr = 0.05 / (4 - k)
            rej = (not stop) and d[3] <= thr
            stop = stop or not rej
            res[lab] = (d, rej, rej and d[0] >= 0.05 and d[1] > 0)
        out.append(f"{name}: {a} - {b}, registered margin 0.05 with lower limit above 0, Holm over 4")
        for lab, (d, rej, ok) in res.items():
            out.append(f"  {lab}: {sg(d[0])} [{r2(d[1])}, {r2(d[2])}] p={d[3]:.4f} Holm reject {rej} -> {'holds' if ok else 'does not hold'}")
    family("H1 (resolution beyond selection)", "E", "A")
    family("H2 (rule against answer-time model resolution)", "E", "B")
    out.append("secondary, report only")
    for a, b, what in (("B", "C", "structure of the answer-time trace"), ("D", "A", "typing alone"), ("E", "D", "applying the rule beyond typing"),
                       ("B", "A", "answer-time trace against raw lines"), ("C", "A", "summary against raw lines")):
        for rd in READERS:
            for dom in (1, 2):
                d = boot(get(rd, dom, a), get(rd, dom, b))
                out.append(f"  {a} - {b} ({what}) {rd} d{dom}: {sg(d[0])} [{r2(d[1])}, {r2(d[2])}]")
    out.append("by family, report only")
    for rd in READERS:
        for dom in (1, 2):
            fams = sorted({r["family"] for r in get(rd, dom, "A")})
            for fam in fams:
                cells = " ".join(f"{a} {r2(bal([r for r in get(rd, dom, a) if r['family'] == fam]))}" for a in ARMS)
                out.append(f"  {rd} d{dom} {fam}: {cells}")
    # resources, Qwen2.5-14B tokens
    from transformers import AutoTokenizer
    from x3_recursive_memory import READERS as RP
    tok = AutoTokenizer.from_pretrained(RP["qwen2.5-14b"][0], local_files_only=True)
    G = {(g["domain"], g["stream"], g["arm"]): g for g in map(json.loads, open(os.path.join(HERE, PFX + "r7_n1_gen_qwen2.5-14b.jsonl")))}
    mem = [json.loads(l) for l in open(os.path.join(HERE, PFX + "r7_n1_mem.jsonl"))]
    for dom, name in ((1, "d1"), (2, "d2")):
        P = {r["content"]: r for r in map(json.loads, open(os.path.join(HERE, PFX + f"r7_n1_parse_{name}.jsonl")))}
        per = lambda c: P[c]["prompt_tokens"] + P[c]["gen_tokens"]
        ms = [m for m in mem if m["domain"] == dom]
        A = [m for m in ms if m["arm"] == "A"]
        n_dep = sum(m["n_dep"] for m in A) / len(A)
        mean_parse = sum(per(c) for c in P) / len(P)
        read = {a: sum(len(tok.encode(m["memory"], add_special_tokens=False)) for m in ms if m["arm"] == a) / len(A) for a in ("A", "D", "E")}
        gen = {a: sum(G[(dom, m["stream"], a)]["gen_tokens"] for m in A) / len(A) for a in ("B", "C")}
        out.append(f"resources d{dom} (per stream, Qwen2.5-14B tokens): dependency lines {n_dep:.1f}, parser call {mean_parse:.0f} tokens; "
                   f"write-time filter {T} parser calls ({T * mean_parse:,.0f} tokens); E' {n_dep:.1f} calls per query "
                   f"({n_dep * mean_parse:,.0f} tokens); read tokens A {read['A']:.0f}, D {read['D']:.0f}, E {read['E']:.0f}; "
                   f"generated per query B {gen['B']:.0f}, C {gen['C']:.0f} (plus their read of A)")
        for Q in (1, 10, 100, 1000):
            cost = {"A": Q * read["A"], "B": Q * (read["A"] + gen["B"]), "C": Q * (read["A"] + gen["C"]),
                    "E write-time": T * mean_parse + Q * read["E"], "E' query-time": Q * (n_dep * mean_parse + read["E"])}
            out.append(f"  Q={Q}: " + ", ".join(f"{k} {v:,.0f}" for k, v in cost.items()))
    open(os.path.join(HERE, PFX + "r7_n1_analyze.out"), "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["parse", "build", "gen", "read", "analyze"], required=True)
    ap.add_argument("--reader", default="")
    a = ap.parse_args()
    {"parse": phase_parse, "build": phase_build, "gen": lambda: phase_gen(a.reader), "read": lambda: phase_read(a.reader),
     "analyze": phase_analyze}[a.phase]()
