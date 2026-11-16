#!/usr/bin/env python3
"""Resource ledger (blind review R3-N3, editor ED-1): what each memory stores, spends when an event arrives and shows
the reader, on the R5-3 streams at t=256, and how archive and state grow with T on the R4-1/R6-3 streams.
Tokens are Qwen2.5-14B tokens. Writes r6_ledger.json and ../paper/generated/tab_ledger.tex. Runs where the
tokenizer and the Mem0 stores are (MEM0_STORE)."""
import json, os, sqlite3, statistics as st, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from r5_comparators import streams  # noqa: E402
from state_machine import filter_state, make_pair_v2, render_state  # noqa: E402
from x6_parse_filter import content, to_event  # noqa: E402

TOK_PATH = os.environ.get("QWEN_TOK", "")


def tokenizer():
    from transformers import AutoTokenizer
    from x3_recursive_memory import READERS
    return AutoTokenizer.from_pretrained(TOK_PATH or READERS["qwen2.5-14b"][0], local_files_only=True)


def B(s):
    return len(s.encode())


def main():
    tok = tokenizer()
    nt = lambda s: len(tok.encode(s, add_special_tokens=False))
    P = {r["content"]: tuple(r["parsed"]) for r in map(json.loads, open(os.path.join(HERE, "x6_parse_qwen2.5-14b.jsonl")))}
    ss = streams()
    log_text = ["\n".join(s["lines"][:256]) for s in ss]
    state = []
    for s in ss:
        evs = [to_event(P[content(l)], i) for i, l in enumerate(s["lines"][:256])]
        state.append("\n".join(render_state(filter_state(evs, 255))))
    mem = lambda m: [json.loads(l) for l in open(os.path.join(HERE, f"r5_mem_{m}.jsonl")) if int(json.loads(l)["t"]) == 256]
    root = os.environ.get("MEM0_STORE", "mem0_store")
    m0 = []
    for s in ss:
        rows = sqlite3.connect(os.path.join(root, s["key"].replace("-", "_") + ".db")).execute("select new_memory from history").fetchall()
        m0.append(sum(B(r[0] or "") for r in rows) + 384 * 4 * len(rows))
    r6 = {}
    p6 = os.path.join(HERE, "r6_read_qwen2.5-14b.jsonl")
    if os.path.exists(p6):
        for r in map(json.loads, open(p6)):
            if r["t"] == 256 and r["B"] is None:
                r6.setdefault(r["method"], []).append(r["memory_tokens"])
    comp = [json.loads(l) for l in open(os.path.join(HERE, "r5_comp_qwen2.5-14b.jsonl"))]
    rt = lambda m: st.mean(r["memory_tokens"] for r in comp if r["method"] == m and int(r["t"]) == 256 and r["B"] is None)
    sm = {v: mem(f"statemem_{v}") for v in ("native", "policy")}
    led = {
        "full log": dict(store=st.mean(map(B, log_text)), calls=0, wtok=0, read=st.mean(map(nt, log_text))),
        "dependency retriever": dict(store=st.mean(map(B, log_text)), calls=0, wtok=0, read=rt("retriever")),
        # parser call: 438 prompt tokens (410 a shared prefix) + 18 generated, measured on the X6 parse (master, Writing and reading)
        "parsed filter": dict(store=st.mean(map(B, state)), calls=256, wtok=256 * (438 + 18), wtok_cached=256 * (28 + 18),
                              read=st.mean(map(nt, state))),
        "parsed filter, query rows": dict(store=st.mean(map(B, state)), calls=256, wtok=256 * (438 + 18),
                                          wtok_cached=256 * (28 + 18), read=st.mean(r6["filter_proj"]) if "filter_proj" in r6 else None),
        "StateMem, native": dict(store=st.mean(B(r["memory"]) for r in sm["native"]), calls=st.mean(int(r["write_calls"]) for r in sm["native"]),
                                 wtok=st.mean(int(r["write_tokens"]) for r in sm["native"]), read=rt("statemem_native")),
        "StateMem, policy": dict(store=st.mean(B(r["memory"]) for r in sm["policy"]), calls=st.mean(int(r["write_calls"]) for r in sm["policy"]),
                                 wtok=st.mean(int(r["write_tokens"]) for r in sm["policy"]), read=rt("statemem_policy")),
        # Mem0: one call per add (20,480 POSTs in r5_mem0_server.log); tokens estimated by integrating the server's 10 s
        # prompt/generation throughput lines over the run, divided by 80 streams
        "Mem0": dict(store=st.mean(m0), calls=256, wtok=158946 + 24767, read=rt("mem0")),
    }
    # growth with T: archive against parsed state, R4-1 (256, 1024) and R6-3 (2048) streams
    grow = {}
    R4P = {r["content"]: tuple(r["parsed"]) for r in map(json.loads, open(os.path.join(HERE, "r4_parse_qwen2.5-14b.jsonl")))}
    for T, base in ((256, 2026210000), (1024, 2026210000), (2048, 2026250000)):
        a, s_, k = [], [], []
        for i in range(20):
            for ep in make_pair_v2(base + T * 10000 + i, T):
                if not all(content(l) in R4P for l in ep["lines"]):
                    continue
                evs = [to_event(R4P[content(l)], j) for j, l in enumerate(ep["lines"])]
                stx = render_state(filter_state(evs, T - 1))
                a.append(nt("\n".join(ep["lines"])))
                s_.append(nt("\n".join(stx)))
                k.append(len(stx))
        if a:
            grow[T] = dict(log_tokens=st.mean(a), state_tokens=st.mean(s_), state_lines=st.mean(k), n=len(a))
    json.dump({"t256": led, "growth": grow}, open(os.path.join(HERE, "r6_ledger.json"), "w"), indent=1)
    for k, v in led.items():
        print(k, {a: (round(b) if isinstance(b, (int, float)) else b) for a, b in v.items()})
    print("growth", grow)
    kb = lambda x: f"{x / 1000:.1f}"
    fmt = lambda x: "--" if x is None else f"{x:,.0f}".replace(",", "{,}")
    lines = [r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
             r"memory & store (kB) & calls & write tokens & read tokens \\", r"\midrule"]
    for k, v in led.items():
        w = fmt(v["wtok"]) + (f" ({fmt(v['wtok_cached'])})" if "wtok_cached" in v else "")
        lines.append(f"{k} & {kb(v['store'])} & {fmt(v['calls'])} & {w} & {fmt(v['read'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    gen = os.path.join(HERE, "..", "paper", "generated")
    os.makedirs(gen, exist_ok=True)
    open(os.path.join(gen, "tab_ledger.tex"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
