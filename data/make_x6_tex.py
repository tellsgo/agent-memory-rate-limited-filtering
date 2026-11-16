#!/usr/bin/env python3
"""paper/generated/x6_table.tex: buildable filter against baselines, full background, both readers."""
import json, os
from collections import defaultdict
from fmt import r2
from x3_analyze import bal
HERE = os.path.dirname(os.path.abspath(__file__))
load = lambda f: [json.loads(l) for l in open(os.path.join(HERE, f))]
rows = []
for rd, name in (("qwen2.5-14b", "Qwen2.5-14B"), ("phi-4", "phi-4"), ("qwen2.5-32b", "Qwen2.5-32B")):
    G = defaultdict(list)
    for r in load(f"x6_read_{rd}.jsonl"):
        if r["design"] == "full":
            G[(r["method"], str(r["B"]), r["t"])].append(r)
    if rd == "qwen2.5-14b":
        for r in load("x3v2_14b_llm.jsonl"):
            if r["method"] == "ledger" and r["B"] == 256:
                G[("ledger", "256", r["t"])].append(r)
    for ph, meth in (("apply", "llm_apply"), ("mem0", "mem0_ops")):
        if os.path.exists(os.path.join(HERE, f"r2_{ph}_{rd}.jsonl")):
            for r in load(f"r2_{ph}_{rd}.jsonl"):
                G[(meth, "None", r["t"])].append(r)
    unp = lambda rs: [dict(r, decision=r["decision"] or "UNPARSED") for r in rs]
    cot = {T: unp([r for r in load(f"x7_{rd}.jsonl") if r["T"] == T]) for T in (64, 256)} if os.path.exists(os.path.join(HERE, f"x7_{rd}.jsonl")) else {}
    cot4 = unp(load(f"r2_cot_{rd}.jsonl")) if os.path.exists(os.path.join(HERE, f"r2_cot_{rd}.jsonl")) else []
    def cell(m, B, t):
        return r2(bal(G[(m, B, t)])) if G[(m, B, t)] else "--"
    block = [
        ("parsed events + transition rule", [cell("parse_filter:qwen2.5-14b", "None", t) for t in (32, 64, 128, 256)]),
        ("replayed state (oracle)", [cell("oracle_state", "None", t) for t in (32, 64, 128, 256)]),
        ("LLM ledger, $B=256$", [cell("ledger", "256", t) for t in (32, 64, 128, 256)]),
        ("LLM applies the parsed updates", [cell("llm_apply", "None", t) for t in (32, 64, 128, 256)]),
        ("LLM fact store (add, update, delete), $B=512$", [cell("mem0_ops", "None", t) for t in (32, 64, 128, 256)]),
        ("BM25 retrieval, $B=512$", [cell("rag_bm25", "512", t) for t in (32, 64, 128, 256)]),
        ("recency $+$ relevance, $B=512$", [cell("recency_relevance", "512", t) for t in (32, 64, 128, 256)]),
        ("full log with reasoning trace, 768 tokens$^\\ast$", ["--", r2(bal(cot[64])), "--", r2(bal(cot[256]))] if cot else None),
        ("full log with reasoning trace, 4096 tokens", ["--", "--", "--", r2(bal(cot4))] if cot4 else None),
    ]
    rows.append((name, [(m, c) for m, c in block if c is not None and any(x != "--" for x in c)]))
out = ["\\begin{tabular}{llcccc}", "\\toprule", "reader & memory & $t=32$ & 64 & 128 & 256\\\\", "\\midrule"]
for name, block in rows:
    for i, (m, cells) in enumerate(block):
        out.append(f"{name if i == 0 else ''} & {m} & " + " & ".join(cells) + "\\\\")
    out.append("\\midrule")
out[-1] = "\\bottomrule"
out.append("\\end{tabular}")
open(os.path.join(HERE, "..", "paper", "generated", "x6_table.tex"), "w").write("\n".join(out) + "\n")
print("\n".join(out))
