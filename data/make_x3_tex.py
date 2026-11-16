#!/usr/bin/env python3
"""paper/generated/x3_table.tex from x3v2_14b_*.jsonl."""
import json, os
from fmt import r2
from collections import defaultdict
from x3_analyze import bal
HERE = os.path.dirname(os.path.abspath(__file__))
rows = [json.loads(l) for f in ("x3v2_14b_cheap.jsonl", "x3v2_14b_llm.jsonl", "x3v2_14b_grad.jsonl") for l in open(os.path.join(HERE, f))]
G = defaultdict(list)
for r in rows:
    G[(r["method"], r["B"], r["t"])].append(r)
ts = [32, 64, 128, 256]
order = [("oracle_state", "replayed state (oracle)"), ("latest_applied", "last applied event per component"),
         ("latest_key", "last event per key"), ("ledger", "LLM state ledger"), ("summary", "LLM summary"),
         ("grad_evict", "gradient eviction"), ("rag_bm25", "BM25 retrieval (full archive)"), ("recency", "recency")]
out = ["\\begin{tabular}{l" + "cccc" * 3 + "}", "\\toprule",
       " & \\multicolumn{4}{c}{$B=128$} & \\multicolumn{4}{c}{$B=256$} & \\multicolumn{4}{c}{$B=512$}\\\\",
       "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\\cmidrule(lr){10-13}",
       "memory & " + " & ".join(str(t) for _ in range(3) for t in ts) + "\\\\", "\\midrule"]
for m, name in order:
    cells = []
    for B in (128, 256, 512):
        for t in ts:
            cells.append(r2(bal(G[(m, B, t)])) if (m, B, t) in G else "--")
    out.append(f"{name} & " + " & ".join(cells) + "\\\\")
out.append("\\midrule")
out.append("full log, no budget & \\multicolumn{12}{l}{" + "\\quad ".join(f"$t={t}$: {r2(bal(G[('raw_full', None, t)]))}" for t in ts) + "}\\\\")
out += ["\\bottomrule", "\\end{tabular}"]
open(os.path.join(HERE, "..", "paper", "generated", "x3_table.tex"), "w").write("\n".join(out) + "\n")
print("ok")
