#!/usr/bin/env python3
"""paper/generated/r3_budget_plot.tex: observed BA of budgeted filters (points) against the frozen prediction
(lines) at K=15, T=256, both eviction rules and both readers."""
import json, os
from r3_budget_analyze import F, bal, table
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
tab = table()
out = []
styles = {("qwen2.5-14b", "random"): "blue", ("qwen2.5-14b", "recency"): "blue", ("phi-4", "random"): "red", ("phi-4", "recency"): "red"}
names = {"qwen2.5-14b": "Qwen2.5-14B", "phi-4": "phi-4"}
for pol in ("random", "recency"):
    lines = []
    for reader in ("qwen2.5-14b", "phi-4"):
        G = defaultdict(list)
        for r in map(json.loads, open(os.path.join(HERE, f"r3_budget_{reader}.jsonl"))):
            if r["K"] == 15 and r["T"] == 256 and r["policy"] == pol:
                G[r["f"]].append(r)
        obs = [bal(G[f]) for f in F]
        pred = [0.5 + (obs[-1] - 0.5) * x for x in tab[(pol, 15, 256)]]
        c = styles[(reader, pol)]
        lines.append(f"\\addplot[{c},thick,no markers] coordinates {{" + " ".join(f"({f},{p:.4f})" for f, p in zip(F, pred)) + "};")
        lines.append(f"\\addlegendentry{{{names[reader]} predicted}}")
        lines.append(f"\\addplot[{c},only marks,mark=*,mark size=1.5pt] coordinates {{" + " ".join(f"({f},{o:.4f})" for f, o in zip(F, obs)) + "};")
        lines.append(f"\\addlegendentry{{{names[reader]} observed}}")
    out.append((pol, "\n".join(lines)))
with open(os.path.join(HERE, "..", "paper", "generated", "r3_budget_plot.tex"), "w") as f:
    for pol, body in out:
        f.write(f"% {pol}\n\\begin{{tikzpicture}}\\begin{{axis}}[width=0.47\\linewidth,height=5cm,xlabel={{fraction of state lines kept}},"
                f"ylabel={{balanced accuracy}},ymin=0.45,ymax=1.0,grid=major,title={{\\small {pol} eviction}},"
                f"legend style={{font=\\scriptsize,at={{(0.5,-0.32)}},anchor=north,legend columns=2}}]\n{body}\n\\end{{axis}}\\end{{tikzpicture}}\n"
                + ("\\hfill\n" if pol == "random" else ""))
print("written")
