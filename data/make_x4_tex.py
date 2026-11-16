#!/usr/bin/env python3
"""paper/generated/x4c_plot.tex: oracle_state balanced accuracy against budget, per K, T=64 solid, T=256 dashed."""
import glob, json, os, re
from fmt import r2
from collections import defaultdict
from x3_analyze import bal
HERE = os.path.dirname(os.path.abspath(__file__))
G = defaultdict(list)
for path in glob.glob(os.path.join(HERE, "x4c_K*_T*.jsonl")):
    K, T = map(int, re.search(r"K(\d+)_T(\d+)", path).groups())
    for l in open(path):
        r = json.loads(l)
        if r["t"] == T:
            G[(r["method"], K, T, r["B"])].append(r)
colors = {1: "blue", 3: "teal", 7: "orange", 15: "red"}
lines = []
for method, out in (("oracle_state", "x4c_plot_state.tex"), ("recency", "x4c_plot_recency.tex")):
    lines = []
    for K in (1, 3, 7, 15):
        for T, style in ((64, ""), (256, ",dashed")):
            pts = " ".join(f"({B},{r2(bal(G[(method, K, T, B)]),3)})" for B in (32, 64, 128, 256, 512))
            lines.append(f"\\addplot[{colors[K]},mark=*{style}] coordinates {{{pts}}};")
    open(os.path.join(HERE, "..", "paper", "generated", out), "w").write("\n".join(lines) + "\n")
print("ok")
