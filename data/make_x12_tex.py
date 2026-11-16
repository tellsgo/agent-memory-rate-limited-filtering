#!/usr/bin/env python3
"""paper/generated/x12_{auc,bal}.tex: reader curves (log solid, state dashed) for three readers."""
import json, os
from collections import defaultdict
from fmt import r2
from x12_analyze import auc, bal
HERE = os.path.dirname(os.path.abspath(__file__))
G = defaultdict(list)
for f in ("x12_qwen14b.jsonl", "x12_qwen7b.jsonl", "x12_phi4.jsonl"):
    for l in open(os.path.join(HERE, f)):
        r = json.loads(l)
        G[(r["reader"], r["cond"], r["T"])].append(r)
Ts = [8, 16, 32, 64, 128, 256]
series = [("qwen2.5-14b", "raw_gov", "mark=*,blue"), ("qwen2.5-7b", "raw_gov", "mark=square*,red"),
          ("qwen2.5-14b", "state_gov", "mark=*,blue,dashed"), ("qwen2.5-7b", "state_gov", "mark=square*,red,dashed"),
          ("phi-4", "raw_gov", "mark=triangle*,teal"), ("phi-4", "state_gov", "mark=triangle*,teal,dashed")]
for name, fn in (("auc", lambda rs: auc([r["margin"] for r in rs if r["truth"] == "DENY"], [r["margin"] for r in rs if r["truth"] == "APPROVE"])),
                 ("bal", bal)):
    lines = []
    for rd, cond, st in series:
        pts = " ".join(f"({T},{r2(fn(G[(rd, cond, T)]))})" for T in Ts)
        lines.append(f"\\addplot[{st}] coordinates {{{pts}}};")
    lines.append("\\legend{Qwen 14B log,Qwen 7B log,Qwen 14B state,Qwen 7B state,phi-4 log,phi-4 state}")
    open(os.path.join(HERE, "..", "paper", "generated", f"x12_{name}.tex"), "w").write("\n".join(lines) + "\n")
    print(name); print("\n".join(lines[:-1]))
