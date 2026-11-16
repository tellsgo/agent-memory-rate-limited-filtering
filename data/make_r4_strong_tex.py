#!/usr/bin/env python3
"""paper/generated/r4_strong_table.tex: R4-1 stronger readers at 256 and 1024 events, balanced accuracy of the full log,
the state from parsed events and the true state, the filter-minus-log difference with its pair-bootstrap interval,
and tokens per decision (prompt + generated)."""
import glob, json, os
from statistics import mean

from fmt import r2
from r2_analyze import bal, boot_diff

HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = {"gpt-oss-120b": "gpt-oss-120b", "qwen2.5-72b-awq": "Qwen2.5-72B (4-bit)", "qwq-32b": "QwQ-32B"}


def load(m):
    fs = [os.path.join(HERE, f"r4_read_{m}.jsonl")] + sorted(glob.glob(os.path.join(HERE, f"r4_read_{m}.rows*.jsonl")))
    return [json.loads(l) for f in fs if os.path.exists(f) for l in open(f)]


def tok(rs):
    return f"{mean(r['prompt_tokens'] for r in rs) / 1000:.1f}k + {mean(r['gen_tokens'] for r in rs) / 1000:.1f}k"


rows = []
for m, name in NAMES.items():
    R = load(m)
    for T in (256, 1024):
        g = {c: [r for r in R if r["t"] == T and r["cond"] == c] for c in ("log", "filter", "state")}
        if not g["log"]:
            continue
        cell = lambda c: r2(bal(g[c])) if g[c] else "--"
        if g["filter"]:
            d, lo, hi = boot_diff(g["filter"], g["log"])
            diff = f"${'+' if d >= 0 else ''}{r2(d)}$ $[{r2(lo)},{r2(hi)}]$"
        else:
            diff = "--"
        rows.append(f"{name} & {T:,} & {cell('log')} & {cell('filter')} & {cell('state')} & {diff} & {tok(g['log'])} & "
                    f"{tok(g['filter']) if g['filter'] else '--'} \\\\".replace("1,024", "1{,}024"))
with open(os.path.join(HERE, "..", "paper", "generated", "r4_strong_table.tex"), "w") as f:
    f.write("\\begin{tabular}{lrccccrr}\n\\toprule\n"
            " & & full & state from & true & state from parsed & \\multicolumn{2}{c}{tokens per decision}\\\\\n"
            "reader & events & log & parsed events & state & minus log & full log & parsed state \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
print("\n".join(rows))
