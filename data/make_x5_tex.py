#!/usr/bin/env python3
"""Generate paper/generated/x5_*.tex from x5_exact.jsonl (and x5_llm.jsonl when present)."""
import json, os, sys
from fmt import r2
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "paper", "generated")
RULES = [("bm25", "BM25"), ("key_mention", "key mention"), ("salience", "salience"), ("grad", "gradient"),
         ("recency", "recency"), ("key_recency", "key recency"), ("latest_applied", "last applied")]


def load(path, reader):
    err = defaultdict(list)
    if not os.path.exists(path):
        return err
    for l in open(path):
        r = json.loads(l)
        if r.get("reader") == reader:
            err[(r["family"], r["rule"], r["b"])].append(r["decision"] != r["truth"])
    return err


def rate(v):
    return sum(v) / len(v) if v else None


def main():
    ex = load(os.path.join(HERE, "x5_llm.jsonl"), "exact")
    ll = load(os.path.join(HERE, "x5_llm.jsonl"), "llm")
    ms = [5, 9, 17, 33, 65]
    bps = [2, 4, 8, 16]
    # figure: exact reader, b=2, chain family
    with open(os.path.join(OUT, "x5_chain_plot.tex"), "w") as f:
        styles = {"bm25": "blue,mark=*", "key_mention": "teal,mark=square*", "salience": "orange,mark=triangle*",
                  "recency": "gray,mark=diamond*", "key_recency": "red,mark=x", "latest_applied": "black,mark=o"}
        for rule, st in styles.items():
            pts = " ".join(f"({m},{rate(ex[(f'chain{m}', rule, 2)]):.3f})" for m in ms if ex[(f'chain{m}', rule, 2)])
            f.write(f"\\addplot[{st}] coordinates {{{pts}}};\n")
        f.write("\\addplot[dashed,thick,domain=5:65,samples=60] {max(0,0.5-2/(x-1))};\n")
        f.write("\\legend{BM25,key mention,salience,recency,key recency,last applied,lower bound}\n")
    # table: both families, both budgets, exact and (if present) llm
    def cell(err, fam, rule, b):
        v = rate(err[(fam, rule, b)])
        return "--" if v is None else r2(v)
    for reader, err in (("exact", ex), ("llm", ll)):
        if not err:
            continue
        with open(os.path.join(OUT, f"x5_table_{reader}.tex"), "w") as f:
            cols = [f"chain{m}" for m in ms] + [f"blocked{b}" for b in bps]
            f.write("\\begin{tabular}{ll" + "c" * len(cols) + "}\n\\toprule\n")
            f.write(" & & \\multicolumn{5}{c}{chain length $m$} & \\multicolumn{4}{c}{blocked events $b'$}\\\\\n")
            f.write("\\cmidrule(lr){3-7}\\cmidrule(lr){8-11}\n")
            f.write("$b$ & rule & " + " & ".join(str(m) for m in ms) + " & " + " & ".join(str(b) for b in bps) + "\\\\\n\\midrule\n")
            for b in (2, 4):
                for i, (rule, name) in enumerate(RULES):
                    if not any(err[(c, rule, b)] for c in cols):
                        continue
                    lead = f"\\multirow{{7}}{{*}}{{{b}}}" if i == 0 else ""
                    f.write(f"{lead} & {name} & " + " & ".join(cell(err, c, rule, b) for c in cols) + "\\\\\n")
                f.write(f" & bound & " + " & ".join(r2(max(0, 0.5 - b / (m - 1))) for m in ms) + " & " + " & ".join("--" for _ in bps) + "\\\\\n")
                if b == 2:
                    f.write("\\midrule\n")
            f.write("\\bottomrule\n\\end{tabular}\n")
    print("written", os.listdir(OUT))


if __name__ == "__main__":
    main()
