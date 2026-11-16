#!/usr/bin/env python3
"""Table of R7-1 (selection and resolution on fresh streams) from the per-row reading files: balanced accuracy of the five
arms and the registered and secondary contrasts with 95% pair-bootstrap intervals (2000 resamples, seed 0).
Writes ../paper/generated/tab_r7_n1.tex."""
import json, os

from fmt import r2
from r2_analyze import bal
from r5_comparators_analyze import boot

HERE = os.path.dirname(os.path.abspath(__file__))
READERS = ("qwen2.5-14b", "phi-4")
COLS = [(1, "qwen2.5-14b"), (1, "phi-4"), (2, "qwen2.5-14b"), (2, "phi-4")]
NAMES = {"A": "A. raw dependency lines", "B": "B. answer-time state trace", "C": "C. answer-time summary",
         "D": "D. typed updates, not resolved", "E": "E. resolved state rows"}


def main():
    R = {rd: [json.loads(l) for l in open(os.path.join(HERE, f"r7_n1_read_{rd}.jsonl"))] for rd in READERS}
    get = lambda dom, rd, arm: [r for r in R[rd] if r["domain"] == dom and r["arm"] == arm]
    sg = lambda x: ("+" if x >= 0 else "") + r2(x)
    L = [r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
         r" & \multicolumn{2}{c}{first domain, 200 pairs} & \multicolumn{2}{c}{second domain, 100 pairs} \\",
         r"\cmidrule(lr){2-3}\cmidrule(l){4-5}", r"arm & Qwen2.5-14B & phi-4 & Qwen2.5-14B & phi-4 \\", r"\midrule"]
    for a in "ABCDE":
        L.append(NAMES[a] + " & " + " & ".join(r2(bal(get(d, rd, a))) for d, rd in COLS) + r" \\")
    L.append(r"\midrule")
    for a, b, lab in (("E", "A", r"E $-$ A (H1, registered)"), ("E", "B", r"E $-$ B (H2, registered)"),
                      ("E", "D", r"E $-$ D, applying the rule"), ("D", "A", r"D $-$ A, typing alone"),
                      ("B", "C", r"B $-$ C, structure of the trace")):
        cells = []
        for d, rd in COLS:
            x = boot(get(d, rd, a), get(d, rd, b))
            cells.append(f"${sg(x[0])}$ $[{r2(x[1])},{r2(x[2])}]$")
        L.append(lab + " & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(HERE, "..", "paper", "generated", "tab_r7_n1.tex"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
