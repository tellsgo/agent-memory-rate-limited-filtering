"""Table of R10 (from r10_analyze.out, which already rounds half up)."""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "paper", "generated", "tab_r10.tex")
txt = open(os.path.join(HERE, "r10_analyze.out")).read()
NAME = {"qwen2.5-14b": "Qwen2.5-14B", "phi-4": "phi-4"}
ARMS = ("RULE", "INJ_K", "FREE_K", "INJ_KT", "INJ_F", "FREE_F")
HEAD = ("RULE", "INJ$_K$", "FREE$_K$", "INJ$_{KT}$", "INJ$_F$", "FREE$_F$")
CI = r"([+-]?[\d.]+) \[([+-]?[\d.]+), ([+-]?[\d.]+)\]"


def ci(m):
    return f"${m[0] if m[0].startswith('-') else '+' + m[0].lstrip('+')}$ $[{m[1]},{m[2]}]$"


lines = [r"\begin{tabular}{l l cccccc cc c}", r"\toprule",
         r" & & \multicolumn{6}{c}{balanced accuracy} & \multicolumn{2}{c}{registered} & \\",
         r"\cmidrule(lr){3-8}\cmidrule(lr){9-10}",
         r"writer & endpoint & " + " & ".join(HEAD) + r" & RULE$-$FREE$_F$ & FREE$_F-$FREE$_K$ & case \\", r"\midrule"]
for r in ("qwen2.5-14b", "phi-4"):
    row = re.search(rf"^  {re.escape(r)}: (RULE .*)$", txt, re.M).group(1)
    case = re.search(rf"^  {re.escape(r)} E10 .*case (\w)", txt, re.M).group(1)
    for k, ep in enumerate(("reader", "state")):
        vals = dict(re.findall(r"(\w+) ([\d.]+) \(([\d.]+)\)", row) and
                    [(a, (v, s)) for a, v, s in re.findall(r"(\w+) ([\d.]+) \(([\d.]+)\)", row)])
        h1 = re.search(rf"^  {re.escape(r)} H10-1 {ep}: {CI}", txt, re.M).groups()
        h2 = re.search(rf"^  {re.escape(r)} H10-2 {ep}: {CI}", txt, re.M).groups()
        lines.append(f"{NAME[r] if k == 0 else ''} & {'writer reads' if ep == 'reader' else 'state answer'} & " +
                     " & ".join(vals[a][k] for a in ARMS) + f" & {ci(h1)} & {ci(h2)} & {case if k == 0 else ''} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
open(OUT, "w").write("\n".join(lines) + "\n")
print(open(OUT).read())
