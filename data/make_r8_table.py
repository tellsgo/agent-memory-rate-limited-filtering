"""Table of R8-1 and R8-2 (from r8_analyze_halfup.out, the frozen analysis reprinted with half-up rounding)."""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "paper", "generated", "tab_r8.tex")
txt = open(os.path.join(HERE, "r8_analyze_halfup.out")).read()
CELLS = (("qwen2.5-14b", "1"), ("phi-4", "1"), ("qwen2.5-14b", "2"), ("phi-4", "2"))
NAME = {"qwen2.5-14b": "Qwen2.5-14B", "phi-4": "phi-4"}
NUM = r"([+-]?[\d.]+)"
CI = r"\[([+-]?[\d.]+), ([+-]?[\d.]+)\]"


def ci(m):
    return f"${m[0] if m[0].startswith('-') else '+' + m[0].lstrip('+')}$ $[{m[1]},{m[2]}]$"


lines = [r"\begin{tabular}{ll ccccc c ccc c}", r"\toprule",
         r" & & \multicolumn{6}{c}{R8-1 reading (BA)} & \multicolumn{4}{c}{R8-2 who executes the transition} \\",
         r"\cmidrule(lr){3-8}\cmidrule(lr){9-12}",
         r"reader & domain & A & Dp & Enat & Etyp & C & resolution & RULE & INJ & FREE & RULE$-$FREE \\",
         r"\midrule"]
for r, d in CELLS:
    arms = dict(re.findall(r"(\w+) ([\d.]+)", re.search(rf"^R8-1 {re.escape(r)} d{d}: (.*)$", txt, re.M).group(1)))
    h1 = re.search(rf"^  {re.escape(r)} d{d}: {NUM} {CI} p=", txt, re.M).groups()
    h2 = re.search(rf"^  {re.escape(r)} d{d}: BA RULE ([\d.]+) INJ ([\d.]+) FREE ([\d.]+); RULE - FREE {NUM} {CI}", txt, re.M).groups()
    lines.append(f"{NAME[r]} & {'first' if d == '1' else 'second'} & " +
                 " & ".join(arms[a] for a in ("A", "Dp", "Enat", "Etyp", "C")) + f" & {ci(h1)} & " +
                 " & ".join(h2[:3]) + f" & {ci(h2[3:])} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, "w").write("\n".join(lines) + "\n")
print(open(OUT).read())
