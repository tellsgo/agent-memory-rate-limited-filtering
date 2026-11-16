"""Table of R9-0 and R9-1 (from r9_analyze_halfup.out, the frozen analysis reprinted with half-up rounding, and r9_step_stats.json)."""
import json
import os
import re

from fmt import r2

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "paper", "generated", "tab_r9.tex")
txt = open(os.path.join(HERE, "r9_analyze_halfup.out")).read()
CELLS = (("qwen2.5-14b", "1"), ("phi-4", "1"), ("qwen2.5-14b", "2"), ("phi-4", "2"))
NAME = {"qwen2.5-14b": "Qwen2.5-14B", "phi-4": "phi-4"}
CI = r"([+-]?[\d.]+) \[([+-]?[\d.]+), ([+-]?[\d.]+)\]"
ARMS = ("RULE", "INJ", "INJ_T", "INJ_U", "FREE", "INJ_K", "FREE_K")
HEAD = ("RULE", "INJ", "INJ$_T$", "INJ$_U$", "FREE", "INJ$_K$", "FREE$_K$")


def ci(m):
    return f"${m[0] if m[0].startswith('-') else '+' + m[0].lstrip('+')}$ $[{m[1]},{m[2]}]$"


STATS = {(z["writer"], z["domain"]): z for z in json.load(open(os.path.join(HERE, "r9_step_stats.json")))}
keyed = txt.split("keyed local edits (registered):")[1].split("open diff local edits")[0]
lines = [r"\begin{tabular}{ll ccccccc cc c cc}", r"\toprule",
         r" & & \multicolumn{7}{c}{balanced accuracy, read by the writer} & \multicolumn{3}{c}{registered (keyed local edits)} & "
         r"\multicolumn{2}{c}{updated errors per update} \\",
         r"\cmidrule(lr){3-9}\cmidrule(lr){10-12}\cmidrule(lr){13-14}",
         r"writer & domain & " + " & ".join(HEAD) + r" & FREE$_K-$FREE & RULE$-$FREE$_K$ & case & full & keyed \\",
         r"\midrule"]
for r, d in CELLS:
    row = re.search(rf"^  {re.escape(r)} d{d}: (RULE .*)$", txt, re.M).group(1)
    ba = {a: v for a, v in re.findall(r"(\w+) ([\d.]+) \(", row)}
    h = re.search(rf"^  {re.escape(r)} d{d}: H9-1 {CI} p=.*?; H9-2 {CI} p=.*?; case (\w)", keyed, re.M).groups()
    z = STATS[(r, int(d))]
    st = (z["full_touched_wrong"] / z["steps"], z["key_touched_wrong"] / z["steps"])
    lines.append(f"{NAME[r]} & {'first' if d == '1' else 'second'} & " + " & ".join(ba[a] for a in ARMS) +
                 f" & {ci(h[0:3])} & {ci(h[3:6])} & {h[6]} & {r2(st[0])} & {r2(st[1])} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, "w").write("\n".join(lines) + "\n")
print(open(OUT).read())
