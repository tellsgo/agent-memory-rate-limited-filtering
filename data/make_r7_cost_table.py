"""Accuracy and total tokens per history for the R7-1 arms A, C, E and E' (from r7_n1_analyze.out)."""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "paper", "generated", "tab_r7_cost.tex")
txt = open(os.path.join(HERE, "r7_n1_analyze.out")).read()

ba = {}
for r, d, rest in re.findall(r"^(qwen2\.5-14b|phi-4) domain (\d): (.*)$", txt, re.M):
    ba[(r, d)] = dict((k, float(v)) for k, v in re.findall(r"(\w) ([\d.]+)", rest))
res = {}
for d, line in re.findall(r"^resources d(\d) \(per stream, Qwen2\.5-14B tokens\): (.*)$", txt, re.M):
    g = lambda pat: float(re.search(pat, line).group(1).replace(",", ""))
    res[d] = dict(write=g(r"256 parser calls \(([\d,]+) tokens\)"), qcall=g(r"E' [\d.]+ calls per query \(([\d,]+) tokens\)"),
                  rA=g(r"read tokens A (\d+)"), rE=g(r"E (\d+); generated"), gC=g(r"C (\d+) \(plus"))

QS = (1, 10, 100, 1000)


def cost(arm, d, q):
    r = res[d]
    return {"A": r["rA"] * q, "C": (r["rA"] + r["gC"]) * q, "E": r["write"] + r["rE"] * q,
            "E'": (r["qcall"] + r["rE"]) * q}[arm]


def fmt(x):
    if x >= 1e6:
        return f"{x / 1e6:.2f}M"
    if x >= 1e3:
        return f"{x / 1e3:.1f}k"
    return f"{x:.0f}"


NAMES = {"A": "A raw dependency lines", "C": "C answer-time summary", "E": "E resolved rows, written per event",
         "E'": "E$'$ resolved rows, parsed per query"}
lines = [r"\begin{tabular}{l cc cc cccc cccc}", r"\toprule",
         r" & \multicolumn{2}{c}{BA, Qwen2.5-14B} & \multicolumn{2}{c}{BA, phi-4} & \multicolumn{4}{c}{tokens, first domain}"
         r" & \multicolumn{4}{c}{tokens, second domain} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-9}\cmidrule(lr){10-13}",
         r"arm & first & second & first & second & " + " & ".join(f"$Q{{=}}{q}$" for q in QS) + " & " +
         " & ".join(f"$Q{{=}}{q}$" for q in QS) + r" \\", r"\midrule"]
for arm in ("A", "C", "E", "E'"):
    k = "E" if arm == "E'" else arm
    accs = [ba[(r, d)][k] for r, d in (("qwen2.5-14b", "1"), ("qwen2.5-14b", "2"), ("phi-4", "1"), ("phi-4", "2"))]
    cells = [f"{a:.2f}" for a in accs] + [fmt(cost(arm, d, q)) for d in ("1", "2") for q in QS]
    lines.append(NAMES[arm] + " & " + " & ".join(cells) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, "w").write("\n".join(lines) + "\n")
for d in ("1", "2"):
    r = res[d]
    print(f"d{d}: E against C breaks even at {r['write'] / (r['rA'] + r['gC'] - r['rE']):.0f} queries, "
          f"E against E' at {r['write'] / (r['qcall']):.0f}")
print(open(OUT).read())
