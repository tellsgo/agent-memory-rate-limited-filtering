"""Reprint the R8 or R9 analysis with the paper's rounding rule (fmt.r2, half up), without changing the frozen scripts.

r8.py and r9.py print estimates with Python's float formatting, which rounds a binary tie such as 0.125 down. The paper
rounds half up (PREREG, rounding correction of 2026-10-27). This script runs the frozen phase_analyze unchanged, with
balanced accuracies and bootstrap estimates wrapped in a float whose two-decimal format rounds half up, and writes
<round>_analyze_halfup.out next to the original. Decisions, intervals and p-values are computed exactly as before.
Usage: python3 halfup_rerun.py r8|r9
"""
import builtins
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from fmt import r2  # noqa: E402


class HF(float):
    def __format__(self, spec):
        m = re.fullmatch(r"(\+?)\.(\d)f", spec)
        if not m:
            return float.__format__(self, spec)
        s = r2(float(self), int(m.group(2)))
        if s.lstrip("-").strip("0.") == "":
            s = s.lstrip("-")
        return ("+" + s if m.group(1) and not s.startswith("-") else s)


def main(rnd):
    import r2_analyze
    import r8
    bal0, boot0 = r2_analyze.bal, r8.boot_combo
    r2_analyze.bal = lambda rows: HF(bal0(rows))

    def boot(*a, **k):
        x = boot0(*a, **k)
        return tuple(HF(v) for v in x[:3]) + tuple(x[3:])
    r8.boot_combo = boot
    r8.fmt = lambda x: f"{HF(x[0]):+.2f} [{HF(x[1]):.2f}, {HF(x[2]):.2f}]"
    mod = r8
    if rnd == "r9":
        import r9
        r9.boot_combo, r9.fmt, mod = boot, r8.fmt, r9

    def opener(f, mode="r", *a, **k):
        if "w" in mode and str(f).endswith(f"{rnd}_analyze.out"):
            f = str(f)[:-4] + "_halfup.out"
        return builtins.open(f, mode, *a, **k)
    mod.open = opener
    mod.phase_analyze()


if __name__ == "__main__":
    main(sys.argv[1])
