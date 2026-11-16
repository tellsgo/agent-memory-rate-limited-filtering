#!/usr/bin/env python3
"""X1/X2 analysis against experiments/PREREG.md. Usage: python3 x12_analyze.py rows.jsonl [...]"""

import json
import random
import sys
from collections import defaultdict


def auc(pos, neg):
    s = sum((x > y) + 0.5 * (x == y) for x in pos for y in neg)
    return s / (len(pos) * len(neg))


def bal(rs):
    a = [r for r in rs if r["truth"] == "APPROVE"]
    d = [r for r in rs if r["truth"] == "DENY"]
    return (sum(r["decision"] == "APPROVE" for r in a) / len(a) + sum(r["decision"] == "DENY" for r in d) / len(d)) / 2


def spearman(x, y):
    def rk(v):
        o = sorted(range(len(v)), key=lambda i: v[i])
        r = [0] * len(v)
        for k, i in enumerate(o):
            r[i] = k
        return r
    rx, ry = rk(x), rk(y)
    n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den


def paired_boot_diff(rs_a, rs_b, n=2000, seed=0):
    """CI of bal(rs_a) - bal(rs_b), resampling pairs within each set independently."""
    def by_pair(rs):
        d = defaultdict(list)
        for r in rs:
            d[r["pair"]].append(r)
        return list(d.values())
    pa, pb = by_pair(rs_a), by_pair(rs_b)
    rng = random.Random(seed)
    diffs = []
    for _ in range(n):
        sa = [r for g in (rng.choice(pa) for _ in pa) for r in g]
        sb = [r for g in (rng.choice(pb) for _ in pb) for r in g]
        diffs.append(bal(sa) - bal(sb))
    diffs.sort()
    return diffs[int(0.025 * n)], diffs[int(0.975 * n) - 1]


def main(paths):
    rows = [json.loads(l) for p in paths for l in open(p)]
    for reader in sorted({r["reader"] for r in rows}):
        R = [r for r in rows if r["reader"] == reader]
        Ts = sorted({r["T"] for r in R})
        print(f"\n===== {reader}  ({len(R)} rows) =====")
        print(f"{'cond':14s} " + " ".join(f"T={T:<4d}" for T in Ts) + "   (balanced acc / AUC)")
        table = {}
        for cond in ("raw_gov", "raw_neutral", "state_gov", "state_raw_gov"):
            cells = []
            for T in Ts:
                rs = [r for r in R if r["cond"] == cond and r["T"] == T]
                A = [r["margin"] for r in rs if r["truth"] == "APPROVE"]
                D = [r["margin"] for r in rs if r["truth"] == "DENY"]
                table[(cond, T)] = (bal(rs), auc(D, A), sum(r["decision"] == "DENY" for r in rs) / len(rs))
                cells.append(f"{table[(cond, T)][0]:.2f}/{table[(cond, T)][1]:.2f}")
            print(f"{cond:14s} " + " ".join(f"{c:9s}" for c in cells))
        print("DENY share raw_gov    : " + " ".join(f"{table[('raw_gov', T)][2]:.2f}" for T in Ts))
        print("DENY share raw_neutral: " + " ".join(f"{table[('raw_neutral', T)][2]:.2f}" for T in Ts))
        # H-X2
        st = [table[("state_gov", T)][0] for T in Ts]
        lo, hi = paired_boot_diff([r for r in R if r["cond"] == "state_gov" and r["T"] == Ts[-1]],
                                  [r for r in R if r["cond"] == "state_gov" and r["T"] == Ts[0]])
        passes_level = all(x >= 0.95 for x in st)
        flat = lo >= -0.03
        verdict = "PASS" if passes_level and flat else ("LEVEL-OK-BUT-DECLINES" if passes_level else "BELOW-0.95")
        print(f"H-X2  state_gov min bal={min(st):.3f}; T{Ts[-1]}-T{Ts[0]} diff CI [{lo:+.3f}, {hi:+.3f}] -> {verdict}")
        # H-X1a
        rho = spearman(Ts, [table[("raw_gov", T)][1] for T in Ts])
        print(f"H-X1a raw_gov AUC Spearman vs T = {rho:+.2f} -> {'monotone decline' if rho < 0 else 'no decline'}")
        # H-X1b
        if 64 in Ts:
            d = table[("raw_neutral", 64)][0] - table[("raw_gov", 64)][0]
            print(f"H-X1b neutral - gov balanced acc at T=64 = {d:+.3f} -> {'switch to neutral' if d >= 0.10 else 'keep gov'}")
        # per-family at T=256 for state_gov
        fam = defaultdict(list)
        for r in R:
            if r["cond"] == "state_gov" and r["T"] == Ts[-1]:
                fam[(r["family"], r["truth"])].append(r["decision"] == r["truth"])
        print("state_gov per family at T=%d: " % Ts[-1] + ", ".join(f"{f}/{t[0]}={sum(v) / len(v):.2f}" for (f, t), v in sorted(fam.items())))
        agree = sum(r["gen_label"] == r["decision"] for r in R) / len(R)
        print(f"generation agreement {agree:.3f}")


if __name__ == "__main__":
    main(sys.argv[1:])
