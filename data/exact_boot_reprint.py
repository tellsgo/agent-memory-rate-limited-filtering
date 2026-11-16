#!/usr/bin/env python3
"""Reprint of the one-sided pair-bootstrap p-values on integer counts, without changing the frozen scripts.

The frozen analyses (r4_holm.py, r5_comparators_analyze.boot, and r8.boot_combo, used by R8, R9 and R10) and the
joint Holm family of R5-3 and H-R5-4d compute each resampled difference as a float difference of float balanced accuracies and count the resamples
whose difference is <= 0. A resample whose difference is exactly zero can come out as a tiny positive number and drop
out of that count. This script draws the same resamples (random.Random(seed 0), 2000 draws, the same pair order) and
computes each difference from integer counts with fractions.Fraction. For every registered test it prints the frozen
float p and limits, the exact p and limits, the number of resamples whose difference is exactly zero, and both
decisions, with Holm over the registered family and margins compared exactly. Since every true zero is counted, the
exact p is never below the float p.

Families: the post-hoc R2/R3 Holm family of r4_holm.py, with 2000 and with 100000 resamples, R5-3 with the joint Holm family of R5-3 and H-R5-4d, R6-1,
R7-1 H1 and H2, R8-1, R8-2, R9 H9-1 and H9-2 (keyed and open-diff arms), R9-3, R10 H10-1 and H10-2 on both end points,
the R10 protocol check and E10. Reads only the shipped reading files.
Run: python3 -B exact_boot_reprint.py > exact_boot_reprint.out
"""
import json
import os
import random
import sys
from collections import defaultdict
from fractions import Fraction

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "domain2"))
import r8  # noqa: E402
from r2_analyze import bal  # noqa: E402
from r5_comparators_analyze import boot as boot5  # noqa: E402

N, SEED = 2000, 0
READERS = ("qwen2.5-14b", "phi-4")
OUT = []
CHANGED = []


def load(name):
    return [json.loads(l) for l in open(os.path.join(HERE, name))]


def F(x):
    """Exact rational of a decimal constant such as 0.05 or a weight such as 0.5."""
    return Fraction(str(x))


def counts(rows, key="arm"):
    g = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [0, 0])))
    for r in rows:
        c = g[r["pair"]][r[key]][r["truth"]]
        c[0] += r.get("decision") == r["truth"]
        c[1] += 1
    return g


def ba_exact(g, ps, arm):
    acc = defaultdict(lambda: [0, 0])
    for p in ps:
        for t, (c, m) in g[p][arm].items():
            acc[t][0] += c
            acc[t][1] += m
    return sum(Fraction(c, m) for c, m in acc.values()) / len(acc)


def exact_combo(rows, weights, n=N, seed=SEED):
    """Same resamples as r8.boot_combo, statistic sum_w w * BA(arm) computed exactly."""
    rows = [r for r in rows if r["arm"] in weights]
    g = counts(rows)
    pairs = sorted(p for p in g if all(a in g[p] for a in weights))
    stat = lambda ps: sum(F(w) * ba_exact(g, ps, a) for a, w in weights.items())
    rng = random.Random(seed)
    ds = sorted(stat([rng.choice(pairs) for _ in pairs]) for _ in range(n))
    return stat(pairs), ds[int(0.025 * n)], ds[int(0.975 * n) - 1], Fraction(sum(x <= 0 for x in ds), n), sum(x == 0 for x in ds)


def exact_two(a, b):
    """Exact counterpart of r5_comparators_analyze.boot(a, b) and r5_domain2_analyze.boot(a, b): pairs in both."""
    keep = {r["pair"] for r in a} & {r["pair"] for r in b}
    rows = [dict(r, arm="__a") for r in a if r["pair"] in keep] + [dict(r, arm="__b") for r in b if r["pair"] in keep]
    return exact_combo(rows, {"__a": 1, "__b": -1})


def float_two(a, b, n=N, seed=SEED):
    """r5_domain2_analyze.boot, which prints no p, with the one-sided p of the joint Holm family of R5-3 and H-R5-4d
    (0.036 for Qwen2.5-14B in the log of 2026-10-31)."""
    ga, gb = defaultdict(list), defaultdict(list)
    for r in a:
        ga[r["pair"]].append(r)
    for r in b:
        gb[r["pair"]].append(r)
    pairs = sorted(set(ga) & set(gb))
    rng = random.Random(seed)
    ds = []
    for _ in range(n):
        s = [rng.choice(pairs) for _ in pairs]
        ds.append(bal([r for p in s for r in ga[p]]) - bal([r for p in s for r in gb[p]]))
    d0 = bal([r for p in pairs for r in ga[p]]) - bal([r for p in pairs for r in gb[p]])
    p = sum(x <= 0 for x in ds) / n
    ds.sort()
    return d0, ds[int(0.025 * n)], ds[int(0.975 * n) - 1], p


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    rej, m = [False] * len(ps), len(ps)
    for j, i in enumerate(order):
        if ps[i] <= F(0.05) / (m - j):
            rej[i] = True
        else:
            break
    return rej


def family(name, tests, margin, record=None):
    """tests: (label, float (est, lo, hi, p), exact (est, lo, hi, p, zeros)). Decision: Holm rejects, est >= margin, lo > 0."""
    hf = holm([F(t[1][3]) for t in tests])
    he = holm([t[2][3] for t in tests])
    for (lab, f, e), a, b in zip(tests, hf, he):
        okf = a and F(f[0]) >= F(margin) and f[1] > 0
        oke = b and e[0] >= F(margin) and e[1] > 0
        flag = "" if okf == oke else "  DECISION CHANGES"
        if F(f[3]) != e[3] and (record is None or lab in record):
            CHANGED.append(f"{name} {lab}: {f[3]:.4f} -> {float(e[3]):.4f}")
        OUT.append(f"{name:26s} {lab:22s} float est {f[0]:+.4f} [{f[1]:+.4f}, {f[2]:+.4f}] p {f[3]:.4f} Holm {a} -> {okf} | "
                   f"exact est {float(e[0]):+.4f} [{float(e[1]):+.4f}, {float(e[2]):+.4f}] p {float(e[3]):.4f} "
                   f"(resamples exactly 0: {e[4]}) Holm {b} -> {oke}{flag}")


def r10():
    R = {rd: load(f"r10_read_{rd}.jsonl") for rd in READERS}
    E = {"reader": R, "state": {rd: [dict(r, decision=r["state_answer"]) for r in R[rd]] for rd in READERS}}
    for lab, w in (("H10-1", {"RULE": 1, "FREE_F": -1}), ("H10-2", {"FREE_F": 1, "FREE_K": -1})):
        for ep in ("reader", "state"):
            family(f"R10 {lab} {ep}", [(rd, r8.boot_combo(E[ep][rd], w), exact_combo(E[ep][rd], w)) for rd in READERS], 0.05)
    for rd in READERS:
        w = {"INJ_F": 1, "INJ_KT": -1}
        f, e = r8.boot_combo(E["state"][rd], w), exact_combo(E["state"][rd], w)
        OUT.append(f"R10 protocol check {rd}: float [{f[1]:+.4f}, {f[2]:+.4f}] within 0.05 {f[1] >= -0.05 and f[2] <= 0.05} | "
                   f"exact [{float(e[1]):+.4f}, {float(e[2]):+.4f}] within 0.05 {e[1] >= -F(0.05) and e[2] <= F(0.05)}")
    for rd in READERS:
        for ep in ("reader", "state"):
            w = {"RULE": 1, "FREE_F": -1}
            f, e = r8.boot_combo(E[ep][rd], w), exact_combo(E[ep][rd], w)
            OUT.append(f"R10 E10 {rd} {ep}: upper limit float {f[2]!r}, exact {e[2]} ({float(e[2]):.4f}); "
                       f"below 0.05: float {f[2] < 0.05}, exact {e[2] < F(0.05)}")


def r9():
    R9 = {rd: load(f"r9_read_{rd}.jsonl") for rd in READERS}
    cells = [(rd, d) for d in (1, 2) for rd in READERS]
    for X in ("K", "D"):
        fr = f"FREE_{X}"
        for lab, w in (("H9-1", {fr: 1, "FREE": -1}), ("H9-2", {"RULE": 1, fr: -1})):
            tests = []
            for rd, d in cells:
                rs = [r for r in R9[rd] if r["domain"] == d]
                tests.append((f"{rd} d{d}", r8.boot_combo(rs, w), exact_combo(rs, w)))
            family(f"R9 {'keyed' if X == 'K' else 'open diff'} {lab}", tests, 0.05)
    tests = []
    for rd in READERS:
        sm = load(f"r9_sm_read_{rd}.jsonl")
        pairs = {r["pair"] for r in sm}
        e7 = [dict(r, arm="E7") for r in load(f"r7_n1_read_{rd}.jsonl") if r["domain"] == 1 and r["arm"] == "E" and r["pair"] in pairs]
        w = {"E7": 1, "SM_FULL_projected": -1}
        tests.append((rd, r8.boot_combo(e7 + sm, w), exact_combo(e7 + sm, w)))
    family("R9-3 E - SM_FULL", tests, 0.10)


def r8_():
    R81 = {rd: load(f"r8_read_{rd}.jsonl") for rd in READERS}
    R82 = {rd: load(f"r8_read2_{rd}.jsonl") for rd in READERS}
    cells = [(rd, d) for d in (1, 2) for rd in READERS]
    w = {"Enat": .5, "A": -.5, "Etyp": .5, "Dp": -.5}
    family("R8-1 H8-1", [(f"{rd} d{d}", r8.boot_combo([r for r in R81[rd] if r["domain"] == d], w),
                          exact_combo([r for r in R81[rd] if r["domain"] == d], w)) for rd, d in cells], 0.05)
    w = {"RULE": 1, "FREE": -1}
    family("R8-2 H8-2", [(f"{rd} d{d}", r8.boot_combo([r for r in R82[rd] if r["domain"] == d], w),
                          exact_combo([r for r in R82[rd] if r["domain"] == d], w)) for rd, d in cells], 0.10)


def r7_1():
    R7 = {rd: load(f"r7_n1_read_{rd}.jsonl") for rd in READERS}
    get = lambda rd, dom, arm: [r for r in R7[rd] if r["domain"] == dom and r["arm"] == arm]
    for name, a, b in (("R7-1 H1 E-A", "E", "A"), ("R7-1 H2 E-B", "E", "B")):
        tests = [(f"{rd} d{dom}", boot5(get(rd, dom, a), get(rd, dom, b)), exact_two(get(rd, dom, a), get(rd, dom, b)))
                 for rd in READERS for dom in (1, 2)]
        family(name, tests, 0.05)


def r5_r6():
    fam = []
    for rd in READERS:
        comp = load(f"r5_comp_{rd}.jsonl")
        x6 = load(f"x6_read_{rd}.jsonl")
        filt = [r for r in x6 if r["design"] == "full" and r["method"] == "parse_filter:qwen2.5-14b" and r["B"] == 512 and r["t"] == 256]
        for m in ("retriever", "statemem_native", "statemem_policy", "mem0"):
            rows = [r for r in comp if r["method"] == m and r["t"] == 256 and r["B"] == 512]
            fam.append((f"{rd} {m}", boot5(filt, rows), exact_two(filt, rows)))
    family("R5-3 (Holm over 8)", fam, 0.10)
    d4 = []
    for rd in READERS:
        rows = [r for r in load(f"r5_d2_read_{rd}.jsonl") if r["T"] == 256]
        a, b = [r for r in rows if r["cond"] == "parsed"], [r for r in rows if r["cond"] == "retriever"]
        d4.append((f"{rd} H-R5-4d", float_two(a, b), exact_two(a, b)))
    family("R5-3 + R5-4d (Holm over 10)", fam + d4, 0.10, record={t[0] for t in d4})
    tests = []
    for rd in READERS:
        r6, r5 = load(f"r6_read_{rd}.jsonl"), load(f"r5_comp_{rd}.jsonl")
        a = [r for r in r6 if r["method"] == "mem0_dep" and r["t"] == 256 and r["B"] == 512]
        b = [r for r in r5 if r["method"] == "mem0" and r["t"] == 256 and r["B"] == 512]
        tests.append((rd, boot5(a, b), exact_two(a, b)))
    family("R6-1 (Holm over 2)", tests, 0.0)


def r2_r3_holm(n_res=N):
    """Exact counterpart of r4_holm.py: p = (count(d <= 0) + 1) / (n + 1), best of several ledgers for R3-2.
    n_res=100000 repeats the Monte Carlo check of r4_holm.py with HOLM_RESAMPLES=100000 (r7_holm_100k.out)."""
    def fbal(rs):
        by = defaultdict(lambda: [0, 0])
        for r in rs:
            by[r["truth"]][0] += r.get("decision") == r["truth"]
            by[r["truth"]][1] += 1
        return sum(Fraction(c, m) for c, m in by.values()) / len(by)

    def boot(a, b, best_of=None, n=n_res):
        sets = [a] + (best_of if best_of else [b])
        g = [defaultdict(list) for _ in sets]
        for gi, rs in zip(g, sets):
            for r in rs:
                gi[r["pair"]].append(r)
        ps = sorted(set.intersection(*[set(x) for x in g]))
        out = {}
        for kind, B in (("float", bal), ("exact", fbal)):
            stat = lambda sel: B([r for p in sel for r in g[0][p]]) - max(B([r for p in sel for r in gi[p]]) for gi in g[1:])
            rng = random.Random(0)
            bs = [stat([rng.choice(ps) for _ in ps]) for _ in range(n)]
            out[kind] = (stat(ps), Fraction(sum(x <= 0 for x in bs) + 1, n + 1), sum(x == 0 for x in bs))
        return out

    T = []
    for m in READERS:
        x6 = load(f"x6_read_{m}.jsonl")
        pf = [r for r in x6 if r["design"] == "full" and r["method"] == "parse_filter:qwen2.5-14b" and r["B"] is None and r["t"] == 256]
        T.append((f"R2-A {m}", boot(pf, load(f"r2_cot_{m}.jsonl"))))
        c2 = load(f"r2c2_{m}.jsonl")
        cot = [r for r in load(f"r2c_{m}.jsonl") if r["cond"] == "cot4096"]
        T.append((f"R2-C2b {m}", boot([r for r in c2 if r["cond"] == "parse_apply_resolved" and r["T"] == 256], cot)))
        led = load(f"r3_ledger_{m}.jsonl")
        T.append((f"R3-2 {m}", boot(pf, None, best_of=[[r for r in led if r["method"] == k and r["t"] == 256]
                                                       for k in ("r3_F16", "r3_F1", "r3_A1", "r3_S")])))
    for m in ("qwq-32b", "r1-distill-qwen-32b"):
        rr = load(f"r3_reason_{m}.jsonl")
        T.append((f"R3-1b {m}", boot([r for r in rr if r["cond"] == "x6_filter"], [r for r in rr if r["cond"] == "x6_log" and r["t"] == 256])))
    rej = {}
    for kind in ("float", "exact"):
        S = sorted(T, key=lambda x: x[1][kind][1])
        stop = False
        for k, (name, o) in enumerate(S):
            thr = F(0.05) / (len(S) - k)
            r = (not stop) and o[kind][1] <= thr
            stop = stop or not r
            rej[(kind, name)] = (r, thr)
    for name, o in sorted(T, key=lambda x: x[1]["float"][1]):
        f, e = o["float"], o["exact"]
        if f[1] != e[1]:
            CHANGED.append(f"R2/R3 Holm ({n_res} resamples) {name}: {float(f[1]):.4f} -> {float(e[1]):.4f}")
        OUT.append(f"R2/R3 Holm ({n_res:6d})    {name:22s} est {float(f[0]):+.4f} | float p {float(f[1]):.4f} Holm {rej[('float', name)][0]} "
                   f"(thr {float(rej[('float', name)][1]):.4f}) | exact p {float(e[1]):.4f} (resamples exactly 0: {e[2]}) "
                   f"Holm {rej[('exact', name)][0]}{'' if rej[('float', name)][0] == rej[('exact', name)][0] else '  DECISION CHANGES'}")


def D(c):
    a, b = c.rsplit(": ", 1)[1].split(" -> ")
    return float(b) - float(a)


def main():
    OUT.append("# One-sided pair-bootstrap p (share of resamples with difference <= 0), frozen float computation against")
    OUT.append("# integer counts on the same 2000 resamples, seed 0 (and 100000 for the Monte Carlo check of the R2/R3 family).")
    OUT.append("# Margins and Holm thresholds compared exactly.")
    for part in (r2_r3_holm, lambda: r2_r3_holm(100000), r5_r6, r7_1, r8_, r9, r10):
        OUT.append("")
        part()
    OUT.append("")
    reg = [c for c in CHANGED if "100000" not in c]
    mc = [c for c in CHANGED if "100000" in c]
    OUT.append(f"# p-values that change on the registered 2000 resamples: {len(reg)}, "
               f"rises from {min(D(c) for c in reg):.4f} to {max(D(c) for c in reg):.4f}")
    OUT.extend("  " + c for c in reg)
    OUT.append(f"# p-values that change in the 100000-resample check (printed to 4 decimals): {len(mc)}")
    OUT.extend("  " + c for c in mc)
    OUT.append(f"# decisions that change: {sum('DECISION CHANGES' in l for l in OUT)}")
    print("\n".join(OUT))


if __name__ == "__main__":
    main()
