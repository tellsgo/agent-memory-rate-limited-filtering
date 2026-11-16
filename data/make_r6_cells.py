#!/usr/bin/env python3
"""Per-cell tables for the bounded-store predictions (blind review R2-N4): every interior cell of R5-1 (domain 1) and
every cell of H-R5-4c (domain 2), with prediction, measurement and gap, and a bootstrap of the aggregate gap that
resamples test pairs and calibration pairs jointly (domain 1: calibration seeds are separate streams, so both are
resampled within each (K, T); domain 2: the end point BA(1) is the parsed state on the same pairs, so one pair
resample moves both). Writes ../paper/generated/tab_cells_r5_1.tex, tab_cells_r5_4.tex and r6_cells.out."""
import json, os, random
from collections import defaultdict

from fmt import r2
from r2_analyze import bal

HERE = os.path.dirname(os.path.abspath(__file__))
GEN = os.path.join(HERE, "..", "paper", "generated")
P1 = json.load(open(os.path.join(HERE, "r5_bounded_pred.json")))
P4 = json.load(open(os.path.join(HERE, "r5_d2_pred.json")))
F = P1["F_GRID"]
CELLS = [(3, 64), (3, 256), (7, 64), (7, 256), (15, 64), (15, 256), (15, 1024)]
POLS = ("random", "lru", "default_first")
R1 = ("qwen2.5-14b", "phi-4")
R4 = ("qwen2.5-14b", "phi-4", "gpt-oss-120b")
NB = 2000


def q(xs):
    xs = sorted(xs)
    return xs[int(0.025 * len(xs))], xs[int(0.975 * len(xs)) - 1]


def domain1(out):
    data = {}
    for rd in R1:
        rows = [json.loads(l) for l in open(os.path.join(HERE, f"r5_bounded_{rd}.jsonl"))]
        cal = defaultdict(lambda: defaultdict(list))
        test = defaultdict(lambda: defaultdict(list))
        caps = {}
        for r in rows:
            if r["set"] == "cal":
                cal[(r["K"], r["T"])][r["pair"]].append(r)
            elif r["set"] == "test" and r["f"] in F[1:-1]:
                test[(r["K"], r["T"])][r["pair"]].append(r)
                caps[(r["K"], r["T"], r["f"])] = r["C"]
        data[rd] = (cal, test, caps)

    def cells(cal, test, cs=None, ts=None):
        res = {}
        for K, T in CELLS:
            cp = cs[(K, T)] if cs else sorted(cal[(K, T)])
            tp = ts[(K, T)] if ts else sorted(test[(K, T)])
            ba1 = bal([r for p in cp for r in cal[(K, T)][p]])
            by = defaultdict(list)
            for p in tp:
                for r in test[(K, T)][p]:
                    by[(r["policy"], r["f"])].append(r)
            for pol in POLS:
                for f in F[1:-1]:
                    fac = P1["cells"][f"{K},{T}"]["rec"][pol][F.index(f)]
                    res[(K, T, pol, f)] = (0.5 + (ba1 - 0.5) * fac, bal(by[(pol, f)]), len(tp), len(cp))
        return res

    point = {rd: cells(*data[rd][:2]) for rd in R1}
    rng = random.Random(0)
    for rd in R1:
        cal, test, _ = data[rd]
        g = [abs(o - p) for p, o, _, _ in point[rd].values()]
        maes, covs = [], []
        for _ in range(NB):
            cs = {c: [rng.choice(sorted(cal[c])) for _ in cal[c]] for c in CELLS}
            ts = {c: [rng.choice(sorted(test[c])) for _ in test[c]] for c in CELLS}
            gg = [abs(o - p) for p, o, _, _ in cells(cal, test, cs, ts).values()]
            maes.append(sum(gg) / len(gg))
            covs.append(sum(x <= 0.10 for x in gg) / len(gg))
        lo, hi = q(maes)
        clo, chi = q(covs)
        out.append(f"domain 1 {rd}: {len(g)} cells, mean |gap| {sum(g) / len(g):.3f} [{lo:.3f}, {hi:.3f}], "
                   f"within 0.10 {sum(x <= 0.10 for x in g)}/{len(g)}, share {clo:.2f} to {chi:.2f} (joint bootstrap)")
    caps = data[R1[0]][2]
    L = [r"\begin{longtable}{@{}rrlrrrrrr@{}}", r"\caption{Every interior cell of the registered bounded-store test R5-1 (first domain): prediction of \Cref{prop:budget} for the store, measured balanced accuracy, 40 test pairs and 40 calibration pairs per cell for each reader. Cells of one $(K,T)$ share their pairs and the calibration end point, so they are not independent tests.}\label{tab:cells-r51}\\",
         r"\toprule", r"$K$ & $T$ & policy & $f$ & $C$ & Qwen pred & Qwen obs & phi-4 pred & phi-4 obs \\", r"\midrule", r"\endfirsthead",
         r"\toprule", r"$K$ & $T$ & policy & $f$ & $C$ & Qwen pred & Qwen obs & phi-4 pred & phi-4 obs \\", r"\midrule", r"\endhead"]
    for K, T in CELLS:
        for pol in POLS:
            for f in F[1:-1]:
                a, b = point["qwen2.5-14b"][(K, T, pol, f)], point["phi-4"][(K, T, pol, f)]
                L.append(f"{K} & {T} & {pol.replace('_', ' ')} & {f:g} & {caps[(K, T, f)]} & {r2(a[0])} & {r2(a[1])} & {r2(b[0])} & {r2(b[1])} \\\\")
    L += [r"\bottomrule", r"\end{longtable}"]
    open(os.path.join(GEN, "tab_cells_r5_1.tex"), "w").write("\n".join(L) + "\n")


def domain4(out):
    L = [r"\begin{tabular}{@{}lrlrrrr@{}}", r"\toprule", r"reader & $T$ & policy & pairs & pred & obs & $|$gap$|$ \\", r"\midrule"]
    allg = []
    groups = {}
    for rd in R4:
        rows = [json.loads(l) for l in open(os.path.join(HERE, f"r5_d2_read_{rd}.jsonl"))]
        for T in (256, 1024):
            g = defaultdict(lambda: defaultdict(list))
            for r in rows:
                if r["T"] == T and r["cond"] in ("parsed", "bounded_random", "bounded_lru"):
                    g[r["cond"]][r["pair"]].append(r)
            if not g["parsed"] or not g["bounded_random"]:
                continue
            groups[(rd, T)] = g
            pairs = sorted(g["parsed"])
            bp = bal([r for p in pairs for r in g["parsed"][p]])
            for pol in ("random", "lru"):
                fac = P4["factor"][f"{T},{pol},0.75"]
                pr, ob = 0.5 + (bp - 0.5) * fac, bal([r for p in pairs for r in g[f"bounded_{pol}"][p]])
                allg.append(abs(ob - pr))
                L.append(f"{rd} & {T} & {pol} & {len(pairs)} & {r2(pr)} & {r2(ob)} & {abs(ob - pr):.3f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(GEN, "tab_cells_r5_4.tex"), "w").write("\n".join(L) + "\n")
    rng = random.Random(0)
    maes, covs = [], []
    Ts = sorted({T for _, T in groups})
    pairsT = {T: sorted(next(g for (_, t), g in groups.items() if t == T)["parsed"]) for T in Ts}
    assert all(sorted(g["parsed"]) == pairsT[T] for (_, T), g in groups.items())   # the readers share their pairs
    for _ in range(NB):
        gg = []
        S = {T: [rng.choice(p) for _ in p] for T, p in pairsT.items()}   # one resample per T, shared by every reader
        for (rd, T), g in groups.items():
            s = S[T]
            bp = bal([r for p in s for r in g["parsed"][p]])
            for pol in ("random", "lru"):
                fac = P4["factor"][f"{T},{pol},0.75"]
                gg.append(abs(bal([r for p in s for r in g[f"bounded_{pol}"][p]]) - (0.5 + (bp - 0.5) * fac)))
        maes.append(sum(gg) / len(gg))
        covs.append(sum(x <= 0.10 for x in gg) / len(gg))
    lo, hi = q(maes)
    clo, chi = q(covs)
    out.append(f"domain 2: {len(allg)} cells, mean |gap| {sum(allg) / len(allg):.3f} [{lo:.3f}, {hi:.3f}], within 0.10 "
               f"{sum(x <= 0.10 for x in allg)}/{len(allg)}, share {clo:.2f} to {chi:.2f} (pair bootstrap shared across readers, end point resampled with the pairs)")


def main():
    out = []
    domain1(out)
    domain4(out)
    open(os.path.join(HERE, "r6_cells.out"), "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
