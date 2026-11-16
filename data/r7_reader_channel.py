#!/usr/bin/env python3
"""Does the reader term of prop:budget add anything on R5-1? (fresh blind review, R1-M2 / R2-M2; written after R5-1
was known, report only.) Both predictions use the registered factor s = 1 - 2 P_e of r5_bounded_pred.json:
  registered      BA = 1/2 + (BA(1) - 1/2) s, with BA(1) read on the calibration seeds
  exact store     BA = 1/2 + s/2, the same formula with a perfect reader (no parameter)
Also the reader channel: the exact answer y_hat of every store that was read is regenerated (the read stores are
deterministic, seeded "{pair}-{truth}-{f}-{pol}-read" as in r5_bounded.read) and each reader's balanced agreement with
y_hat on interior stores is compared with its BA(1) on full states. prop:budget assumes the two are equal.
Joint bootstrap of the MAE difference resamples test and calibration pairs within each (K, T), factors held fixed.
The same comparison is made on the R3-3 read-time projections, whose end point is the f = 1 cell of the same streams.
Writes r7_reader_channel.out and ../paper/generated/tab_reader_channel.tex."""
import json, os, random, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r5_bounded as rb  # noqa: E402
from state_machine import answer_from_state, replay_events  # noqa: E402
from r2_analyze import bal  # noqa: E402

PRED = json.load(open(os.path.join(HERE, "r5_bounded_pred.json")))
F = PRED["F_GRID"]
READERS = ("qwen2.5-14b", "phi-4")


def yhat_rows():
    out = []
    for K, T in rb.CELLS:
        eps = rb.cell_streams(K, T, rb.TEST_BASE)
        caps, _ = rb.capacities(eps)
        for ep in eps:
            evs = replay_events(ep)
            for pol in rb.POLICIES:
                for f in rb.F_GRID:
                    rng = random.Random(f"{ep['pair']}-{ep['truth']}-{f}-{pol}-read")
                    st = rb.bounded_state(evs, T, caps[f], pol, rng)
                    out.append((K, T, f, pol, ep["pair"], ep["truth"], answer_from_state(st, ep["p"], ep["r"])))
    return out


def main():
    Y = yhat_rows()
    out, tab = [], {}
    for rd in READERS:
        rows = [json.loads(l) for l in open(os.path.join(HERE, f"r5_bounded_{rd}.jsonl"))]
        test = [r for r in rows if r["set"] == "test"]
        assert len(test) == len(Y)
        for r, y in zip(test, Y):
            assert (r["K"], r["T"], r["f"], r["policy"], r["pair"], r["truth"]) == y[:6]
            r["yhat"] = y[6]
        cal = defaultdict(list)
        for r in rows:
            if r["set"] == "cal":
                cal[(r["K"], r["T"])].append(r)
        ba1 = {c: bal(v) for c, v in cal.items()}
        G = defaultdict(list)
        for r in test:
            G[(r["K"], r["T"], r["policy"], r["f"])].append(r)
        cells = []
        for K, T in rb.CELLS:
            for pol in rb.POLICIES:
                for f in F[1:-1]:
                    s = PRED["cells"][f"{K},{T}"]["rec"][pol][F.index(f)]
                    o = bal(G[(K, T, pol, f)])
                    cells.append(dict(K=K, T=T, pol=pol, f=f, C=PRED["cells"][f"{K},{T}"]["C"][str(f)], s=s, o=o,
                                      pc=0.5 + (ba1[(K, T)] - 0.5) * s, ps=0.5 + s / 2))
        M = lambda k, R: sum(abs(c["o"] - c[k]) for c in R) / len(R)
        hi = [c for c in cells if c["s"] >= 0.25]
        out.append(f"{rd}: {len(cells)} interior cells, MAE registered {M('pc', cells):.4f}, exact store {M('ps', cells):.4f}; "
                   f"cells with s >= 0.25 ({len(hi)}): {M('pc', hi):.4f} and {M('ps', hi):.4f}; mean signed residual "
                   f"observed - registered {sum(c['o'] - c['pc'] for c in cells) / len(cells):+.4f}")
        # reader channel against y_hat, interior stores, per (K, T)
        for K, T in rb.CELLS:
            R = [r for r in test if (r["K"], r["T"]) == (K, T) and r["f"] not in (F[0], F[-1])]
            by = defaultdict(list)
            for r in R:
                by[r["yhat"]].append(r["decision"] == r["yhat"])
            agree = sum(sum(v) / len(v) for v in by.values()) / len(by)
            tab.setdefault((K, T), {})[rd] = (agree, ba1[(K, T)])
        # joint bootstrap of MAE(registered) - MAE(exact store)
        calp = {c: defaultdict(list) for c in rb.CELLS}
        tp = {c: defaultdict(list) for c in rb.CELLS}
        for c, v in cal.items():
            for r in v:
                calp[c][r["pair"]].append(r)
        for r in test:
            if r["f"] not in (F[0], F[-1]):
                tp[(r["K"], r["T"])][r["pair"]].append(r)
        rng, diffs = random.Random(0), []
        for _ in range(2000):
            dc = ds = n = 0
            for K, T in rb.CELLS:
                cp, pp = sorted(calp[(K, T)]), sorted(tp[(K, T)])
                b1 = bal([r for p in (rng.choice(cp) for _ in cp) for r in calp[(K, T)][p]])
                g = defaultdict(list)
                for p in (rng.choice(pp) for _ in pp):
                    for r in tp[(K, T)][p]:
                        g[(r["policy"], r["f"])].append(r)
                for (pol, f), rs in g.items():
                    s = PRED["cells"][f"{K},{T}"]["rec"][pol][F.index(f)]
                    o = bal(rs)
                    dc += abs(o - (0.5 + (b1 - 0.5) * s))
                    ds += abs(o - (0.5 + s / 2))
                    n += 1
            diffs.append((dc - ds) / n)
        diffs.sort()
        out.append(f"  MAE registered - exact store, joint bootstrap 2000: [{diffs[50]:.4f}, {diffs[1949]:.4f}], "
                   f"share > 0 {sum(d > 0 for d in diffs) / len(diffs):.3f}")
    from r3_budget_analyze import F as F3, table
    tab3 = table()
    for rd in READERS:
        G3 = defaultdict(list)
        for r in map(json.loads, open(os.path.join(HERE, f"r3_budget_{rd}.jsonl"))):
            G3[(r["policy"], r["K"], r["T"], r["f"])].append(r)
        dc, ds = [], []
        for (pol, K, T), fac in tab3.items():
            b1 = bal(G3[(pol, K, T, 1.0)])
            for j in range(1, 8):
                o = bal(G3[(pol, K, T, F3[j])])
                dc.append(abs(o - (0.5 + (b1 - 0.5) * fac[j])))
                ds.append(abs(o - (0.5 + fac[j] / 2)))
        out.append(f"R3-3 projections {rd}: {len(dc)} interior cells, MAE registered {sum(dc) / len(dc):.4f}, exact projection {sum(ds) / len(ds):.4f}")
    for (K, T), v in tab.items():
        out.append(f"  K={K} T={T}: " + ", ".join(f"{rd} agreement with y_hat {a:.3f} vs BA(1) {b:.3f}" for rd, (a, b) in v.items()))
    L = [r"\begin{tabular}{@{}rrcccc@{}}", r"\toprule",
         r" & & \multicolumn{2}{c}{Qwen2.5-14B} & \multicolumn{2}{c}{phi-4} \\", r"\cmidrule(lr){3-4}\cmidrule(l){5-6}",
         r"$K$ & $T$ & agreement with $\hat y$ & $\mathrm{BA}(1)$ & agreement with $\hat y$ & $\mathrm{BA}(1)$ \\", r"\midrule"]
    for (K, T), v in tab.items():
        L.append(f"{K} & {T:,} & ".replace(",", "{,}") + " & ".join(f"{a:.3f} & {b:.3f}" for a, b in (v[rd] for rd in READERS)) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(HERE, "..", "paper", "generated", "tab_reader_channel.tex"), "w").write("\n".join(L) + "\n")
    open(os.path.join(HERE, "r7_reader_channel.out"), "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
