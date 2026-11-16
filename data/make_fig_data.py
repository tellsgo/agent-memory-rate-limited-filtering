#!/usr/bin/env python3
"""CSV data for the journal figures, with pair-bootstrap 95% intervals (2000 resamples, seed 0).
Writes to the directory given as argv[1] (default ../figdata):
  fig_decay.csv    x12 reader curves: reader, cond (log|state), T, ba, lo, hi
  fig_horizon.csv  R4-1 / R5-5 / R6-3 strong readers on the 100 new pairs: reader, T, seed, log, filter, diff, lo, hi
  fig_store.csv    R5-1 bounded store at K=15: reader, T, policy, f, obs, lo, hi, pred_store, pred_proj
  fig_chain.csv    X5 chain counterfactuals (exact reader): rule, m, error  (copied from paper/generated/x5_chain_plot.tex)
"""
import csv, glob, json, os, random, re, sys
from collections import defaultdict

from r2_analyze import bal, boot_diff

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "figdata")
os.makedirs(OUT, exist_ok=True)


def boot_ci(rows, n=2000, seed=0):
    g = defaultdict(list)
    for r in rows:
        g[r["pair"]].append(r)
    pairs = sorted(g)
    rng = random.Random(seed)
    xs = sorted(bal([r for p in [rng.choice(pairs) for _ in pairs] for r in g[p]]) for _ in range(n))
    return bal(rows), xs[int(0.025 * n)], xs[int(0.975 * n) - 1]


def write(name, header, rows):
    with open(os.path.join(OUT, name), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"{name}: {len(rows)} rows")


def decay():
    G = defaultdict(list)
    for f in ("x12_qwen7b.jsonl", "x12_qwen14b.jsonl", "x12_phi4.jsonl"):
        for l in open(os.path.join(HERE, f)):
            r = json.loads(l)
            G[(r["reader"], r["cond"], r["T"])].append(r)
    out = []
    for (rd, cond, T), rs in sorted(G.items()):
        if cond in ("raw_gov", "state_gov"):
            ba, lo, hi = boot_ci(rs)
            out.append([rd, "log" if cond == "raw_gov" else "state", T, f"{ba:.4f}", f"{lo:.4f}", f"{hi:.4f}"])
    write("fig_decay.csv", ["reader", "cond", "T", "ba", "lo", "hi"], out)


def horizon():
    def load(m, suffix=""):
        fs = [os.path.join(HERE, f"r4_read_{m}{suffix}.jsonl")]
        if not suffix:
            fs += sorted(glob.glob(os.path.join(HERE, f"r4_read_{m}.rows*.jsonl")))
        return [json.loads(l) for f in fs if os.path.exists(f) for l in open(f)]
    sel = lambda rows, T, c: [r for r in rows if r["t"] == T and r["cond"] == c]
    out = []
    g0, g512, g1 = load("gpt-oss-120b"), load("gpt-oss-120b", "_r5t512"), load("gpt-oss-120b", "_r5seed1")
    q72 = load("qwen2.5-72b-awq")
    g2k = {s: load("gpt-oss-120b", f"_r6t2048s{s}") for s in (0, 1)}  # R6-3: fresh pairs, both arms per seed
    for name, rows, T, seed, filt in (("qwen2.5-72b", q72, 256, 0, None), ("qwen2.5-72b", q72, 1024, 0, None),
                                      ("gpt-oss-120b", g0, 256, 0, None), ("gpt-oss-120b", g512, 512, 0, None),
                                      ("gpt-oss-120b", g0, 1024, 0, None), ("gpt-oss-120b", g1, 1024, 1, sel(g0, 1024, "filter")),
                                      ("gpt-oss-120b", g2k[0], 2048, 0, None), ("gpt-oss-120b", g2k[1], 2048, 1, None)):
        lg, fl = sel(rows, T, "log"), filt if filt is not None else sel(rows, T, "filter")
        d, lo, hi = boot_diff(fl, lg)
        out.append([name, T, seed, f"{bal(lg):.4f}", f"{bal(fl):.4f}", f"{d:.4f}", f"{lo:.4f}", f"{hi:.4f}"])
    write("fig_horizon.csv", ["reader", "T", "seed", "log", "filter", "diff", "lo", "hi"], out)


def store():
    pred = json.load(open(os.path.join(HERE, "r5_bounded_pred.json")))
    F = pred["F_GRID"]
    out = []
    for reader in ("qwen2.5-14b", "phi-4"):
        rows = [json.loads(l) for l in open(os.path.join(HERE, f"r5_bounded_{reader}.jsonl"))]
        cal = defaultdict(list)
        G = defaultdict(list)
        for r in rows:
            (cal[(r["K"], r["T"])] if r["set"] == "cal" else G[(r["K"], r["T"], r["policy"], r["f"])]).append(r)
        for T in (64, 256, 1024):
            ba1 = bal(cal[(15, T)])
            for pol in ("random", "lru"):
                for i, f in enumerate(F):
                    ba, lo, hi = boot_ci(G[(15, T, pol, f)])
                    c = pred["cells"][f"15,{T}"]
                    ps, pp = (0.5 + (ba1 - 0.5) * c[k][pol][i] for k in ("rec", "proj"))
                    out.append([reader, T, pol, f, f"{ba:.4f}", f"{lo:.4f}", f"{hi:.4f}", f"{ps:.4f}", f"{pp:.4f}"])
    write("fig_store.csv", ["reader", "T", "policy", "f", "obs", "lo", "hi", "pred_store", "pred_proj"], out)


def chain():
    src = open(os.path.join(HERE, "..", "paper", "generated", "x5_chain_plot.tex")).read()
    names = re.search(r"\\legend\{(.*)\}", src).group(1).split(",")
    out = []
    for name, coords in zip(names, re.findall(r"coordinates \{(.*?)\}", src)):
        for m, e in re.findall(r"\(([\d.]+),([\d.]+)\)", coords):
            out.append([name, int(m), e])
    write("fig_chain.csv", ["rule", "m", "error"], out)


if __name__ == "__main__":
    decay(); horizon(); store(); chain()
