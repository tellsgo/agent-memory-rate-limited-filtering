#!/usr/bin/env python3
"""Holm adjustment over the registered R2/R3 comparisons with a pass criterion (referee 4, T5).
One-sided bootstrap p-value: share of 2000 pair resamples (seed 0) in which the difference is <= 0, as registered.
HOLM_RESAMPLES=100000 repeats it with more resamples to measure the Monte Carlo error (blind review, report only)."""
import json, os, random
from collections import defaultdict
from r2_analyze import bal, load

def pairs_of(*sets):
    g = [defaultdict(list) for _ in sets]
    for gi, rs in zip(g, sets):
        for r in rs:
            gi[r["pair"]].append(r)
    return g, sorted(set.intersection(*[set(x) for x in g]))

def boot(a, b, best_of=None, n=int(os.environ.get("HOLM_RESAMPLES", "2000"))):
    sets = [a] + (best_of if best_of else [b])
    g, ps = pairs_of(*sets)
    def stat(sel):
        A = bal([r for p in sel for r in g[0][p]])
        B = max(bal([r for p in sel for r in gi[p]]) for gi in g[1:])
        return A - B
    d = stat(ps); rng = random.Random(0)
    bs = [stat([rng.choice(ps) for _ in ps]) for _ in range(n)]
    return d, (sum(x <= 0 for x in bs) + 1) / (n + 1)

T = []
for m in ("qwen2.5-14b", "phi-4"):
    x6 = load(f"x6_read_{m}.jsonl"); pf = [r for r in x6 if r["design"] == "full" and r["method"] == "parse_filter:qwen2.5-14b" and r["B"] is None and r["t"] == 256]
    T.append((f"R2-A {m}", *boot(pf, load(f"r2_cot_{m}.jsonl"))))
    c2 = load(f"r2c2_{m}.jsonl"); cot = [r for r in load(f"r2c_{m}.jsonl") if r["cond"] == "cot4096"]
    T.append((f"R2-C2b {m}", *boot([r for r in c2 if r["cond"] == "parse_apply_resolved" and r["T"] == 256], cot)))
    led = load(f"r3_ledger_{m}.jsonl")
    T.append((f"R3-2 {m}", *boot(pf, None, best_of=[[r for r in led if r["method"] == k and r["t"] == 256] for k in ("r3_F16", "r3_F1", "r3_A1", "r3_S")])))
for m in ("qwq-32b", "r1-distill-qwen-32b"):
    rr = load(f"r3_reason_{m}.jsonl")
    T.append((f"R3-1b {m}", *boot([r for r in rr if r["cond"] == "x6_filter"], [r for r in rr if r["cond"] == "x6_log" and r["t"] == 256])))
T.sort(key=lambda x: x[2]); m = len(T)
print(f"{'test':28s} diff   p      Holm threshold  reject H0 (diff<=0)")
stop = False
for k, (name, d, p) in enumerate(T):
    thr = 0.05 / (m - k); rej = (not stop) and p <= thr
    stop = stop or not rej
    print(f"{name:28s} {d:+.3f} {p:.4f} {thr:.4f}         {rej}")
