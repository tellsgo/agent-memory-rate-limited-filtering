#!/usr/bin/env python3
"""X3 analysis against experiments/PREREG.md (H-X3a, H-X3b, H-X3c). Usage: python3 x3_analyze.py rows.jsonl [...]"""

import json
import random
import sys
from collections import defaultdict

NONFILTER = ["recency", "summary", "grad_evict", "rag_bm25"]


def bal(rs):
    a = [r for r in rs if r["truth"] == "APPROVE"]
    d = [r for r in rs if r["truth"] == "DENY"]
    if not a or not d:
        return float("nan")
    return (sum(r["decision"] == "APPROVE" for r in a) / len(a) + sum(r["decision"] == "DENY" for r in d) / len(d)) / 2


def groups(rows):
    g = defaultdict(list)
    for r in rows:
        g[(r["method"], r["B"], r["t"])].append(r)
    return g


def pairs_of(rs):
    d = defaultdict(list)
    for r in rs:
        d[r["pair"]].append(r)
    return d


def boot(fn, pair_ids, n=2000, seed=0):
    rng = random.Random(seed)
    vals = sorted(fn([rng.choice(pair_ids) for _ in pair_ids]) for _ in range(n))
    return vals[int(0.025 * n)], vals[int(0.975 * n) - 1]


def main(paths):
    rows = [json.loads(l) for p in paths for l in open(p)]
    G = groups(rows)
    methods = sorted({k[0] for k in G})
    budgets = sorted({k[1] for k in G if k[1] is not None})
    ts = sorted({k[2] for k in G})
    print(f"rows={len(rows)} methods={methods}")
    for B in budgets + [None]:
        print(f"\nB={B}: balanced acc by checkpoint t = {ts}")
        for m in methods:
            if (m, B, ts[-1]) not in G:
                continue
            cells = " ".join(f"{bal(G[(m, B, t)]):.2f}" for t in ts)
            tok = sum(r["memory_tokens"] for r in G[(m, B, ts[-1])]) / len(G[(m, B, ts[-1])])
            print(f"  {m:13s} {cells}   mem_tokens@t={ts[-1]}: {tok:.0f}")

    # H-X3a: ledger minus best non-filter at t=256, B=256, paired bootstrap over pairs
    tB, BB = ts[-1], 256
    if ("ledger", BB, tB) in G:
        P = {m: pairs_of(G[(m, BB, tB)]) for m in ["ledger"] + NONFILTER if (m, BB, tB) in G}
        ids = sorted(set.intersection(*[set(v) for v in P.values()]))

        def stat(sample, m):
            return bal([r for i in sample for r in P[m][i]])
        best = max((m for m in P if m != "ledger"), key=lambda m: stat(ids, m))
        d = stat(ids, "ledger") - stat(ids, best)
        lo, hi = boot(lambda s: stat(s, "ledger") - max(stat(s, m) for m in P if m != "ledger"), ids)
        v = "PASS" if d >= 0.10 and lo > 0 else ("SMALL-POSITIVE" if d > 0 else "NOT-BETTER")
        print(f"\nH-X3a t={tB} B={BB}: ledger {stat(ids, 'ledger'):.3f} vs best non-filter {best} {stat(ids, best):.3f}; "
              f"diff {d:+.3f}, CI(vs best-in-resample) [{lo:+.3f}, {hi:+.3f}] -> {v}")

    # H-X3b: drop from t=32 to t=256 at B=128, grad_evict vs latest_key
    if all((m, 128, t) in G for m in ("grad_evict", "latest_key") for t in (32, tB)):
        P = {(m, t): pairs_of(G[(m, 128, t)]) for m in ("grad_evict", "latest_key") for t in (32, tB)}
        ids = sorted(set.intersection(*[set(v) for v in P.values()]))

        def drop(sample, m):
            return bal([r for i in sample for r in P[(m, 32)][i]]) - bal([r for i in sample for r in P[(m, tB)][i]])
        d = drop(ids, "grad_evict") - drop(ids, "latest_key")
        lo, hi = boot(lambda s: drop(s, "grad_evict") - drop(s, "latest_key"), ids)
        print(f"H-X3b B=128: drop(grad_evict)={drop(ids, 'grad_evict'):+.3f} drop(latest_key)={drop(ids, 'latest_key'):+.3f}; "
              f"diff {d:+.3f} CI [{lo:+.3f}, {hi:+.3f}] -> {'PASS' if lo > 0 else 'NOT-SUPPORTED'}")

    if ("oracle_state", 512, tB) in G:
        print(f"H-X3c oracle_state B=512 t={tB}: {bal(G[('oracle_state', 512, tB)]):.3f} (compare X2 state_gov T=256)")


if __name__ == "__main__":
    main(sys.argv[1:])
