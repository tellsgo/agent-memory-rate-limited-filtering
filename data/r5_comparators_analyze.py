#!/usr/bin/env python3
"""R5-3 analysis (PREREG H-R5-3): capped parsed filter (X6, B=512) minus each comparator (B=512) at t=256, per reader,
pair bootstrap 2000 seed 0, one-sided bootstrap p (share of resamples with difference <= 0), Holm over the family.
Also the uncapped comparison, t=64, and write costs from the memory files."""
import json, os, random
from collections import defaultdict
from statistics import mean

from fmt import r2
from r2_analyze import bal, fmt_d

HERE = os.path.dirname(os.path.abspath(__file__))
METHODS = ["retriever", "statemem_native", "statemem_policy", "mem0"]


def boot(a, b, n=2000, seed=0):
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


def main():
    fam = []
    for reader in ("qwen2.5-14b", "phi-4", "qwen2.5-32b"):
        path = os.path.join(HERE, f"r5_comp_{reader}.jsonl")
        if not os.path.exists(path):
            continue
        comp = [json.loads(l) for l in open(path)]
        x6 = [json.loads(l) for l in open(os.path.join(HERE, f"x6_read_{reader}.jsonl"))]
        filt = {(t, B): [r for r in x6 if r["design"] == "full" and r["method"] == "parse_filter:qwen2.5-14b" and r["B"] == B and r["t"] == t]
                for t in (64, 256) for B in (None, 512)}
        print(f"\n=== {reader}: parsed filter t=256 {r2(bal(filt[(256, None)]))} uncapped, {r2(bal(filt[(256, 512)]))} at 512")
        for m in METHODS:
            for t in (64, 256):
                for B in (512, None):
                    rows = [r for r in comp if r["method"] == m and r["t"] == t and r["B"] == B]
                    if not rows:
                        continue
                    d = boot(filt[(t, B)], rows)
                    tag = ""
                    if t == 256 and B == 512 and reader != "qwen2.5-32b":
                        fam.append((reader, m, d))
                        tag = " [registered]"
                    print(f"  {m:16s} t={t:3d} B={str(B):4s} BA {r2(bal(rows))}  tokens {mean(r['memory_tokens'] for r in rows):.0f}  "
                          f"filter - it {fmt_d(d[:3])} p={d[3]:.4f}{tag}")
    print("\nH-R5-3 family (filter at 512 - comparator at 512, t=256), Holm over", len(fam), "tests:")
    order = sorted(range(len(fam)), key=lambda i: fam[i][2][3])
    k = len(fam)
    stop = False
    for rank, i in enumerate(order):
        reader, m, d = fam[i]
        thr = 0.05 / (k - rank)
        rej = (not stop) and d[3] <= thr
        stop = stop or not rej
        crit = d[0] >= 0.10 and d[1] > 0
        print(f"  {reader:12s} {m:16s} {fmt_d(d[:3])} p={d[3]:.4f} Holm thr {thr:.4f} {'rejects' if rej else 'no'}; "
              f"registered margin {'met' if crit else 'not met'}")
    for m in METHODS:
        p = os.path.join(HERE, f"r5_mem_{m}.jsonl")
        if os.path.exists(p):
            rows = [json.loads(l) for l in open(p) if json.loads(l)["t"] == 256]
            if rows and "write_calls" in rows[0]:
                print(f"cost {m}: write calls {mean(r['write_calls'] for r in rows):.0f}, write tokens {mean(r['write_tokens'] for r in rows):,.0f}"
                      + (f", active units {mean(r['units_active'] for r in rows):.0f}, parse failures {rows[0].get('parse_failures')}" if "units_active" in rows[0] else ""))


if __name__ == "__main__":
    main()
