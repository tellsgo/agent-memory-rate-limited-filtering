#!/usr/bin/env python3
"""R3-3 analysis (PREREG H-B): BA of budgeted filters against the frozen factor table of
theory/REGISTRATION_round3.md. Usage: python3 r3_budget_analyze.py"""
import json, os, random, re
from collections import defaultdict

from fmt import r2

HERE = os.path.dirname(os.path.abspath(__file__))
F = [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]


def table():
    tab = {}
    for line in open(os.path.join(HERE, "..", "theory", "REGISTRATION_round3.md")):
        m = re.match(r"\|\s*(random|recency)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|(.*)\|\s*$", line)
        if m:
            vals = [float(x) for x in m.group(4).split("|")]
            assert len(vals) == 9
            tab[(m.group(1), int(m.group(2)), int(m.group(3)))] = vals
    assert len(tab) == 12
    return tab


def bal(rs):
    by = defaultdict(list)
    for r in rs:
        by[r["truth"]].append(r["decision"] == r["truth"])
    return sum(sum(v) / len(v) for v in by.values()) / len(by)


def main():
    tab = table()
    for reader in ("qwen2.5-14b", "phi-4"):
        p = os.path.join(HERE, f"r3_budget_{reader}.jsonl")
        if not os.path.exists(p):
            print(reader, "missing"); continue
        rows = [json.loads(l) for l in open(p)]
        G = defaultdict(list)
        for r in rows:
            G[(r["policy"], r["K"], r["T"], r["f"])].append(r)
        devs, b2 = [], []
        print(f"== {reader}")
        for (pol, K, T), fac in sorted(tab.items()):
            ba1 = bal(G[(pol, K, T, 1.0)])
            obs = [bal(G[(pol, K, T, f)]) for f in F]
            pred = [0.5 + (ba1 - 0.5) * x for x in fac]
            for j in range(1, 8):
                devs.append(abs(obs[j] - pred[j]))
            b2.append(abs(obs[0] - 0.5))
            print(f"{pol:7} K={K:<2} T={T:<3} obs " + " ".join(r2(x) for x in obs) + " | pred " + " ".join(r2(x) for x in pred))
        mean_dev = sum(devs) / len(devs); within = sum(d <= 0.10 for d in devs) / len(devs)
        print(f"H-B1: mean |obs-pred| {mean_dev:.3f} (<=0.05?), share within 0.10 {within:.2f} (>=0.90?) -> {'holds' if mean_dev <= 0.05 and within >= 0.9 else 'fails'}")
        print(f"H-B2: max |BA(f=0)-0.5| {max(b2):.3f} -> {'holds' if max(b2) <= 0.05 else 'fails'}")
        for K in (7, 15):
            for T in (64, 256):
                obs = [bal(G[("random", K, T, f)]) for f in F]
                n = len(F); mx = sum(F) / n; my = sum(obs) / n
                slope = sum((x - mx) * (y - my) for x, y in zip(F, obs)) / sum((x - mx) ** 2 for x in F); icpt = my - slope * mx
                ok = abs(slope - (obs[-1] - 0.5)) <= 0.10 and abs(icpt - 0.5) <= 0.05
                print(f"H-B3 K={K} T={T}: slope {slope:.3f} vs {obs[-1] - 0.5:.3f}, intercept {icpt:.3f} -> {'holds' if ok else 'fails'}")
        fs = [0.25, 0.375, 0.5, 0.625, 0.75]
        byp = defaultdict(lambda: defaultdict(list))
        for pol in ("random", "recency"):
            for f in fs:
                for r in G[(pol, 15, 256, f)]:
                    byp[r["pair"]][(pol, f)].append(r)
        pairs = sorted(byp)
        def stat(ps):
            return sum(bal([r for p in ps for r in byp[p][("random", f)]]) - bal([r for p in ps for r in byp[p][("recency", f)]]) for f in fs) / len(fs)
        d = stat(pairs); rng = random.Random(0)
        bs = sorted(stat([rng.choice(pairs) for _ in pairs]) for _ in range(2000))
        print(f"H-B4: random - recency at K=15 T=256 {d:+.3f} [{bs[50]:.3f}, {bs[1949]:.3f}] -> {'holds' if d > 0 and bs[50] > 0 else 'fails'}")


if __name__ == "__main__":
    main()
