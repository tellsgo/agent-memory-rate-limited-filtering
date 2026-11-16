#!/usr/bin/env python3
"""Estimate of an upper bound on the entropy H(Z_T) of the filter state under the benchmark generator (v2,
make_pair_v2): by subadditivity H(Z_T) <= sum over the 72 state components of each component's entropy.
Each component entropy is estimated by the plug-in rule (biased low) and with the Miller-Madow correction
(k - 1) / (2 n ln 2) bits, k = number of observed values; a bootstrap over episodes gives 95% intervals.
Compare with log2 N_Q (110.0 bits without leases, 269.5-280.8 with the T=256 lease range).
Usage: python3 hz_estimate.py [pairs per T]"""
import math
import random
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
from state_machine import GROUPS, PRINCIPALS, RESOURCES, make_pair_v2  # noqa: E402
from verify_nq import Filter  # noqa: E402


def components(f):
    out = {}
    for p in PRINCIPALS:
        for r in RESOURCES:
            out[("d", p, r)] = f.direct.get((p, r))
        for g in GROUPS:
            out[("m", p, g)] = (p, g) in f.mem
    for g in GROUPS:
        for r in RESOURCES:
            out[("gg", g, r)] = (g, r) in f.gg
    for r in RESOURCES:
        out[("nw", r)] = r in f.nw
        out[("op", r)] = r in f.op
    return out


def H(counter, n):
    return -sum(c / n * math.log2(c / n) for c in counter.values())


if __name__ == "__main__":
    n_pairs = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    for T in (8, 16, 64, 256):
        counts, n, samples = defaultdict(Counter), 0, defaultdict(list)
        for i in range(n_pairs):
            for ep in make_pair_v2(2026900000 + T * 10000 + i, T):
                f = Filter()
                for ev in ep["events"]:
                    ev = dict(ev)
                    if "until_rt" in ev:
                        ev["until"] = ev["until_rt"]
                    f.step(ev)
                for k, v in components(f).items():
                    counts[k][v] += 1
                    samples[k].append(v)
                n += 1
        per = {k: H(c, n) for k, c in counts.items()}
        mm = {k: per[k] + (len(c) - 1) / (2 * n * math.log(2)) for k, c in counts.items()}
        d = sum(v for k, v in per.items() if k[0] == "d")
        aux = sum(v for k, v in per.items() if k[0] != "d")
        boots = []
        rng = random.Random(T)
        keys = list(counts)
        for _ in range(200):
            idx = [rng.randrange(n) for _ in range(n)]
            tot = 0.0
            for k in keys:
                c = Counter(samples[k][i] for i in idx)
                tot += H(c, n) + (len(c) - 1) / (2 * n * math.log(2))
            boots.append(tot)
        boots.sort()
        print(f"T={T:<4} episodes={n}: plug-in sum {d + aux:6.1f} bits (direct {d:5.1f}, groups/windows {aux:5.1f}); "
              f"Miller-Madow {sum(mm.values()):6.1f} bits, bootstrap 95% [{boots[4]:.1f}, {boots[194]:.1f}]")
