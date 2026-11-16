#!/usr/bin/env python3
"""Monte Carlo check of prop:parser (theory.tex) on the exact lease-free rule for one key.
(a) one rank (L = 1, delta_l = 0), alternating chain: closed form vs simulation.
(b) L = 3, delta_l = 0.06: top-rank decisive event followed by B blocked events with the opposite decision,
    bounds vs simulation; and alternating chains at the top and at the lowest rank, upper bound vs simulation."""
import random


def replay(evs):
    cur = None
    for a, l in evs:
        if cur is None or l >= cur[1]:
            cur = (a, l)
    return 0 if cur is None else cur[0]


def parse(ev, d0, da, dl, L, rng):
    a, l = ev
    u = rng.random()
    if u < d0:
        return None
    if u < d0 + da:
        return (1 - a, l)
    if u < d0 + da + dl:
        return (a, rng.choice([x for x in range(1, L + 1) if x != l]))
    return ev


def sim(evs, d0, da, dl, L, n=200000, seed=1):
    rng = random.Random(seed)
    truth, e = replay(evs), 0
    for _ in range(n):
        p = [x for x in (parse(v, d0, da, dl, L, rng) for v in evs) if x is not None]
        e += replay(p) != truth
    return e / n


if __name__ == "__main__":
    d0, da = 0.1, 0.05
    for N in (3, 9, 33):
        evs = [(1 if j % 2 == 0 else 0, 1) for j in range(N)]
        aN = evs[-1][0]
        f = sum(d0 ** k * (da if evs[N - 1 - k][0] == aN else (1 - d0 - da)) for k in range(N)) + d0 ** N * (aN == 1)
        print(f"(a) N={N}: formula {f:.4f}  simulation {sim(evs, d0, da, 0, 1):.4f}")
    L, dl, a_star = 3, 0.06, 1
    g = 1 - d0 - da - dl
    for B in (2, 8, 32):
        evs = [(a_star, 3)] + [(1 - a_star, 1 if j % 2 else 2) for j in range(B)]
        ps = dl * (L - 3 + 1) / (L - 1)
        print(f"(b) B={B}: lower {g * (1 - (1 - ps) ** B):.4f}  upper {1 - g * (1 - ps) ** B:.4f}  simulation {sim(evs, d0, da, dl, L):.4f}")
    for r in (3, 1):
        for N in (3, 9, 33):
            evs = [(1 if j % 2 == 0 else 0, r) for j in range(N)]
            p_gt = dl * (L - r) / (L - 1)          # an earlier event misread above the decisive (last) one
            print(f"(b) L=3 chain at rank {r}, N={N}: upper {1 - g * (1 - p_gt) ** (N - 1):.4f}  simulation {sim(evs, d0, da, dl, L):.4f}")
