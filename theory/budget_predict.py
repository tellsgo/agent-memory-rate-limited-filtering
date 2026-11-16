#!/usr/bin/env python3
"""Registrable prediction for a budgeted filter (theory.tex, prop:budget).

A budgeted filter holds the exact filter state but keeps only k = floor(f*|S| + 1/2) of its |S| rendered state
lines (render_state). Missing lines are read as absent (closed world). Two eviction policies:
  random  a uniformly random k-subset of the lines (exact enumeration, Monte Carlo above 20 000 subsets)
  recent  the k lines whose component was last touched most recently (ties broken by line order)
For each episode, p_e(f) = probability that the closed-world answer of the kept lines differs from the truth.
P_e(f) = (mean of p_e over APPROVE episodes + mean over DENY episodes) / 2 is the balanced error of an exact
reader of the kept lines. Under the reader model of prop:budget,

    BA(f) - 1/2 = (BA(1) - 1/2) * (1 - 2 P_e(f)),

so the curve is fixed by one measured number, BA(1), per reader and cell.
Episodes are those of X4c: make_pair_v2(2026160000 + T*10000 + K*100000 + i, T, bg_keys=K, bg_access_only=True).
Usage: python3 budget_predict.py [pairs]    (default 40, as in X4c)"""
import itertools
import math
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
from state_machine import answer_from_state, filter_state, make_pair_v2, replay_events  # noqa: E402

F_GRID = [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]


def components(ep):
    """State components as rendered lines: (kind, id) -> last time any event touched the component."""
    evs = replay_events(ep)
    T = ep["T"]
    st = filter_state(evs, T - 1)
    last = {}
    for ev in evs:
        k, t = ev["type"], ev["t"]
        if k in ("GRANT", "GRANT_UNTIL", "WITHDRAW", "RECONFIRM", "ALARM_WITHDRAW"):
            last[("d", ev["p"], ev["r"])] = t
        elif k in ("MEMBER", "ADD", "REMOVE"):
            last[("m", ev["p"], ev["g"])] = t
        elif k == "GROUP_GRANT":
            last[("gg", ev["g"], ev["r"])] = t
        elif k in ("REQ_WINDOW", "OPEN", "CLOSE"):
            last[("w", ev["r"])] = max(t, last.get(("w", ev["r"]), -1))
    comps = [("d",) + key for key in sorted(st["direct"])]
    comps += [("m",) + pg for pg in sorted(st["members"])]
    comps += [("gg",) + gr for gr in sorted(st["group_grants"])]
    comps += [("w", r) for r in sorted(st["need_window"])]
    return st, comps, {c: last.get(c, -1) for c in comps}


def answer_kept(st, kept, q):
    sub = {"direct": {k: v for k, v in st["direct"].items() if ("d",) + k in kept},
           "members": {pg for pg in st["members"] if ("m",) + pg in kept},
           "group_grants": {gr for gr in st["group_grants"] if ("gg",) + gr in kept},
           "need_window": {r for r in st["need_window"] if ("w", r) in kept},
           "window_open": {r for r in st["window_open"] if ("w", r) in kept}}
    return answer_from_state(sub, *q)


def p_err(st, comps, truth, q, k, rng, mc=20000):
    n = len(comps)
    if k >= n:
        return 0.0 if answer_kept(st, set(comps), q) == truth else 1.0
    total = math.comb(n, k)
    if total <= mc:
        bad = sum(answer_kept(st, set(c), q) != truth for c in itertools.combinations(comps, k))
        return bad / total
    bad = sum(answer_kept(st, set(rng.sample(comps, k)), q) != truth for _ in range(mc))
    return bad / mc


if __name__ == "__main__":
    pairs = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    rng = random.Random(0)
    print("policy  K   T   |S| mean  " + "  ".join(f"f={f:<5}" for f in F_GRID))
    for K in (3, 7, 15):
        for T in (64, 256):
            eps = []
            for i in range(pairs):
                eps += make_pair_v2(2026160000 + T * 10000 + K * 100000 + i, T, bg_keys=K, bg_access_only=True)
            rows = {"random": [], "recent": []}
            sizes = []
            for ep in eps:
                st, comps, last = components(ep)
                q = (ep["p"], ep["r"])
                sizes.append(len(comps))
                order = sorted(comps, key=lambda c: (-last[c], comps.index(c)))
                for pol in rows:
                    r = []
                    for f in F_GRID:
                        k = int(math.floor(f * len(comps) + 0.5))
                        if pol == "random":
                            r.append(p_err(st, comps, ep["truth"], q, k, rng))
                        else:
                            r.append(0.0 if answer_kept(st, set(order[:k]), q) == ep["truth"] else 1.0)
                    rows[pol].append((ep["truth"], r))
            for pol, rr in rows.items():
                pe = []
                for j in range(len(F_GRID)):
                    a = [r[j] for y, r in rr if y == "APPROVE"]
                    d = [r[j] for y, r in rr if y == "DENY"]
                    pe.append((sum(a) / len(a) + sum(d) / len(d)) / 2)
                print(f"{pol:<7} {K:<3} {T:<4} {sum(sizes) / len(sizes):5.1f}    " + "  ".join(f"{1 - 2 * x:7.3f}" for x in pe))
    print("\nentries are the factor 1 - 2 P_e(f); predicted BA(f) = 1/2 + (BA(1) - 1/2) * factor")
