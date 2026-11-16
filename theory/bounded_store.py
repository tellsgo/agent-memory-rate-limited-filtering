#!/usr/bin/env python3
"""Irreversible bounded recursive store (theory.tex, def:budget generalized, prop:store).

Definition (the one sent to the experiments session for experiments/r5_bounded.py):
  components d(p,r) (allowed, rank, until, src), m(p,g), gg(g,r), w(r) = (need, open); absent = default.
  An event on c is processed by the exact rule of replay() against store[c], or against the default if c is
  absent (never written, evicted or deleted). REMOVE deletes its membership record (frees a slot); every other
  applied event writes its record (a window record even when it is (False, False)), an update if c is stored and
  otherwise an INSERTION, which first evicts one victim when |store| = C. Matches experiments/r5_bounded.py.
  A blocked direct event and a same-value rewrite are touches. Filler lines do nothing.
  Policies: random (uniform victim), lru (earliest last touch), default-first (bottom-equivalent records first,
  rank-1 withdrawals and grants expired at t, by LRU; otherwise LRU).
  Query after the last event: closed-world answer of the stored records, expiry resolved at t_query = T-1.

Modes:
  python3 bounded_store.py table     factor 1 - 2 P_e of prop:budget for the store on the X4c backgrounds,
                                     C = round(f * mean |S_T| of the cell), next to the read-time projection
  python3 bounded_store.py uniform   prop:store(iv): stores against time sharing and the converse eq:rd on the
                                     uniform direct-only source (K = 24, L = 3)
"""
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "experiments")]
from state_machine import RANK, answer_from_state, make_pair_v2, replay_events  # noqa: E402
from budget_predict import F_GRID, components  # noqa: E402

DIRECT = ("GRANT", "GRANT_UNTIL", "WITHDRAW", "RECONFIRM", "ALARM_WITHDRAW")


class Store:
    def __init__(self, C, policy, rng):
        self.C, self.policy, self.rng = C, policy, rng
        self.rec, self.touch = {}, {}
        self.insertions = self.evictions = 0
        self.evicted = []                      # (time, component)

    def _bottom(self, c, t):
        if c[0] != "d":
            return False
        allowed, rank, until, _ = self.rec[c]
        return (not allowed and rank == 1) or (allowed and until is not None and until < t)

    def _victim(self, t):
        if self.policy == "random":
            return self.rng.choice(sorted(self.rec))
        pool = list(self.rec)
        if self.policy == "default-first":
            low = [c for c in pool if self._bottom(c, t)]
            pool = low or pool
        return min(pool, key=lambda c: self.touch[c])

    def _write(self, c, v, t):
        if c in self.rec:
            self.rec[c] = v
            self.touch[c] = t
            return
        if self.C == 0:
            return
        if len(self.rec) >= self.C:
            x = self._victim(t)
            del self.rec[x], self.touch[x]
            self.evictions += 1
            self.evicted.append((t, x))
        self.rec[c] = v
        self.touch[c] = t
        self.insertions += 1

    def _delete(self, c):
        self.rec.pop(c, None)
        self.touch.pop(c, None)

    def step(self, ev):
        k, t = ev["type"], ev["t"]
        if k in DIRECT:
            c = ("d", ev["p"], ev["r"])
            rank = RANK[ev["src"]]
            cur = self.rec.get(c)
            cur_rank = 0 if cur is None or (cur[2] is not None and cur[2] < t) else cur[1]
            if rank >= cur_rank:
                self._write(c, (k in ("GRANT", "GRANT_UNTIL", "RECONFIRM"), rank, ev.get("until"), ev["src"]), t)
            else:
                self.touch[c] = t              # blocked: a touch, c is stored because cur_rank > 0
        elif k == "GROUP_GRANT":
            self._write(("gg", ev["g"], ev["r"]), True, t)
        elif k in ("MEMBER", "ADD"):
            self._write(("m", ev["p"], ev["g"]), True, t)
        elif k == "REMOVE":
            self._delete(("m", ev["p"], ev["g"]))
        elif k in ("REQ_WINDOW", "OPEN", "CLOSE"):
            c = ("w", ev["r"])
            need, op = self.rec.get(c, (False, False))
            if k == "REQ_WINDOW":
                need = True
            else:
                op = k == "OPEN"
            self._write(c, (need, op), t)       # a window event always writes its record, even (False, False)

    def state(self, tq):
        st = {"direct": {}, "members": set(), "group_grants": set(), "need_window": set(), "window_open": set()}
        for c, v in self.rec.items():
            if c[0] == "d":
                allowed, rank, until, src = v
                exp = allowed and until is not None and until < tq
                st["direct"][(c[1], c[2])] = {"allowed": allowed and not exp, "src": src, "until": until, "expired": exp}
            elif c[0] == "m":
                st["members"].add((c[1], c[2]))
            elif c[0] == "gg":
                st["group_grants"].add((c[1], c[2]))
            else:
                if v[0]:
                    st["need_window"].add(c[1])
                if v[1]:
                    st["window_open"].add(c[1])
        return st


def run_store(evs, T, C, policy, rng):
    s = Store(C, policy, rng)
    for ev in evs:
        s.step(ev)
    return s


def store_answer(ep, C, policy, rng):
    s = run_store(replay_events(ep), ep["T"], C, policy, rng)
    return answer_from_state(s.state(ep["T"] - 1), ep["p"], ep["r"]), s


def cell_episodes(K, T, pairs=40):
    eps = []
    for i in range(pairs):
        eps += make_pair_v2(2026160000 + T * 10000 + K * 100000 + i, T, bg_keys=K, bg_access_only=True)
    return eps


def factor(eps, C_of, policy, reps, rng):
    errs = {"APPROVE": [], "DENY": []}
    for ep in eps:
        C = C_of(ep)
        n = reps if policy == "random" else 1
        bad = sum(store_answer(ep, C, policy, rng)[0] != ep["truth"] for _ in range(n)) / n
        errs[ep["truth"]].append(bad)
    pe = 0.5 * (sum(errs["APPROVE"]) / len(errs["APPROVE"]) + sum(errs["DENY"]) / len(errs["DENY"]))
    return 1 - 2 * pe


def table(reps=30):
    rng = random.Random(0)
    cells = [(K, T) for K in (3, 7, 15) for T in (64, 256)] + [(15, 1024)]
    print("store factor 1 - 2 P_e, C = round(f * mean |S_T| of the cell); f grid", F_GRID)
    for K, T in cells:
        eps = cell_episodes(K, T)
        sizes = [len(components(ep)[1]) for ep in eps]
        mean_s = sum(sizes) / len(sizes)
        peak = 0
        for ep in eps:
            s = run_store(replay_events(ep), T, 10 ** 9, "lru", rng)
            peak = max(peak, len(s.rec))
        for pol in ("random", "lru", "default-first"):
            row = [factor(eps, lambda ep, f=f: int(math.floor(f * mean_s + 0.5)), pol, reps, rng) for f in F_GRID]
            print(f"{pol:13s} K={K:2d} T={T:4d} mean|S|={mean_s:5.1f} max|S|={max(sizes):2d} "
                  + " ".join(f"{x:6.3f}" for x in row), flush=True)


# ---------------- uniform direct-only source (prop:store(iv)) ----------------

def phi(d, n):
    h = 0.0 if d in (0, 1) else -d * math.log2(d) - (1 - d) * math.log2(1 - d)
    return h + d * math.log2(n - 1)


def phi_inv(x, n):
    if x <= 0:
        return 0.0
    lo, hi = 0.0, 1 - 1 / n
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if phi(mid, n) < x else (lo, mid)
    return (lo + hi) / 2


def uniform(K=24, L=3):
    n = 2 * L
    q = 1 - 1 / n
    binom = [math.comb(K, m) * q ** m * (1 - q) ** (K - m) for m in range(K + 1)]
    print(f"uniform source K={K} n={n}: rows R/K, time sharing, store (one event per key), converse")
    rows = []
    for C in range(K + 1):
        r_ts = C * math.log2(n) / K
        d_ts = (1 - C / K) * q
        r_st = math.log2(sum(math.comb(K, j) * (n - 1) ** j for j in range(C + 1))) / K
        d_st = sum(binom[m] * max(0, m - C) for m in range(K + 1)) / K
        rows.append((C, r_ts, d_ts, r_st, d_st))
        print(f"C={C:2d}  TS: R/K={r_ts:5.3f} err={d_ts:.3f} conv={phi_inv(math.log2(n) - r_ts, n):.3f}   "
              f"store: R/K={r_st:5.3f} err={d_st:.3f} conv={phi_inv(math.log2(n) - r_st, n):.3f}")
    half = math.log2(n) / 2
    ts = max((r for r in rows if r[1] <= half + 1e-12), key=lambda r: r[1])
    st = max((r for r in rows if r[3] <= half + 1e-12), key=lambda r: r[3])
    print(f"at R/K = log2(n)/2: time sharing C={ts[0]} err={ts[2]:.3f}; store C={st[0]} err={st[4]:.3f}; "
          f"converse {phi_inv(math.log2(n) - half, n):.3f}")


if __name__ == "__main__":
    {"table": table, "uniform": uniform}[sys.argv[1] if len(sys.argv) > 1 else "table"]()
