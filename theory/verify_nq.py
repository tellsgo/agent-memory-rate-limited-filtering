#!/usr/bin/env python3
"""
Machine checks for the counting results in theory.tex, run against the benchmark's own replay().

  check_filter     the reduced filter (Prop. 3 state) answers every query exactly like replay()
                   on random long histories with the full v2 alphabet (6 principals, 4 resources,
                   4 groups, 3 ranks, leases)
  check_no_lease   one key, one group, L = 3, no leases: reachable reduced states = distinct anytime
                   signatures = 2L * 2^4 = 96
  check_lease      one key, no groups or windows, L = 2, lease bound 3, and L = 3, lease bound 2: reachable
                   reduced states = distinct signatures = L(Lambda + 2) + 1 = 11 and 13
  check_pruning    why the enumeration runs over exact reduced states, not over signatures
  check_mask       the masking collapse of Remark 2: with two granting groups, an expiring-now grant
                   and no entry are indistinguishable by every continuation up to length 4

Signatures are answers to the target query after every prefix of every continuation up to length k
(including the empty one). The enumeration runs over exact reduced states; see count_classes.
This is a check of the proof, not a substitute for it.
"""

import itertools
import os
import random
import sys

sys.path[:0] = [os.path.join(os.path.dirname(__file__), "..", d) for d in ("experiments", "data")]  # data/ in the audit package
from state_machine import GROUPS, PRINCIPALS, RANK, RESOURCES, replay  # noqa: E402

SRC_OF = {v: k for k, v in RANK.items()}
DIRECT = ("GRANT", "GRANT_UNTIL", "WITHDRAW", "RECONFIRM", "ALARM_WITHDRAW")


# --- reduced filter (theory.tex, Proposition 3) ------------------------------------------------
# direct state: None (no entry, equal to a rank-1 withdrawal), ("D", l) for l >= 2,
# ("A", l, rho) with rho in {1..Lam} or None (= no expiry), ("A0",) granted and expiring now.

def _age(d):
    if d is None or d[0] == "D":
        return d
    if d[0] == "A0":
        return None
    l, rho = d[1], d[2]
    if rho is None:
        return d
    return ("A", l, rho - 1)          # rho - 1 = 0: still holds rank for this step's event


def _rank(d):
    if d is None or d[0] == "A0":
        return 0
    return d[1]


def _norm(allowed, l, rho):
    if not allowed:
        return None if l == 1 else ("D", l)
    if rho is None:
        return ("A", l, None)
    if rho < 0:
        return None
    return ("A0",) if rho == 0 else ("A", l, rho)


class Filter:
    def __init__(self):
        self.direct, self.mem, self.gg, self.nw, self.op = {}, set(), set(), set(), set()

    def step(self, ev):
        self.direct = {k: _age(d) for k, d in self.direct.items()}
        k = ev["type"]
        if k in DIRECT:
            key, l = (ev["p"], ev["r"]), RANK[ev["src"]]
            cur = self.direct.get(key)
            if l >= _rank(cur):
                allowed = k in ("GRANT", "GRANT_UNTIL", "RECONFIRM")
                rho = None if ev.get("until") is None else ev["until"] - ev["t"]
                self.direct[key] = _norm(allowed, l, rho)
        elif k == "GROUP_GRANT":
            self.gg.add((ev["g"], ev["r"]))
        elif k in ("MEMBER", "ADD"):
            self.mem.add((ev["p"], ev["g"]))
        elif k == "REMOVE":
            self.mem.discard((ev["p"], ev["g"]))
        elif k == "REQ_WINDOW":
            self.nw.add(ev["r"])
        elif k == "OPEN":
            self.op.add(ev["r"])
        elif k == "CLOSE":
            self.op.discard(ev["r"])
        # end of step: a grant valid through now is equivalent for every rank (it expires next step)
        self.direct = {k: (("A0",) if d is not None and d[0] == "A" and d[2] == 0 else d)
                       for k, d in self.direct.items()}

    def answer(self, p, r):
        d = self.direct.get((p, r))
        perm = d is not None and d[0] in ("A", "A0")
        perm = perm or any((p, g) in self.mem and (g, r) in self.gg for g in GROUPS)
        if r in self.nw and r not in self.op:
            perm = False
        return "APPROVE" if perm else "DENY"


def random_event(rng, t, lam):
    u = rng.random()
    src = rng.choice(list(RANK))
    p, r, g = rng.choice(PRINCIPALS), rng.choice(RESOURCES), rng.choice(GROUPS)
    if u < 0.45:
        typ = rng.choice(DIRECT)
        ev = dict(type=typ, src=src, p=p, r=r, t=t)
        if typ == "GRANT_UNTIL":
            ev["until"] = t + rng.randint(-1, lam)          # includes an already-over lease
        return ev
    if u < 0.55:
        return dict(type="GROUP_GRANT", src=src, g=g, r=r, t=t)
    if u < 0.75:
        return dict(type=rng.choice(["MEMBER", "ADD", "REMOVE"]), src=src, p=p, g=g, t=t)
    if u < 0.9:
        return dict(type=rng.choice(["REQ_WINDOW", "OPEN", "CLOSE"]), src=src, r=r, t=t)
    return dict(type="FILLER", t=t)


def check_filter(n_hist=300, T=120, lam=12, seed=0):
    rng = random.Random(seed)
    n_cmp = 0
    for _ in range(n_hist):
        f, evs = Filter(), []
        for t in range(T):
            ev = random_event(rng, t, lam)
            evs.append(ev)
            f.step(ev)
            if t % 7 == 0 or t == T - 1:
                for p in PRINCIPALS:
                    for r in RESOURCES:
                        assert f.answer(p, r) == replay(evs, p, r, t), (t, p, r)
                        n_cmp += 1
    return n_cmp


# --- signature counting on tiny instances --------------------------------------------------------

def signature(hist, alphabet, k, p, r):
    t0 = len(hist)
    sig = [replay(hist, p, r, t0 - 1) if hist else "DENY"]
    for n in range(1, k + 1):
        for w in itertools.product(alphabet, repeat=n):
            evs = list(hist)
            for j, sym in enumerate(w):
                evs.append(sym(t0 + j))
            sig.append(replay(evs, p, r, t0 + n - 1))
    return tuple(sig)


def filter_key(f):
    """Exact reduced state of a Filter (absent and None direct entries are the same state)."""
    direct = tuple(sorted(((k, d) for k, d in f.direct.items() if d is not None), key=repr))
    return direct, frozenset(f.mem), frozenset(f.gg), frozenset(f.nw), frozenset(f.op)


def run_filter(hist):
    f = Filter()
    for ev in hist:
        f.step(ev)
    return f


def count_classes(alphabet, cont_alphabet, k, p, r):
    """Enumerate the reachable graph of exact reduced states (one representative history each), then
    count distinct bounded signatures over the representatives.

    Returns (states, signatures). Reachable reduced states bound N_Q from above (the reduced filter is
    a congruence that answers exactly, and check_filter tests it against replay()). Distinct
    signatures bound N_Q from below (each distinct pair is separated by an explicit continuation).
    Equal numbers give the exact count. Pruning histories by bounded signature instead is unsound:
    two states with equal k-signatures can have different successors (check_pruning shows one)."""
    reps = {filter_key(Filter()): []}
    frontier = [[]]
    while frontier:
        nxt = []
        for h in frontier:
            for sym in alphabet:
                h2 = h + [sym(len(h))]
                key = filter_key(run_filter(h2))
                if key not in reps:
                    reps[key] = h2
                    nxt.append(h2)
        frontier = nxt
    sigs = {signature(h, cont_alphabet, k, p, r) for h in reps.values()}
    return len(reps), len(sigs)


def mk(typ, **kw):
    return lambda t: dict(type=typ, t=t, **kw)


def mk_lease(src, p, r, lam):
    return lambda t: dict(type="GRANT_UNTIL", src=src, p=p, r=r, t=t, until=t + lam)


def check_no_lease():
    p, r, g = PRINCIPALS[0], RESOURCES[0], GROUPS[0]
    al = [mk(typ, src=s, p=p, r=r) for typ in ("GRANT", "WITHDRAW") for s in RANK]
    al += [mk("GROUP_GRANT", src="access_admin", g=g, r=r), mk("ADD", src="access_admin", p=p, g=g),
           mk("REMOVE", src="access_admin", p=p, g=g), mk("REQ_WINDOW", src="access_admin", r=r),
           mk("OPEN", src="access_admin", r=r), mk("CLOSE", src="access_admin", r=r), mk("FILLER")]
    return count_classes(al, al, 3, p, r)


def check_lease(lam=3, srcs=("ops_console", "access_admin")):    # L = len(srcs), default 2
    p, r = PRINCIPALS[0], RESOURCES[0]
    plain = [mk(typ, src=s, p=p, r=r) for typ in ("GRANT", "WITHDRAW") for s in srcs] + [mk("FILLER")]
    al = plain + [mk_lease(s, p, r, l) for s in srcs for l in range(lam + 1)]
    return count_classes(al, plain, lam + 2, p, r)


def check_mask():
    p, r = PRINCIPALS[0], RESOURCES[0]
    g1, g2 = GROUPS[0], GROUPS[1]
    setup = [mk("GROUP_GRANT", src="access_admin", g=g1, r=r), mk("GROUP_GRANT", src="access_admin", g=g2, r=r),
             mk("ADD", src="access_admin", p=p, g=g1), mk("ADD", src="access_admin", p=p, g=g2)]
    hx = [s(t) for t, s in enumerate(setup)]
    hx.append(mk_lease("security_office", p, r, 0)(4))        # granted, expiring now
    hy = [s(t) for t, s in enumerate(setup)] + [mk("FILLER")(4)]  # no direct entry
    cont = [mk(typ, src=s, p=p, r=r) for typ in ("GRANT", "WITHDRAW") for s in RANK]
    cont += [mk("REMOVE", src="access_admin", p=p, g=g) for g in (g1, g2)] + [mk("FILLER")]
    return signature(hx, cont, 4, p, r) == signature(hy, cont, 4, p, r)


def check_pruning():
    """The reviewer's counterexample to signature pruning: rank-one leases of length 2 and 1 plus null,
    null-only continuations. With k = 1 two reachable states share a signature, with k = 2 they do not."""
    p, r = PRINCIPALS[0], RESOURCES[0]
    al = [mk_lease("ops_console", p, r, 2), mk_lease("ops_console", p, r, 1), mk("FILLER")]
    return count_classes(al, [mk("FILLER")], 1, p, r), count_classes(al, [mk("FILLER")], 2, p, r)


if __name__ == "__main__":
    print("filter == replay on", check_filter(), "query checks")
    print("no-lease (pred 96): reachable reduced states, distinct signatures =", check_no_lease())
    print("lease L=2, Lambda=3 (pred 11): reachable reduced states, distinct signatures =", check_lease())
    print("lease L=3, Lambda=2 (pred 13): reachable reduced states, distinct signatures =",
          check_lease(2, ("ops_console", "access_admin", "security_office")))
    print("pruning counterexample (k=1, k=2):", check_pruning())
    print("masking collapse holds:", check_mask())
