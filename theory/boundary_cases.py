#!/usr/bin/env python3
"""Boundary cases of the round-5 repairs, checked on the exact rule (replay() of the benchmark).

  A  prop:recency needs the non-masking hypotheses: group access, a closed required window or an expiring lease
     on e* makes the two versions agree; under the hypotheses they differ. The X5 blocked family satisfies them.
  B  conj:lease is false for L = 1 (last event per key with expiry at query time is exact, exhaustive check),
     and the two L >= 2 counterexamples of the text hold.
  C  rem:parser, corrected lease claim: true expiry 10, parsed 20 -> decisions differ exactly on 11..20; with a
     lower-rank event between the expiries the difference persists past both.
  D  prop:plugin-map: with known event positions rho = 1 occurs and the posterior ties; with unknown positions
     MAP equals the plug-in answer on every parsed stream (exhaustive, small model).
  E  prop:store: DENY -> APPROVE by eviction; LRU survival iff fewer than C distinct other components are touched
     after the last touch (no deletions); random-eviction survival E_Q[(1-1/C)^N]; exact when C >= peak size.
  F  counterexamples of the blind review: (F1) prop:recency without the non-masking hypothesis: a closed required
     window makes both versions DENY, so a reader that always answers DENY has error 0, below the bound 1/2;
     (F2) prop:parser(b): the exact error is not monotone in N_after (L = 2, rank-2 grant, N rank-1 withdrawals);
     (F3) prop:store(b) needs s = last touch of x: C = 2, x at 1, y at 2, s = 2, z at 3 evicts x.
Usage: python3 boundary_cases.py
"""
import itertools
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "experiments")]
from state_machine import make_blocked_pair, replay, replay_events  # noqa: E402
from bounded_store import Store, cell_episodes, run_store, store_answer  # noqa: E402
from state_machine import answer_from_state  # noqa: E402

SRC = {1: "ops_console", 2: "access_admin", 3: "security_office"}
P0, R0, G0 = "svc_kestrel", "db_ledger", "grp_alpha"
OK = []


def check(name, cond):
    OK.append(cond)
    print(("PASS " if cond else "FAIL ") + name)


def d(t, kind, rank, p=P0, r=R0, until=None):
    ev = {"t": t, "type": kind, "src": SRC[rank], "p": p, "r": r}
    if until is not None:
        ev["type"] = "GRANT_UNTIL"
        ev["until"] = until
    return ev


def ans(evs, tq):
    return replay(evs, P0, R0, tq)


# ---------------- A ----------------
def case_a():
    base = lambda a: [d(0, "GRANT" if a else "WITHDRAW", 3), d(3, "WITHDRAW", 1), d(5, "GRANT", 1)]  # noqa: E731
    differ = ans(base(1), 9) != ans(base(0), 9)
    check("A0 non-masked versions differ (APPROVE/DENY)", differ and ans(base(1), 9) == "APPROVE")
    grp = [{"t": 6, "type": "GROUP_GRANT", "src": SRC[2], "g": G0, "r": R0}, {"t": 7, "type": "ADD", "src": SRC[2], "p": P0, "g": G0}]
    check("A1 group access of p to r masks e* (both APPROVE)", ans(base(1) + grp, 9) == ans(base(0) + grp, 9) == "APPROVE")
    win = [{"t": 6, "type": "REQ_WINDOW", "src": SRC[2], "r": R0}]
    check("A2 closed required window masks e* (both DENY)", ans(base(1) + win, 9) == ans(base(0) + win, 9) == "DENY")
    lease = [d(0, "GRANT", 3, until=4), d(5, "WITHDRAW", 1)]
    lease0 = [d(0, "WITHDRAW", 3), d(5, "WITHDRAW", 1)]
    check("A3 lease on e* expiring before a later withdrawal: the withdrawal applies, versions agree (both DENY)",
          ans(lease, 9) == ans(lease0, 9) == "DENY")
    lease_q = [d(0, "GRANT", 3, until=7), d(3, "WITHDRAW", 1)]
    check("A4 lease on e* expiring before the query: versions agree (both DENY)",
          ans(lease_q, 9) == ans([d(0, "WITHDRAW", 3), d(3, "WITHDRAW", 1)], 9) == "DENY")
    lease_ok = [d(0, "GRANT", 3, until=20), d(3, "WITHDRAW", 1)]
    check("A5 lease covering the query keeps the versions apart", ans(lease_ok, 9) == "APPROVE")
    bad = 0
    n = 0
    for bp in (2, 8, 32):
        for T in (64, 256):
            for i in range(20):
                for ep in make_blocked_pair(2026150000 + 1000 * bp + 10 * T + i, bp, T):
                    n += 1
                    p, r = ep["p"], ep["r"]
                    evs = replay_events(ep)
                    on_key = [e for e in evs if e.get("p") == p and e.get("r") == r and e["type"] != "FILLER"]
                    top = [e for e in on_key if e["src"] == "security_office"]
                    cond = len(top) == 1 and len(on_key) == bp + 1
                    cond = cond and (top[0].get("until") is None or top[0]["until"] >= T - 1)
                    cond = cond and not any(e["type"] in ("ADD", "MEMBER", "REMOVE") and e.get("p") == p for e in evs)
                    cond = cond and not any(e["type"] in ("GROUP_GRANT", "REQ_WINDOW", "OPEN", "CLOSE") and e.get("r") == r for e in evs)
                    bad += not cond
    check(f"A6 X5 blocked family satisfies the hypotheses ({n} episodes)", bad == 0)


# ---------------- B ----------------
def last_event_answer(evs, tq):
    on = [e for e in evs if e.get("p") == P0 and e.get("r") == R0]
    if not on:
        return "DENY"
    e = on[-1]
    ok = e["type"] in ("GRANT", "GRANT_UNTIL", "RECONFIRM") and (e.get("until") is None or e["until"] >= tq)
    return "APPROVE" if ok else "DENY"


def case_b():
    kinds = [("WITHDRAW", None), ("GRANT", None)] + [("GRANT", lam) for lam in range(4)]
    n = bad = 0
    for T in range(1, 6):
        for seq in itertools.product(range(len(kinds) + 1), repeat=T):     # last index = filler
            evs = []
            for t, j in enumerate(seq):
                if j < len(kinds):
                    k, lam = kinds[j]
                    evs.append(d(t, k, 1, until=None if lam is None else t + lam))
            for tq in range(T - 1, T + 5):
                n += 1
                bad += ans(evs, tq) != last_event_answer(evs, tq)
    check(f"B1 L = 1: last event per key with expiry at query time is exact ({n} history-query pairs)", bad == 0)
    v_after = [d(0, "GRANT", 3, until=10), d(15, "GRANT", 1)]
    v_before = [d(0, "GRANT", 3, until=10), d(5, "GRANT", 1)]
    check("B2 (rank, time) score: versions differ (APPROVE after / DENY before) while top line is the lease grant",
          ans(v_after, 20) == "APPROVE" and ans(v_before, 20) == "DENY")
    w_after = [d(2, "WITHDRAW", 2), d(3, "GRANT", 3, until=10), d(15, "GRANT", 1)]
    w_before = [d(2, "WITHDRAW", 2), d(3, "GRANT", 3, until=10), d(5, "GRANT", 1)]
    # expiry-discounted score: drop grants expired at query time, then top-1 by (rank, time) = the rank-2 withdrawal
    check("B3 expiry-discounted score: versions differ while top line is the rank-2 withdrawal in both",
          ans(w_after, 20) == "APPROVE" and ans(w_before, 20) == "DENY")


# ---------------- C ----------------
def case_c():
    true = [d(0, "GRANT", 3, until=10)]
    parsed = [d(0, "GRANT", 3, until=20)]
    diff = [tq for tq in range(1, 40) if ans(true, tq) != ans(parsed, tq)]
    check("C1 true expiry 10, parsed 20: decisions differ exactly on 11..20", diff == list(range(11, 21)))
    true2 = true + [d(15, "GRANT", 1)]
    parsed2 = parsed + [d(15, "GRANT", 1)]
    diff2 = [tq for tq in range(15, 60) if ans(true2, tq) != ans(parsed2, tq)]
    check("C2 rank-1 grant at 15: applied in the true stream, blocked in the parsed one; differ on 21..59 too",
          diff2 == list(range(21, 60)))


# ---------------- D ----------------
def posterior_known(N, parsed, d0, da):
    """Exact posterior P(answer = APPROVE | parse) with N events at known positions, one rank, i.i.d. uniform
    decisions. parsed[j] in {None (dropped), 0, 1}."""
    num = den = 0.0
    for a in itertools.product((0, 1), repeat=N):
        w = 0.5 ** N
        for j in range(N):
            x = parsed[j]
            w *= d0 if x is None else ((1 - d0 - da) if x == a[j] else da)
        den += w
        num += w * a[-1]
    return num / den


def plugin(parsed):
    xs = [x for x in parsed if x is not None]
    return xs[-1] if xs else 0


def case_d():
    d0, da = 0.2, 0.1
    pa = posterior_known(2, (1, None), d0, da)
    check(f"D1 known positions, last event dropped: posterior {pa:.3f} ties (rho = 1)", abs(pa - 0.5) < 1e-12)
    # unknown positions: each of T slots holds an event on kappa with probability pi; a dropped event and a
    # filler are both parsed as null
    T, pi = 4, 0.5
    bad = ties = 0
    for obs in itertools.product((None, 0, 1), repeat=T):
        num = den = 0.0
        for occ in itertools.product((0, 1), repeat=T):
            for a in itertools.product((0, 1), repeat=T):
                w = 1.0
                for j in range(T):
                    if occ[j]:
                        w *= pi * 0.5
                        w *= d0 if obs[j] is None else ((1 - d0 - da) if obs[j] == a[j] else da)
                    else:
                        w *= (1 - pi) * 0.5 * (1.0 if obs[j] is None else 0.0)    # a[j] unused: factor 1/2 cancels
                if w == 0:
                    continue
                last = [a[j] for j in range(T) if occ[j]]
                den += w
                num += w * (last[-1] if last else 0)
        p = num / den
        ties += abs(p - 0.5) < 1e-12
        bad += (1 if p > 0.5 else 0) != plugin(obs)
    check(f"D2 unknown positions: MAP = plug-in on all {3 ** T} parsed streams (ties {ties})", bad == 0)


# ---------------- E ----------------
def comp_of(ev):
    k = ev["type"]
    if k in ("GRANT", "GRANT_UNTIL", "WITHDRAW", "RECONFIRM", "ALARM_WITHDRAW"):
        return ("d", ev["p"], ev["r"])
    if k in ("MEMBER", "ADD", "REMOVE"):
        return ("m", ev["p"], ev["g"])
    if k == "GROUP_GRANT":
        return ("gg", ev["g"], ev["r"])
    if k in ("REQ_WINDOW", "OPEN", "CLOSE"):
        return ("w", ev["r"])
    return None


class LogStore(Store):
    """Store that logs every touch (insertion included) and every deletion."""
    def __init__(self, *a):
        super().__init__(*a)
        self.log, self.deletions = [], []

    def step(self, ev):
        c = comp_of(ev)
        was = c in self.rec
        super().step(ev)
        if c is None:
            return
        if c in self.rec and self.touch[c] == ev["t"]:
            self.log.append((ev["t"], c))
        elif was and c not in self.rec:
            self.deletions.append((ev["t"], c))


def case_e():
    rng = random.Random(1)
    # E1: DENY -> APPROVE by eviction
    evs = [d(0, "WITHDRAW", 3)] + [d(1 + j, "GRANT", 2, p="svc_marlin", r=r)
                                     for j, r in enumerate(("db_orders", "bucket_audit"))] + [d(5, "GRANT", 1)]
    s = run_store(evs, 6, 2, "lru", rng)
    check("E1 LRU C=2: evicted top-rank withdrawal, later rank-1 grant applied: store APPROVE, truth DENY",
          answer_from_state(s.state(5), P0, R0) == "APPROVE" and ans(evs, 5) == "DENY")
    evs2 = [d(0, "GRANT", 3)] + [d(1 + j, "GRANT", 2, p="svc_marlin", r=r)
                                  for j, r in enumerate(("db_orders", "bucket_audit"))] + [d(5, "WITHDRAW", 1)]
    s = run_store(evs2, 6, 2, "lru", rng)
    check("E2 and APPROVE -> DENY the same way", answer_from_state(s.state(5), P0, R0) == "DENY" and ans(evs2, 5) == "APPROVE")
    # E3: LRU survival iff fewer than C distinct other components touched after the last touch (no deletions)
    n = bad = skipped = 0
    for K, T in ((7, 64), (15, 256)):
        for ep in cell_episodes(K, T, pairs=20):
            for C in (3, 6, 10):
                st = LogStore(C, "lru", rng)
                for ev in replay_events(ep):
                    st.step(ev)
                last = {}
                for t, c in st.log:
                    last[c] = t
                for x, u in last.items():
                    if any(t > u for t, c in st.deletions):
                        skipped += 1
                        continue
                    others = {c for t, c in st.log if t > u and c != x}
                    n += 1
                    bad += ((len(others) < C) != (x in st.rec))
    check(f"E3 LRU survival rule on {n} (record, run) cases from X4c streams (excluded, a deletion after the last touch: {skipped})", bad == 0)
    # E4: random eviction, fresh insertions -> (1-1/C)^N; insertions with returns -> E_Q[(1-1/C)^N_Q]
    C, N, reps = 4, 6, 40000
    hit = 0
    for _ in range(reps):
        st = Store(C, "random", rng)
        for j in range(C):
            st.step(d(j, "GRANT", 1, p="svc_marlin", r=("db_ledger", "db_orders", "bucket_audit", "queue_billing")[j]))
        st.step(d(C, "GRANT", 2))
        for j in range(N):
            st.step(d(C + 1 + j, "GRANT", 1, p="svc_heron", r=("db_ledger", "db_orders", "bucket_audit", "queue_billing")[j % 4])
                    if j < 4 else d(C + 1 + j, "GRANT", 1, p="svc_otter", r="db_ledger" if j == 4 else "db_orders"))
        hit += ("d", P0, R0) in st.rec
    exp = (1 - 1 / C) ** N
    check(f"E4 fresh insertions: survival {hit / reps:.4f} vs (1-1/C)^N = {exp:.4f}", abs(hit / reps - exp) < 0.01)
    # returns: a cycle over 2C other keys, so evicted keys come back and N is random
    keys = [(p, r) for p in ("svc_marlin", "svc_heron") for r in ("db_ledger", "db_orders", "bucket_audit", "queue_billing")]
    steps = 12

    def run(protect):
        st = Store(C, "random", rng)
        for j in range(C):
            st.step(d(j, "GRANT", 1, p=keys[j][0], r=keys[j][1]))
        x = ("d", P0, R0)
        st._write(x, (True, 2, None, SRC[2]), C)
        if protect:
            st._victim = lambda t: rng.choice(sorted(c for c in st.rec if c != x))
        ev0 = st.evictions
        for j in range(steps):
            p, r = keys[(C + j) % len(keys)]
            st.step(d(C + 1 + j, "GRANT", 1, p=p, r=r))
        return x in st.rec, st.evictions - ev0
    surv = sum(run(False)[0] for _ in range(reps)) / reps
    eq = sum((1 - 1 / C) ** run(True)[1] for _ in range(reps)) / reps
    check(f"E5 insertions with returns: survival {surv:.4f} vs E_Q[(1-1/C)^N] = {eq:.4f}", abs(surv - eq) < 0.01)
    # E6: C >= peak size -> exact
    n = bad = 0
    for K, T in ((3, 64), (7, 256), (15, 256)):
        for ep in cell_episodes(K, T, pairs=20):
            peak = 0
            st = Store(10 ** 9, "lru", rng)
            for ev in replay_events(ep):
                st.step(ev)
                peak = max(peak, len(st.rec))
            for pol in ("random", "lru", "default-first"):
                a, s2 = store_answer(ep, peak, pol, rng)
                n += 1
                bad += a != ep["truth"] or s2.evictions > 0
    check(f"E6 C = peak state size: no eviction and exact answer ({n} runs)", bad == 0)


# ---------------- F ----------------
def flip(ev):
    e = dict(ev)
    e["type"] = "WITHDRAW" if ev["type"] == "GRANT" else "GRANT"
    return e


def parse_error(true_evs, tq, d0, da):
    """Exact error of the exact rule on the parsed stream, enumerating the mutually exclusive outcomes
    drop (d0), flip (da), correct (1 - d0 - da) of every event, independently (Definition def:parser, d_ell = 0)."""
    truth = ans(true_evs, tq)
    err = 0.0
    for outs in itertools.product(range(3), repeat=len(true_evs)):
        w, parsed = 1.0, []
        for ev, o in zip(true_evs, outs):
            if o == 0:
                w *= d0
            elif o == 1:
                w *= da
                parsed.append(flip(ev))
            else:
                w *= 1 - d0 - da
                parsed.append(ev)
        err += w * (ans(parsed, tq) != truth)
    return err


def case_f():
    # F1: masking by a closed required window
    win = [{"t": 6, "type": "REQ_WINDOW", "src": SRC[2], "r": R0}]
    vers = [[d(0, "GRANT" if a else "WITHDRAW", 3), d(3, "WITHDRAW", 1), d(5, "GRANT", 1)] + win for a in (0, 1)]
    always_deny_err = sum(ans(v, 9) != "DENY" for v in vers) / 2
    check("F1 closed required window, no non-masking hypothesis: both versions DENY, constant-DENY reader error "
          f"{always_deny_err:.1f} < 1/2", ans(vers[0], 9) == ans(vers[1], 9) == "DENY" and always_deny_err == 0)
    # F2: non-monotone parser error
    d0, da = 0.1, 0.05
    errs, ok = [], True
    for N in range(4):
        evs = [d(0, "GRANT", 2)] + [d(1 + j, "WITHDRAW", 1) for j in range(N)]
        e = parse_error(evs, N + 1, d0, da)
        formula = da + d0 * (1 - da * (1 - d0 ** N) / (1 - d0))
        ok = ok and abs(e - formula) < 1e-12
        errs.append(e)
    check("F2 parser error, rank-2 grant then N rank-1 withdrawals: Err_N = "
          + ", ".join(f"{e:.4f}" for e in errs) + " (N = 0..3), equals the closed form and decreases",
          ok and all(a > b for a, b in zip(errs, errs[1:]))
          and [round(e, 4) for e in errs[:3]] == [0.15, 0.145, 0.1445])
    # F3: LRU with s after the last touch of x
    rng = random.Random(0)
    x, y, z = ("db_ledger", "db_orders", "bucket_audit")
    st = LogStore(2, "lru", rng)
    for t, r in ((1, x), (2, y), (3, z)):
        st.step(d(t, "GRANT", 1, r=r))
    others_after_2 = {c for t, c in st.log if t > 2 and c != ("d", P0, x)}
    others_after_1 = {c for t, c in st.log if t > 1 and c != ("d", P0, x)}
    check(f"F3 LRU C=2, s=2 is not the last touch of x: x evicted with {len(others_after_2)} other component touched "
          f"in (s,T]; with s=1 (last touch) {len(others_after_1)} = C, as (b) states",
          ("d", P0, x) not in st.rec and len(others_after_2) == 1 and len(others_after_1) == 2 and not st.deletions)


if __name__ == "__main__":
    for f in (case_a, case_b, case_c, case_d, case_e, case_f):
        f()
    print(f"{sum(OK)}/{len(OK)} checks pass")
