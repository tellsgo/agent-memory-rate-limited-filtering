#!/usr/bin/env python3
"""
Executable state-machine episodes for access decisions.

v1 (used by E1) covers GRANT / REVOKE / RESTORE plus filler lines. Ground truth is computed
by replaying the events, never by keywords. Generator constraints from E0:
  * no capitalised banners or "CRITICAL" markers; all event types share one register
  * line lengths are kept close (templates of similar length, checked by tests)
  * key events are spread over the timeline, not parked near the end

E2 extends this file with EXPIRE, SCOPE, SOURCE priority, DECOY, IMPLICIT and CONJ events and
with order-swapped pairs.
"""

import random

PRINCIPALS = ["svc_kestrel", "svc_marlin", "svc_heron", "svc_otter", "svc_lynx", "svc_wren"]
RESOURCES = ["db_ledger", "db_orders", "bucket_audit", "queue_billing"]
SOURCES = ["policy_engine", "access_admin", "ops_console"]

GRANT_T = [
    "{src} recorded that {p} may now write to {r} under ticket {tk}.",
    "{src} approved write access for {p} on {r}; reference {tk}.",
    "Per {tk}, {src} added {p} to the writer list of {r}.",
]
REVOKE_T = [
    "{src} recorded that {p} may no longer write to {r} under ticket {tk}.",
    "{src} withdrew write access for {p} on {r}; reference {tk}.",
    "Per {tk}, {src} removed {p} from the writer list of {r}.",
]
RESTORE_T = [
    "{src} recorded that write access for {p} on {r} is back in force, ticket {tk}.",
    "{src} reinstated the earlier write access of {p} on {r}; reference {tk}.",
    "Per {tk}, {src} put {p} back on the writer list of {r}.",
]
FILLER_T = [
    "{src} reported that node {n} finished its scheduled health check in {ms} ms.",
    "{src} rotated the log archive for shard {n}; {ms} entries were compacted.",
    "{src} noted that replica {n} caught up with the primary after {ms} ms.",
    "{src} measured queue depth {ms} on worker {n}; no action was needed.",
]

QUERY_T = (
    "{p} requests to write a batch of records to {r}.\n"
    "Should the governance agent APPROVE or DENY the request? Answer format: start with "
    "'DECISION: APPROVE' or 'DECISION: DENY'."
)


def _ticket(rng):
    return f"T-{rng.randint(1000, 9999)}"


def make_episode(seed, n_events=24, n_pairs=4):
    """One episode: event lines, final state per (principal, resource) pair, and queries.

    Each active pair gets a random chain of GRANT / REVOKE / RESTORE events; remaining slots
    are fillers. Queries ask about active pairs and one never-mentioned pair (denied by default).
    """
    rng = random.Random(seed)
    pairs = rng.sample([(p, r) for p in PRINCIPALS for r in RESOURCES], n_pairs + 1)
    active, unseen = pairs[:n_pairs], pairs[n_pairs]

    # Final state is drawn first (50/50), then a chain length with that parity:
    # allowed -> GRANT or GRANT-REVOKE-RESTORE; denied -> GRANT-REVOKE or GRANT-REVOKE-RESTORE-REVOKE.
    chains = []
    for pr in active:
        allowed = rng.random() < 0.5
        k = (1 if rng.random() < 0.4 else 3) if allowed else rng.choice([2, 4])
        chain = ["GRANT"]
        for _ in range(k - 1):
            chain.append("REVOKE" if chain[-1] in ("GRANT", "RESTORE") else "RESTORE")
        chains.append((pr, chain))
    n_key = sum(len(c) for _, c in chains)
    assert n_key <= n_events, "too many key events for the horizon"

    # Spread key events uniformly over the timeline while keeping each chain in order.
    slots = sorted(rng.sample(range(n_events), n_key))
    tagged = [(pr, j, ev) for pr, chain in chains for j, ev in enumerate(chain)]
    rng.shuffle(tagged)
    tagged.sort(key=lambda x: x[1])  # earlier chain positions first, pairs interleaved
    placement = {}
    per_pair_next = {pr: 0 for pr in active}
    pending = list(tagged)
    for s in slots:
        for idx, (pr, j, ev) in enumerate(pending):
            if j == per_pair_next[pr]:
                placement[s] = (pr, ev)
                per_pair_next[pr] += 1
                pending.pop(idx)
                break
    assert not pending

    lines, events, state = [], [], {}
    for t in range(n_events):
        src = rng.choice(SOURCES)
        if t in placement:
            (p, r), ev = placement[t]
            tmpl = {"GRANT": GRANT_T, "REVOKE": REVOKE_T, "RESTORE": RESTORE_T}[ev]
            text = rng.choice(tmpl).format(src=src, p=p, r=r, tk=_ticket(rng))
            state[(p, r)] = ev in ("GRANT", "RESTORE")
            events.append({"t": t, "type": ev, "p": p, "r": r})
        else:
            text = rng.choice(FILLER_T).format(src=src, n=rng.randint(2, 64), ms=rng.randint(100, 999))
            events.append({"t": t, "type": "FILLER"})
        lines.append(f"Event {t + 1:02d}: {text}")

    queries = []
    for (p, r) in active + [unseen]:
        allowed = state.get((p, r), False)
        last = max((e["t"] for e in events if e.get("p") == p and e.get("r") == r), default=None)
        queries.append({
            "p": p, "r": r, "text": QUERY_T.format(p=p, r=r),
            "truth": "APPROVE" if allowed else "DENY", "last_event_t": last,
        })
    return {"seed": seed, "lines": lines, "events": events, "queries": queries}


def balanced_queries(ep, k, rng):
    """Pick k queries with APPROVE/DENY as balanced as the episode allows."""
    pos = [q for q in ep["queries"] if q["truth"] == "APPROVE"]
    neg = [q for q in ep["queries"] if q["truth"] == "DENY"]
    rng.shuffle(pos)
    rng.shuffle(neg)
    n_pos = min(len(pos), max(k // 2, k - len(neg)))
    return pos[:n_pos] + neg[:k - n_pos]


if __name__ == "__main__":
    ep = make_episode(0)
    print("\n".join(ep["lines"]))
    for q in ep["queries"]:
        print(q["p"], q["r"], q["truth"], q["last_event_t"])


# =============================================================================================
# v2 generator (E2-0 onward): seven families, executable replay, order-swapped pairs.
#
# Every episode is rendered from structured events; the answer comes from replay(), and each
# family also states the label it intends, which is asserted against the replay. Pairs share
# background, slots and wording; only the target family's lines differ (order or one field).
# =============================================================================================

RANK = {"security_office": 3, "access_admin": 2, "ops_console": 1}
SOURCES_V2 = list(RANK)
GROUPS = ["grp_alpha", "grp_beta", "grp_gamma", "grp_delta"]

POLICY_V2 = (
    "Access policy:\n"
    "1. Sources are ranked: security_office above access_admin above ops_console. An access event "
    "(grant, withdrawal or reconfirmation) changes the current decision only if its source ranks at "
    "least as high as the source of the decision it would replace.\n"
    "2. A grant stated as valid through Event N no longer applies after Event N.\n"
    "3. A service may write to a resource if it holds a current grant, or if it belongs to a group "
    "that holds a grant on that resource.\n"
    "4. If a resource accepts writes only while its change window is open, a write needs both "
    "permission and an open window. Windows start closed.\n"
    "5. Without an applicable grant the answer is DENY."
)
OVERRIDE_RULE_V2 = (
    "\n6. Events are listed in time order. A later applicable event replaces an earlier one: a "
    "withdrawal is undone by a later grant or reconfirmation, and a grant is undone by a later withdrawal."
)

T2 = {
    "GRANT": ["{src} granted {p} write access to {r} under ticket {tk}.",
              "{src} gave {p} permission to write to {r}, reference {tk}."],
    "GRANT_UNTIL": ["{src} granted {p} write access to {r} valid through Event {n:03d}, ticket {tk}.",
                    "{src} gave {p} permission to write to {r} until Event {n:03d}, reference {tk}."],
    "WITHDRAW": ["{src} withdrew the write access of {p} on {r} under ticket {tk}.",
                 "{src} took away the permission of {p} to write to {r}, reference {tk}."],
    "RECONFIRM": ["{src} reconfirmed that {p} holds write access to {r}, ticket {tk}.",
                  "{src} confirmed again that {p} may write to {r}, reference {tk}."],
    "ALARM_WITHDRAW": ["{src} flagged leaked credentials for {p} and withdrew its write access on {r}.",
                       "{src} found a credential leak at {p} and took away its write access on {r}."],
    "GROUP_GRANT": ["{src} granted group {g} write access to {r} under ticket {tk}.",
                    "{src} gave group {g} permission to write to {r}, reference {tk}."],
    "MEMBER": ["{src} lists {p} as a member of group {g}, ticket {tk}.",
               "{src} records {p} among the members of group {g}, reference {tk}."],
    "ADD": ["{src} added {p} to group {g} under ticket {tk}.",
            "{src} put {p} into group {g}, reference {tk}."],
    "REMOVE": ["{src} removed {p} from group {g} under ticket {tk}.",
               "{src} took {p} out of group {g}, reference {tk}."],
    "REQ_WINDOW": ["{src} set {r} to accept writes only while its change window is open.",
                   "{src} configured {r} so that writes need an open change window."],
    "OPEN": ["{src} opened the change window of {r} under ticket {tk}.",
             "{src} started a change window on {r}, reference {tk}."],
    "CLOSE": ["{src} closed the change window of {r} under ticket {tk}.",
              "{src} ended the change window on {r}, reference {tk}."],
}

QUERY_V2 = (
    "Request received after Event {t:03d}: {p} asks to write a batch of records to {r}.\n"
    "Should the governance agent APPROVE or DENY the request? Answer format: start with "
    "'DECISION: APPROVE' or 'DECISION: DENY'."
)

FAMILIES = {          # name: (sampling weight, description)
    "restore": (2, "grant, withdraw, reconfirm; order of the last two swapped"),
    "order": (1, "grant and withdraw only; order swapped"),
    "expiry": (1, "grant valid through Event N, N before or after the request"),
    "priority": (1, "grant, withdraw, re-grant; sources of the last two swapped"),
    "decoy": (1, "grant plus an alarm-worded withdrawal for another or the same principal"),
    "implicit": (1, "group grant and membership; remove/add order swapped, no alarm words"),
    "conj": (1, "grant plus change-window requirement; open/close order swapped"),
}


def replay(events, p, r, t_query):
    """Executable ground truth for 'may p write to r after event t_query'."""
    direct = {}        # (p, r) -> (allowed, rank, until)
    members, group_grants = set(), set()
    need_window, window_open = set(), set()
    for ev in events:
        k = ev["type"]
        if k in ("GRANT", "GRANT_UNTIL", "WITHDRAW", "RECONFIRM", "ALARM_WITHDRAW"):
            key, rank = (ev["p"], ev["r"]), RANK[ev["src"]]
            cur = direct.get(key)
            if cur is not None and cur[2] is not None and cur[2] < ev["t"]:
                cur = None                                   # expired grants hold no rank
            if cur is None or rank >= cur[1]:
                allowed = k in ("GRANT", "GRANT_UNTIL", "RECONFIRM")
                direct[key] = (allowed, rank, ev.get("until"))
        elif k == "GROUP_GRANT":
            group_grants.add((ev["g"], ev["r"]))
        elif k in ("MEMBER", "ADD"):
            members.add((ev["p"], ev["g"]))
        elif k == "REMOVE":
            members.discard((ev["p"], ev["g"]))
        elif k == "REQ_WINDOW":
            need_window.add(ev["r"])
        elif k == "OPEN":
            window_open.add(ev["r"])
        elif k == "CLOSE":
            window_open.discard(ev["r"])
    d = direct.get((p, r))
    perm = d is not None and d[0] and (d[2] is None or d[2] >= t_query)
    perm = perm or any((p, g) in members and (g, r) in group_grants for g in GROUPS)
    if r in need_window and r not in window_open:
        perm = False
    return "APPROVE" if perm else "DENY"


def _target_events(family, rng, p, r, p_other, g, T, slots):
    """Two versions (A = APPROVE, D = DENY) of the target family's events, as ordered lists of
    structured events without times. Times come from the shared sorted slots. Tags mark subsets."""
    adm, sec, ops = "access_admin", "security_office", "ops_console"
    E = lambda typ, **kw: dict(type=typ, **kw)
    if family == "restore":
        G, W, C = E("GRANT", src=adm, p=p, r=r), E("WITHDRAW", src=adm, p=p, r=r), E("RECONFIRM", src=adm, p=p, r=r)
        return [G, W, C], [G, C, W], {"A": ["restore", "order_pair"], "D": ["order_pair"]}
    if family == "order":
        G, W = E("GRANT", src=adm, p=p, r=r), E("WITHDRAW", src=adm, p=p, r=r)
        return [W, G], [G, W], {"A": ["order_pair"], "D": ["order_pair"]}
    if family == "expiry":
        t_grant = slots[0]
        n_after = T + rng.randint(5, 20)
        n_before = rng.randint(t_grant + 2, T - 1) if t_grant + 2 <= T - 1 else T - 1
        return ([E("GRANT_UNTIL", src=adm, p=p, r=r, until=n_after)],
                [E("GRANT_UNTIL", src=adm, p=p, r=r, until=n_before)],
                {"A": [], "D": ["no_alarm"]})
    if family == "priority":
        G0 = E("GRANT", src=ops, p=p, r=r)
        A = [G0, E("WITHDRAW", src=ops, p=p, r=r), E("GRANT", src=sec, p=p, r=r)]
        D = [G0, E("WITHDRAW", src=sec, p=p, r=r), E("GRANT", src=ops, p=p, r=r)]
        return A, D, {"A": ["restore"], "D": ["latest_cannot_override"]}
    if family == "decoy":
        G = E("GRANT", src=adm, p=p, r=r)
        return ([G, E("ALARM_WITHDRAW", src=sec, p=p_other, r=r)],
                [G, E("ALARM_WITHDRAW", src=sec, p=p, r=r)],
                {"A": ["decoy_other_entity"], "D": []})
    if family == "implicit":
        GG, M = E("GROUP_GRANT", src=adm, g=g, r=r), E("MEMBER", src=adm, p=p, g=g)
        rem, add = E("REMOVE", src=adm, p=p, g=g), E("ADD", src=adm, p=p, g=g)
        return [GG, M, rem, add], [GG, M, add, rem], {"A": ["restore", "order_pair", "no_alarm"], "D": ["order_pair", "no_alarm"]}
    if family == "conj":
        G, Q = E("GRANT", src=adm, p=p, r=r), E("REQ_WINDOW", src=adm, r=r)
        o, c = E("OPEN", src=adm, r=r), E("CLOSE", src=adm, r=r)
        return [G, Q, c, o], [G, Q, o, c], {"A": ["order_pair", "conjunction"], "D": ["order_pair", "conjunction", "no_alarm"]}
    raise ValueError(family)


def _render(ev, rng_text):
    tmpl = rng_text.choice(T2[ev["type"]])
    return tmpl.format(src=ev["src"], p=ev.get("p"), r=ev.get("r"), g=ev.get("g"),
                       n=ev.get("until", 0), tk=ev["tk"])


def make_pair_v2(seed, T, family=None, bg_keys=None, bg_access_only=False):
    """An (APPROVE, DENY) pair of episodes with T events each.

    bg_keys: if given, background access events use only this many (principal, resource) keys
    (besides the target), drawn with a separate generator so the default path is unchanged.
    """
    rng = random.Random(seed)
    if family is None:
        names = list(FAMILIES)
        family = rng.choices(names, weights=[FAMILIES[n][0] for n in names])[0]
    p, p_other = rng.sample(PRINCIPALS, 2)
    r = rng.choice(RESOURCES)
    g = rng.choice(GROUPS)

    n_target = {"restore": 3, "order": 2, "expiry": 1, "priority": 3, "decoy": 2, "implicit": 4, "conj": 4}[family]
    slots = sorted(rng.sample(range(T), n_target))
    A, D, tags = _target_events(family, rng, p, r, p_other, g, T, slots)

    # Background: other pairs with grant/withdraw/reconfirm chains, other groups, other windows.
    # It never touches (p, r), never puts p in a group and never touches r's window.
    others = [(pp, rr) for pp in PRINCIPALS for rr in RESOURCES if (pp, rr) != (p, r)]
    if bg_keys is not None:
        others = random.Random(seed * 7 + 3).sample(others, min(bg_keys, len(others)))
    free = [t for t in range(T) if t not in slots]
    n_bg = min(len(free), max(0, T // 3))
    bg_slots = sorted(rng.sample(free, n_bg))
    bg = []
    other_groups = [x for x in GROUPS if x != g]
    other_res = [x for x in RESOURCES if x != r]
    for _ in bg_slots:
        kind = rng.random()
        src = rng.choice(SOURCES_V2)
        if bg_access_only or kind < 0.7:
            pp, rr = rng.choice(others)
            bg.append(dict(type=rng.choice(["GRANT", "WITHDRAW", "RECONFIRM"]), src=src, p=pp, r=rr))
        elif kind < 0.85:
            pp = rng.choice([x for x in PRINCIPALS if x != p])
            bg.append(dict(type=rng.choice(["ADD", "REMOVE"]), src=src, p=pp, g=rng.choice(other_groups)))
        else:
            bg.append(dict(type=rng.choice(["OPEN", "CLOSE"]), src=src, r=rng.choice(other_res)))

    episodes = []
    for label, target in (("APPROVE", A), ("DENY", D)):
        rng_text = random.Random(seed * 7 + 1)               # same wording draws for both versions
        events = {}
        for t, ev in zip(slots, target):
            events[t] = dict(ev, t=t)
        for t, ev in zip(bg_slots, bg):
            events[t] = dict(ev, t=t)
        lines, evlist = [], []
        tick = random.Random(seed * 7 + 2)
        for t in range(T):
            tk = _ticket(tick)
            if t in events:
                ev = dict(events[t], tk=tk)
                text = _render(ev, rng_text)
                evlist.append(ev)
            else:
                src = rng_text.choice(SOURCES_V2)
                text = rng_text.choice(FILLER_T).format(src=src, n=tick.randint(2, 64), ms=tick.randint(100, 999))
                evlist.append({"t": t, "type": "FILLER"})
            lines.append(f"Event {t + 1:03d}: {text}")
        # Event numbers in text are 1-based; replay uses 0-based t, so 'valid through Event N'
        # (1-based) means until = N - 1 in replay time, and the request comes after Event T.
        for ev in evlist:
            if "until" in ev:
                ev["until_rt"] = ev["until"] - 1
        rp = [dict(ev, until=ev.get("until_rt")) if "until" in ev else ev for ev in evlist]
        truth = replay(rp, p, r, T - 1)
        assert truth == label, f"family {family} seed {seed}: replay {truth} != intended {label}"
        episodes.append({
            "pair": seed, "family": family, "T": T, "truth": truth,
            "tags": tags["A" if label == "APPROVE" else "D"],
            "lines": lines, "events": evlist, "p": p, "r": r,
            "query": QUERY_V2.format(t=T, p=p, r=r),
            "target_slots": slots,
        })
    return episodes


# =============================================================================================
# Filter state (the sufficient statistic) and its rendering, for state-memory experiments.
# filter_state() runs the same semantics as replay(); answer_from_state() must agree with replay()
# on every query (checked by _self_test_filter_state).
# =============================================================================================

def filter_state(events, t_query):
    direct, members, group_grants, need_window, window_open = {}, set(), set(), set(), set()
    for ev in events:
        k = ev["type"]
        if k in ("GRANT", "GRANT_UNTIL", "WITHDRAW", "RECONFIRM", "ALARM_WITHDRAW"):
            key, rank = (ev["p"], ev["r"]), RANK[ev["src"]]
            cur = direct.get(key)
            if cur is not None and cur[2] is not None and cur[2] < ev["t"]:
                cur = None
            if cur is None or rank >= cur[1]:
                allowed = k in ("GRANT", "GRANT_UNTIL", "RECONFIRM")
                direct[key] = (allowed, rank, ev.get("until"), ev["src"])
        elif k == "GROUP_GRANT":
            group_grants.add((ev["g"], ev["r"]))
        elif k in ("MEMBER", "ADD"):
            members.add((ev["p"], ev["g"]))
        elif k == "REMOVE":
            members.discard((ev["p"], ev["g"]))
        elif k == "REQ_WINDOW":
            need_window.add(ev["r"])
        elif k == "OPEN":
            window_open.add(ev["r"])
        elif k == "CLOSE":
            window_open.discard(ev["r"])
    # Resolve expiry at query time: an expired grant is a current "not allowed".
    resolved = {}
    for key, (allowed, rank, until, src) in direct.items():
        expired = allowed and until is not None and until < t_query
        resolved[key] = {"allowed": allowed and not expired, "src": src, "until": until, "expired": expired}
    return {"direct": resolved, "members": members, "group_grants": group_grants,
            "need_window": need_window, "window_open": window_open}


def answer_from_state(st, p, r):
    d = st["direct"].get((p, r))
    perm = d is not None and d["allowed"]
    perm = perm or any((p, g) in st["members"] and (g, r) in st["group_grants"] for g in GROUPS)
    if r in st["need_window"] and r not in st["window_open"]:
        perm = False
    return "APPROVE" if perm else "DENY"


def render_state(st):
    """The filter state as memory lines. Lists facts, never the answer to any query."""
    lines = []
    for (p, r) in sorted(st["direct"]):
        d = st["direct"][(p, r)]
        if d["expired"]:
            what = f"write access expired after Event {d['until'] + 1:03d}"
        elif d["allowed"]:
            what = "write access granted" + (f", valid through Event {d['until'] + 1:03d}" if d["until"] is not None else "")
        else:
            what = "write access withdrawn"
        lines.append(f"State: {p} on {r}: {what} (decided by {d['src']}).")
    for (p, g) in sorted(st["members"]):
        lines.append(f"State: {p} is a member of group {g}.")
    for (g, r) in sorted(st["group_grants"]):
        lines.append(f"State: group {g} holds write access to {r}.")
    for r in sorted(st["need_window"]):
        lines.append(f"State: {r} accepts writes only while its change window is open; the window is {'open' if r in st['window_open'] else 'closed'}.")
    return lines or ["State: no access decisions recorded."]


def replay_events(ep):
    """Episode events in replay time (0-based 'until')."""
    return [dict(ev, until=ev.get("until_rt")) if "until" in ev else ev for ev in ep["events"]]


def _self_test_filter_state(n=300):
    for T in (8, 16, 64, 256):
        for s in range(n):
            for ep in make_pair_v2(77000 + s, T):
                evs = replay_events(ep)
                st = filter_state(evs, T - 1)
                for p in PRINCIPALS:
                    for r in RESOURCES:
                        assert answer_from_state(st, p, r) == replay(evs, p, r, T - 1), (T, s, p, r)
    return True


# =============================================================================================
# Chain counterfactuals (theory def:chain): one key receives m direct events of one rank and no
# lease, decisions 1,0,1,...,0,1 in version A; version D swaps the CONTENTS at the last two key
# slots (wording and ticket move with the content), so the multiset of contents is identical.
# Background: n_other access events on other keys (never the target key, never p's membership,
# never r's group grants or window) and fillers up to T.
# =============================================================================================

def make_chain_pair(seed, m, T, n_other=None):
    assert m % 2 == 1 and m >= 3
    rng = random.Random(seed)
    p = rng.choice(PRINCIPALS)
    r = rng.choice(RESOURCES)
    if n_other is None:
        n_other = T // 3
    assert m + n_other <= T, "chain and background do not fit the horizon"
    slots = sorted(rng.sample(range(T), m))
    free = [t for t in range(T) if t not in slots]
    bg_slots = sorted(rng.sample(free, n_other))
    others = [(pp, rr) for pp in PRINCIPALS for rr in RESOURCES if (pp, rr) != (p, r)]
    src = "access_admin"

    contents, events = {}, {}
    kinds = ["GRANT"] + ["WITHDRAW" if j % 2 == 1 else "RECONFIRM" for j in range(1, m)]
    for t, k in zip(slots, kinds):
        ev = dict(type=k, src=src, p=p, r=r, tk=_ticket(rng))
        contents[t] = rng.choice(T2[k]).format(src=src, p=p, r=r, g=None, n=0, tk=ev["tk"])
        events[t] = ev
    for t in bg_slots:
        pp, rr = rng.choice(others)
        k = rng.choice(["GRANT", "WITHDRAW", "RECONFIRM"])
        s = rng.choice(SOURCES_V2)
        ev = dict(type=k, src=s, p=pp, r=rr, tk=_ticket(rng))
        contents[t] = rng.choice(T2[k]).format(src=s, p=pp, r=rr, g=None, n=0, tk=ev["tk"])
        events[t] = ev
    for t in range(T):
        if t not in contents:
            s = rng.choice(SOURCES_V2)
            contents[t] = rng.choice(FILLER_T).format(src=s, n=rng.randint(2, 64), ms=rng.randint(100, 999))
            events[t] = {"type": "FILLER"}

    out = []
    for label in ("APPROVE", "DENY"):
        c, e = dict(contents), {t: dict(v) for t, v in events.items()}
        if label == "DENY":
            a, b = slots[-2], slots[-1]
            c[a], c[b] = c[b], c[a]
            e[a], e[b] = e[b], e[a]
        evlist = [dict(e[t], t=t) for t in range(T)]
        lines = [f"Event {t + 1:03d}: {c[t]}" for t in range(T)]
        truth = replay(evlist, p, r, T - 1)
        assert truth == label, (seed, m, T, truth, label)
        out.append({"pair": seed, "family": f"chain{m}", "T": T, "truth": truth, "tags": ["chain"],
                    "lines": lines, "contents": [c[t] for t in range(T)], "events": evlist,
                    "p": p, "r": r, "query": QUERY_V2.format(t=T, p=p, r=r),
                    "target_slots": slots, "m": m, "n_other": n_other})
    return out


def make_blocked_pair(seed, b_prime, T, n_other=None):
    """prop:recency: a top-rank decision a on the key, then b_prime lower-rank events on the same
    key that are not applied and are identical in both versions. Version A: a = grant; D: withdraw."""
    rng = random.Random(seed)
    p = rng.choice(PRINCIPALS)
    r = rng.choice(RESOURCES)
    if n_other is None:
        n_other = T // 3
    assert 1 + b_prime + n_other <= T
    slots = sorted(rng.sample(range(T), 1 + b_prime))
    free = [t for t in range(T) if t not in slots]
    bg_slots = sorted(rng.sample(free, n_other))
    others = [(pp, rr) for pp in PRINCIPALS for rr in RESOURCES if (pp, rr) != (p, r)]
    contents, events = {}, {}
    tk0 = _ticket(rng)
    w0 = rng.randrange(2)
    for t in slots[1:]:
        k = rng.choice(["GRANT", "WITHDRAW", "RECONFIRM"])
        ev = dict(type=k, src="ops_console", p=p, r=r, tk=_ticket(rng))
        contents[t] = rng.choice(T2[k]).format(src="ops_console", p=p, r=r, g=None, n=0, tk=ev["tk"])
        events[t] = ev
    for t in bg_slots:
        pp, rr = rng.choice(others)
        k = rng.choice(["GRANT", "WITHDRAW", "RECONFIRM"])
        s = rng.choice(SOURCES_V2)
        ev = dict(type=k, src=s, p=pp, r=rr, tk=_ticket(rng))
        contents[t] = rng.choice(T2[k]).format(src=s, p=pp, r=rr, g=None, n=0, tk=ev["tk"])
        events[t] = ev
    for t in range(T):
        if t not in contents and t != slots[0]:
            s = rng.choice(SOURCES_V2)
            contents[t] = rng.choice(FILLER_T).format(src=s, n=rng.randint(2, 64), ms=rng.randint(100, 999))
            events[t] = {"type": "FILLER"}
    out = []
    for label, k in (("APPROVE", "GRANT"), ("DENY", "WITHDRAW")):
        c, e = dict(contents), {t: dict(v) for t, v in events.items()}
        e[slots[0]] = dict(type=k, src="security_office", p=p, r=r, tk=tk0)
        c[slots[0]] = T2[k][w0].format(src="security_office", p=p, r=r, g=None, n=0, tk=tk0)
        evlist = [dict(e[t], t=t) for t in range(T)]
        lines = [f"Event {t + 1:03d}: {c[t]}" for t in range(T)]
        truth = replay(evlist, p, r, T - 1)
        assert truth == label, (seed, b_prime, truth, label)
        out.append({"pair": seed, "family": f"blocked{b_prime}", "T": T, "truth": truth, "tags": ["blocked"],
                    "lines": lines, "contents": [c[t] for t in range(T)], "events": evlist,
                    "p": p, "r": r, "query": QUERY_V2.format(t=T, p=p, r=r),
                    "target_slots": slots, "b_prime": b_prime, "n_other": n_other})
    return out


def last_applied_indices(rev):
    """prop:aware: indices of the last APPLIED event of each state component (direct events are
    rank and lease aware; memberships, windows: last event; group grants and window requirements:
    first occurrence)."""
    cur, last = {}, {}
    for e in rev:
        k = e["type"]
        if k in ("GRANT", "GRANT_UNTIL", "WITHDRAW", "RECONFIRM", "ALARM_WITHDRAW"):
            key, rank = (e["p"], e["r"]), RANK[e["src"]]
            c = cur.get(key)
            if c is not None and c[1] is not None and c[1] < e["t"]:
                c = None
            if c is None or rank >= c[0]:
                cur[key] = (rank, e.get("until"))
                last[("d",) + key] = e["t"]
        elif k in ("MEMBER", "ADD", "REMOVE"):
            last[("m", e["p"], e["g"])] = e["t"]
        elif k in ("GROUP_GRANT", "REQ_WINDOW"):
            last.setdefault(("c", k, e.get("g"), e["r"]), e["t"])
        elif k in ("OPEN", "CLOSE"):
            last[("w", e["r"])] = e["t"]
    return sorted(last.values())
