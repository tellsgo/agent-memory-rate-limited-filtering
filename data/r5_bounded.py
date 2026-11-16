#!/usr/bin/env python3
"""
R5-1 (PREREG, ACT-1 of reviews/REFEREE_v2.md): an irreversible bounded recursive filter.

The store holds at most C component records. A component is a direct record (p, r), a membership (p, g), a group
grant (g, r) or the window of a resource r. Each event is applied by the exact transition rule against the stored
record; an absent record is treated as no record. Writing a component that is not stored into a full store first
evicts one stored record, which is deleted (there is no archive). A REMOVE deletes its membership and frees the
slot. Any event that addresses a stored component touches it (for LRU), whether or not the rule lets it change the
record. At query time the stored records are resolved for expiry and rendered by render_state.

Policies: random (uniform victim), lru (least recently touched), default_first (evict a record that is equivalent
to no record for every query, i.e. a rank-1 withdrawal or a grant already expired at the current event, before LRU).

  --phase predict   CPU: balanced error of the exact rule on the store contents, per cell, policy and f; writes
                    r5_bounded_pred.json (registered before any reader runs). Also the read-time projection with
                    the same C (R3-3 semantics) for reference.
  --phase read      GPU: reader margins on the rendered store (likelihood scoring, gov prompt), plus the full state
                    on separate calibration seeds for BA(1); writes r5_bounded_<reader>.jsonl.
"""
import argparse, json, math, os, random, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "theory"))
from budget_predict import F_GRID, components  # noqa: E402
from state_machine import RANK, answer_from_state, filter_state, make_pair_v2, render_state, replay_events  # noqa: E402

DIRECT = ("GRANT", "GRANT_UNTIL", "WITHDRAW", "RECONFIRM", "ALARM_WITHDRAW")
CELLS = [(3, 64), (3, 256), (7, 64), (7, 256), (15, 64), (15, 256), (15, 1024)]
POLICIES = ("random", "lru", "default_first")
TEST_BASE, CAL_BASE = 2026220000, 2026230000          # fresh seeds; CAL streams are only used for BA(1)
EMPTY = "State: no access decisions recorded."


def cell_streams(K, T, base, n=40):
    out = []
    for i in range(n):
        out += make_pair_v2(base + T * 10000 + K * 100000 + i, T, bg_keys=K, bg_access_only=True)
    return out


def comp_of(ev):
    k = ev["type"]
    if k in DIRECT:
        return ("d", ev["p"], ev["r"])
    if k in ("MEMBER", "ADD", "REMOVE"):
        return ("m", ev["p"], ev["g"])
    if k == "GROUP_GRANT":
        return ("gg", ev["g"], ev["r"])
    if k in ("REQ_WINDOW", "OPEN", "CLOSE"):
        return ("w", ev["r"])
    return None


def bounded_state(evs, t_query, C, policy, rng):
    """Run the bounded store over events with t < t_query; return a state in filter_state's resolved format."""
    store, touch = {}, {}

    def victim(t):
        if policy == "random":
            return rng.choice(sorted(store))
        if policy == "default_first":
            dead = [c for c in store if c[0] == "d" and ((not store[c][0] and store[c][1] == 1) or
                                                         (store[c][0] and store[c][2] is not None and store[c][2] < t))]
            if dead:
                return min(dead, key=lambda c: (touch[c], c))
        return min(store, key=lambda c: (touch[c], c))

    def put(c, v, t):
        if c not in store:
            if C <= 0:
                return
            if len(store) >= C:
                x = victim(t)
                del store[x], touch[x]
        store[c] = v
        touch[c] = t

    for ev in evs:
        t = ev["t"]
        if t >= t_query:
            break
        k, c = ev["type"], comp_of(ev)
        if c is None:
            continue
        if k in DIRECT:
            cur = store.get(c)
            if cur is not None and cur[2] is not None and cur[2] < t:
                cur = None
            if cur is None or RANK[ev["src"]] >= cur[1]:
                put(c, (k in ("GRANT", "GRANT_UNTIL", "RECONFIRM"), RANK[ev["src"]], ev.get("until"), ev["src"]), t)
            elif c in store:
                touch[c] = t
        elif k == "GROUP_GRANT" or k in ("MEMBER", "ADD"):
            put(c, True, t)
        elif k == "REMOVE":
            store.pop(c, None)
            touch.pop(c, None)
        else:
            need, op = store.get(c, (False, False))
            put(c, (need or k == "REQ_WINDOW", True if k == "OPEN" else False if k == "CLOSE" else op), t)
    st = {"direct": {}, "members": set(), "group_grants": set(), "need_window": set(), "window_open": set()}
    tq = t_query - 1
    for c, v in store.items():
        if c[0] == "d":
            allowed, rank, until, src = v
            expired = allowed and until is not None and until < tq
            st["direct"][(c[1], c[2])] = {"allowed": allowed and not expired, "src": src, "until": until, "expired": expired}
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


def projected_state(ep, C, policy, rng):
    """R3-3 semantics with a fixed C: exact final state, keep min(C, |S|) components (random or most recent)."""
    st, comps, last = components(ep)
    k = min(C, len(comps))
    if policy == "random":
        keep = set(rng.sample(comps, k))
    else:
        keep = set(sorted(comps, key=lambda c: (-last[c], comps.index(c)))[:k])
    return {"direct": {kk: v for kk, v in st["direct"].items() if ("d",) + kk in keep},
            "members": {pg for pg in st["members"] if ("m",) + pg in keep},
            "group_grants": {gr for gr in st["group_grants"] if ("gg",) + gr in keep},
            "need_window": {r for r in st["need_window"] if ("w", r) in keep},
            "window_open": {r for r in st["window_open"] if ("w", r) in keep}}


def capacities(eps):
    sizes = [len(components(ep)[1]) for ep in eps]
    m = sum(sizes) / len(sizes)
    return {f: int(math.floor(f * m + 0.5)) for f in F_GRID}, m


def bal_err(errs):
    return 0.5 * (sum(errs["APPROVE"]) / len(errs["APPROVE"]) + sum(errs["DENY"]) / len(errs["DENY"]))


def self_test():
    """With unbounded capacity the store equals the exact filter state."""
    for K, T in CELLS[:6]:
        for ep in cell_streams(K, T, TEST_BASE, n=10):
            evs = replay_events(ep)
            a = bounded_state(evs, T, 10 ** 9, "lru", random.Random(0))
            b = filter_state(evs, T - 1)
            assert a["direct"] == {k: {kk: v[kk] for kk in ("allowed", "src", "until", "expired")} for k, v in b["direct"].items()}, (K, T)
            for key in ("members", "group_grants", "need_window"):
                assert a[key] == b[key], (K, T, key)
            assert answer_from_state(a, ep["p"], ep["r"]) == ep["truth"]
    print("self-test ok: unbounded store == filter_state on 60 episodes")


def predict(reps=50):
    self_test()
    out = {"F_GRID": F_GRID, "cells": {}}
    for K, T in CELLS:
        eps = cell_streams(K, T, TEST_BASE)
        caps, mean_size = capacities(eps)
        evs = {(ep["pair"], ep["truth"]): replay_events(ep) for ep in eps}
        cell = {"mean_state_size": mean_size, "C": caps, "rec": {}, "proj": {}}
        for pol in POLICIES:
            rec, proj = [], []
            for f in F_GRID:
                C = caps[f]
                e_rec, e_proj = {"APPROVE": [], "DENY": []}, {"APPROVE": [], "DENY": []}
                for ep in eps:
                    n = reps if pol == "random" else 1
                    rng = random.Random(f"{ep['pair']}-{ep['truth']}-{f}-{pol}")
                    q = (ep["p"], ep["r"])
                    e_rec[ep["truth"]].append(sum(answer_from_state(bounded_state(evs[(ep["pair"], ep["truth"])], T, C, pol, rng), *q)
                                                  != ep["truth"] for _ in range(n)) / n)
                    if pol != "default_first":
                        e_proj[ep["truth"]].append(sum(answer_from_state(projected_state(ep, C, pol, rng), *q)
                                                       != ep["truth"] for _ in range(n)) / n)
                rec.append(1 - 2 * bal_err(e_rec))
                if pol != "default_first":
                    proj.append(1 - 2 * bal_err(e_proj))
            cell["rec"][pol] = rec
            if proj:
                cell["proj"][pol] = proj
            print(f"K={K:2d} T={T:4d} |S|={mean_size:5.1f} {pol:13s} rec  " + " ".join(f"{x:.3f}" for x in rec), flush=True)
            if proj:
                print(f"{'':27s}{pol:13s} proj " + " ".join(f"{x:.3f}" for x in proj), flush=True)
        out["cells"][f"{K},{T}"] = cell
    json.dump(out, open(os.path.join(HERE, "r5_bounded_pred.json"), "w"), indent=1)


def items():
    its = []
    for K, T in CELLS:
        eps = cell_streams(K, T, TEST_BASE)
        caps, _ = capacities(eps)
        for ep in eps:
            evs = replay_events(ep)
            for pol in POLICIES:
                for f in F_GRID:
                    rng = random.Random(f"{ep['pair']}-{ep['truth']}-{f}-{pol}-read")
                    st = bounded_state(evs, T, caps[f], pol, rng)
                    lines = render_state(st)
                    its.append({"set": "test", "K": K, "T": T, "f": f, "C": caps[f], "policy": pol, "pair": ep["pair"],
                                "family": ep["family"], "truth": ep["truth"], "kept": len(st["direct"]) + len(st["members"]) +
                                len(st["group_grants"]) + len(st["need_window"]), "memory": "\n".join(lines), "query": ep["query"]})
        for ep in cell_streams(K, T, CAL_BASE):
            st = filter_state(replay_events(ep), T - 1)
            its.append({"set": "cal", "K": K, "T": T, "f": 1.0, "C": None, "policy": "full", "pair": ep["pair"],
                        "family": ep["family"], "truth": ep["truth"], "kept": None,
                        "memory": "\n".join(render_state(st)), "query": ep["query"]})
    return its


def items_full_background():
    """X6 test streams (full background, T=256) with the Qwen-14B parses, bounded store at C in {8,16,24,32}."""
    from x6_parse_filter import content, streams_for, to_event
    P = {r["content"]: tuple(r["parsed"]) for r in map(json.loads, open(os.path.join(HERE, "r3_parse_qwen2.5-14b.jsonl")))}
    test, _ = streams_for("full")
    its = []
    for ep in test:
        evs = [to_event(P[content(l)], i) for i, l in enumerate(ep["lines"])]
        for pol in ("random", "lru"):
            for C in (8, 16, 24, 32):
                rng = random.Random(f"{ep['pair']}-{ep['truth']}-{C}-{pol}-x6")
                st = bounded_state(evs, 256, C, pol, rng)
                its.append({"set": "x6", "K": None, "T": 256, "f": None, "C": C, "policy": pol, "pair": ep["pair"],
                            "family": ep["family"], "truth": ep["truth"], "memory": "\n".join(render_state(st)), "query": ep["query"]})
    return its


def read(reader):
    from x3_recursive_memory import load_reader, margins_batch, ntok, reader_prompt
    tok, model, _ = load_reader(reader, "cuda:0")
    w_head = model.lm_head.weight.detach().float()
    label_ids = {k: tok.encode(v, add_special_tokens=False) for k, v in {"APPROVE": " APPROVE", "DENY": " DENY"}.items()}
    its = items() + items_full_background()
    out_path = os.path.join(HERE, f"r5_bounded_{reader}.jsonl")
    done = sum(1 for _ in open(out_path)) if os.path.exists(out_path) else 0
    for a in range(done, len(its), 2000):
        chunk = its[a:a + 2000]
        ms = margins_batch(tok, model, w_head, label_ids, [reader_prompt(tok, "gov", it["memory"], it["query"]) for it in chunk], "cuda:0", bs=32)
        with open(out_path, "a") as f:
            for it, m in zip(chunk, ms):
                f.write(json.dumps({k: v for k, v in it.items() if k not in ("memory", "query")} |
                                   {"margin": m, "decision": "DENY" if m > 0 else "APPROVE", "memory_tokens": ntok(tok, it["memory"])}) + "\n")
        print(f"{reader}: {a + len(chunk)}/{len(its)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["predict", "read", "count"], required=True)
    ap.add_argument("--reader", default="")
    a = ap.parse_args()
    if a.phase == "predict":
        predict()
    elif a.phase == "count":
        print(len(items()), len(items_full_background()))
    else:
        read(a.reader)
