#!/usr/bin/env python3
"""Bayes (forward-recursion) filter against the plug-in filter behind a noisy one-line parser
(theory.tex, sec:theory-parser, prop:bayes-plugin; registration theory/REGISTRATION_round4.md).

Model. Each log line is parsed independently. For a state component c (direct record of a key, membership,
group grant, window of a resource) the line's effect on c is unobserved; its posterior given the parsed tuple is
P(effect on c | relation of the parse to c), where the relation records the parsed type, where c's identifiers
appear in the parsed fields (own slot, another slot, absent), and for direct records the parsed source and
whether a lease was parsed. These posteriors are tied across components of the same kind (the generator is
symmetric in names) and fitted by counting over the CALIBRATION streams only, with additive smoothing.
The forward recursion runs the exact transition rule of replay() over a distribution on each relevant
component's state. The decision is MAP: APPROVE iff P(answer = APPROVE) > 1/2, computed from the product of the
component marginals (exact for each component; the product ignores the weak coupling that one line touches at
most one component).

Plug-in. The exact rule applied to the parsed events (to_event), as in the registered H-P analysis.

Modes:
  python3 bayes_filter.py cv          calibration only: two-fold cross-validation over calibration streams
                                      (fit on one half, score Bayes and plug-in on the other), plus the
                                      line-independence statistics on calibration lines
  python3 bayes_filter.py test        fit on all calibration streams, score on the X6 test streams
                                      (run only after REGISTRATION_round4.md is registered)
"""
import json
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(HERE, "..", "experiments")
sys.path.insert(0, EXP)
from state_machine import GROUPS, PRINCIPALS, RANK, RESOURCES, make_pair_v2, replay, replay_events  # noqa: E402
# Verbatim copies of the pure helpers of experiments/x6_parse_filter.py (that module imports torch).
TYPE_MAP = {"GRANT_UNTIL": "GRANT", "ALARM_WITHDRAW": "WITHDRAW", "MEMBER": "ADD", "FILLER": "NONE"}


def content(line):
    return line.split(": ", 1)[1]


def truth_tuple(ev):
    k = TYPE_MAP.get(ev["type"], ev["type"])
    if k == "NONE":
        return ("NONE",)
    until = ev.get("until")
    return (k, ev.get("src"), ev.get("p"), ev.get("r"), ev.get("g"), None if until is None else until + 1)


def to_event(tup, t):
    """Parsed tuple -> replay event (None if not applicable)."""
    if tup[0] == "NONE":
        return {"t": t, "type": "FILLER"}
    k, src, p, r, g, vt = tup
    need = {"GRANT": (p, r), "WITHDRAW": (p, r), "RECONFIRM": (p, r), "GROUP_GRANT": (g, r), "ADD": (p, g),
            "REMOVE": (p, g), "REQ_WINDOW": (r,), "OPEN": (r,), "CLOSE": (r,)}[k]
    if any(x is None for x in need) or (k in ("GRANT", "WITHDRAW", "RECONFIRM") and src not in RANK):
        return {"t": t, "type": "FILLER"}                # an unusable parse changes no state
    ev = {"t": t, "type": k, "src": src if src in RANK else "ops_console", "p": p, "r": r, "g": g}
    if k == "GRANT" and vt is not None:
        ev["type"] = "GRANT_UNTIL"
        ev["until"] = vt - 1                     # 1-based "valid through Event N" -> replay time
    return ev


def streams_for(design):
    assert design == "full"
    out = []
    for i in range(40):
        out += make_pair_v2(2026130000 + 256 * 10000 + i, 256)
    return out, [32, 64, 128, 256]
from state_machine import answer_from_state, filter_state  # noqa: E402

PARSERS = ["qwen2.5-0.5b", "qwen2.5-1.5b", "qwen2.5-3b", "qwen2.5-7b", "qwen2.5-14b"]
CAL_SEED = 2026170000
CHECKPOINTS = (64, 256)
ALPHA = 0.5
BETA = 1.0
DIRECT = ("GRANT", "WITHDRAW", "RECONFIRM")
SLOTS = {"D": (("p", 2), ("r", 3)), "M": (("p", 2), ("g", 4)), "G": (("g", 4), ("r", 3)), "W": (("r", 3),)}
UNIVERSE = {"D": [{"p": p, "r": r} for p in PRINCIPALS for r in RESOURCES],
            "M": [{"p": p, "g": g} for p in PRINCIPALS for g in GROUPS],
            "G": [{"g": g, "r": r} for g in GROUPS for r in RESOURCES],
            "W": [{"r": r} for r in RESOURCES]}


def calib_streams():
    out = []
    for i in range(40):
        out += make_pair_v2(CAL_SEED + 256 * 10000 + i, 256)
    return out


def load_parses(parser):
    return {json.loads(l)["content"]: json.loads(l) for l in open(os.path.join(EXP, f"r3_parse_{parser}.jsonl"))}


# ---------------- features and labels ----------------

def feature(kind, cid, pt):
    if pt[0] == "NONE":
        return (kind, "NONE")
    fields = (pt[2], pt[3], pt[4])
    sig = []
    for name, idx in SLOTS[kind]:
        x = cid[name]
        sig.append("own" if pt[idx] == x else ("other" if x in fields else "none"))
    sig = tuple(sig)
    if all(s == "none" for s in sig):
        return (kind, "UNREL")
    if kind == "D":
        lease = "none" if pt[5] is None else ("idx" if pt[5] < 1000 else "big")   # an event index vs a ticket-sized number
        return (kind, pt[0], sig, pt[1] if pt[1] in RANK else "bad", lease)
    return (kind, pt[0], sig)


def label(kind, cid, tt):
    if tt[0] == "NONE":
        return "none"
    k, src, p, r, g, vt = tt
    if kind == "D" and k in DIRECT and (p, r) == (cid["p"], cid["r"]):
        return ("pos" if k in ("GRANT", "RECONFIRM") else "neg", src, vt is not None)
    if kind == "M" and k in ("ADD", "REMOVE") and (p, g) == (cid["p"], cid["g"]):
        return k
    if kind == "G" and k == "GROUP_GRANT" and (g, r) == (cid["g"], cid["r"]):
        return "GG"
    if kind == "W" and k in ("REQ_WINDOW", "OPEN", "CLOSE") and r == cid["r"]:
        return k
    return "none"


def summary(kind, z):
    """Coarse summary of a component state, on which the event prior is conditioned (HMM prior)."""
    if kind == "D":
        return "none" if z is None else ("allow" if z[0] else "deny")
    if kind == "W":
        return z[0]
    return z


def true_step(kind, z, lab, t, vt):
    """Exact update of one component state by a true label (calibration side)."""
    if lab == "none":
        return z
    if kind == "D":
        return max(step_direct({z: 1.0}, lab, t, vt).items(), key=lambda kv: kv[1])[0]
    if kind == "M":
        return 1 if lab == "ADD" else 0
    if kind == "G":
        return 1
    return (1, z[1]) if lab == "REQ_WINDOW" else (z[0], 1 if lab == "OPEN" else 0)


def init(kind):
    return None if kind == "D" else (0 if kind in "MG" else (0, 0))


def fit(streams, P):
    """counts[(feature, state summary)][label] over every line occurrence of the given streams and every
    component; the summary is that of the true component state just before the line."""
    counts = defaultdict(Counter)
    labels = defaultdict(set)
    for s in streams:
        z = {(kind, tuple(cid.values())): init(kind) for kind, uni in UNIVERSE.items() for cid in uni}
        for i, (line, ev) in enumerate(zip(s["lines"], replay_events(s))):
            tt = truth_tuple(ev)
            pt = tuple(P[content(line)]["parsed"])
            for kind, uni in UNIVERSE.items():
                for cid in uni:
                    key = (kind, tuple(cid.values()))
                    lab = label(kind, cid, tt)
                    counts[("S", summary(kind, z[key])) + feature(kind, cid, pt)][lab] += 1
                    labels[kind].add(lab)
                    z[key] = true_step(kind, z[key], lab, i, tt[5] if len(tt) == 6 else None)
    return aggregate(counts), labels


def parent(f):
    """Back-off chain. A state-conditioned feature ('S', summary, *feature) backs off to the feature itself.
    Direct: (D, type, sig, src, lease) -> (D, type, sig, lease) -> (D, type, sig).
    Every kind: (k, type, sig) -> (k, 'SIG', sig) -> (k, 'ROOT'). (k, 'NONE') and (k, 'UNREL') are dense leaves."""
    if f[0] == "S":
        return f[2:]
    if f[0] == "D" and len(f) == 5:
        return f[:3] + (f[4],)
    if f[0] == "D" and len(f) == 4:
        return f[:3]
    if len(f) == 3 and f[1] != "SIG":
        return (f[0], "SIG", f[2])
    if len(f) == 3:
        return (f[0], "ROOT")
    return None


def aggregate(counts):
    """Counts of every feature on the back-off chain: a coarse feature holds the sum of its descendants."""
    agg = defaultdict(Counter)
    for g, c in counts.items():
        h = g
        while h is not None:
            agg[h].update(c)
            h = parent(h)
    return agg


def posterior(counts, labels, f, cache):
    """Hierarchical smoothing: counts at f plus BETA pseudo-counts distributed as the parent's posterior;
    the root is smoothed with ALPHA toward uniform over the labels of its kind."""
    if f in cache:
        return cache[f]
    c = counts.get(f, Counter())
    par = parent(f)
    if par is None:
        labs = labels[f[0] if f[0] != "S" else f[2]]
        n = sum(c.values()) + ALPHA * len(labs)
        out = {lab: (c.get(lab, 0) + ALPHA) / n for lab in labs}
    else:
        pp = posterior(counts, labels, par, cache)
        n = sum(c.values()) + BETA
        out = {lab: (c.get(lab, 0) + BETA * pp.get(lab, 0.0)) / n for lab in set(pp) | set(c)}
    cache[f] = out
    return out


# ---------------- forward recursion ----------------

def step_direct(dist, lab, t, pvt):
    """dist over records None / (allowed, rank, until); apply one possible true event at time t."""
    if lab == "none":
        return dist
    cls, src, has_lease = lab
    rank = RANK.get(src, 1)
    until = (pvt - 1) if (has_lease and pvt is not None) else None
    out = defaultdict(float)
    for rec, pr in dist.items():
        cur_rank = 0 if rec is None or (rec[2] is not None and rec[2] < t) else rec[1]
        new = (cls == "pos", rank, until) if rank >= cur_rank else rec
        out[new] += pr
    return out


def true_step_soft(kind, z, lab, t, vt):
    """Update of a hypothesized state by a hypothesized true label; the lease (if any) is the parsed one."""
    if kind == "D":
        if lab == "none":
            return z
        cls, src, has_lease = lab
        rank = RANK.get(src, 1)
        until = (vt - 1) if (has_lease and vt is not None and vt < 1000) else None
        cur_rank = 0 if z is None or (z[2] is not None and z[2] < t) else z[1]
        return (cls == "pos", rank, until) if rank >= cur_rank else z
    return true_step(kind, z, lab, t, vt)


def mix(dists_weights):
    out = defaultdict(float)
    for d, w in dists_weights:
        for k, v in d.items():
            out[k] += w * v
    return out


def bayes_answer(s, t, P, counts, labels, cache):
    p, r = s["p"], s["r"]
    comps = [("D", {"p": p, "r": r})] + [("M", {"p": p, "g": g}) for g in GROUPS] + \
            [("G", {"g": g, "r": r}) for g in GROUPS] + [("W", {"r": r})]
    state = {}
    for kind, cid in comps:
        state[(kind, tuple(cid.values()))] = {None: 1.0} if kind == "D" else ({0: 1.0} if kind in "MG" else {(0, 0): 1.0})
    for i, line in enumerate(s["lines"][:t]):
        pt = tuple(P[content(line)]["parsed"])
        for kind, cid in comps:
            key = (kind, tuple(cid.values()))
            f = feature(kind, cid, pt)
            vt = pt[5] if pt[0] != "NONE" else None
            new = defaultdict(float)
            for z, pz in state[key].items():
                post = posterior(counts, labels, ("S", summary(kind, z)) + f, cache)
                for lab, w in post.items():
                    if w > 0:
                        new[true_step_soft(kind, z, lab, i, vt)] += pz * w
            state[key] = {k: v for k, v in new.items() if v > 1e-12}
    tq = t - 1
    d = state[("D", (p, r))]
    p_direct = sum(v for rec, v in d.items() if rec is not None and rec[0] and (rec[2] is None or rec[2] >= tq))
    none_group = 1.0
    for g in GROUPS:
        pm = state[("M", (p, g))].get(1, 0.0)
        pg = state[("G", (g, r))].get(1, 0.0)
        none_group *= 1 - pm * pg
    p_perm = 1 - (1 - p_direct) * none_group
    w = state[("W", (r,))]
    p_window_ok = 1 - w.get((1, 0), 0.0)
    pa = p_perm * p_window_ok
    return ("APPROVE" if pa > 0.5 else "DENY"), pa


def plugin_answer(s, t, P):
    evs = [to_event(tuple(P[content(l)]["parsed"]), i) for i, l in enumerate(s["lines"][:t])]
    return answer_from_state(filter_state(evs, t - 1), s["p"], s["r"])


def truth_at(s, t):
    rev = replay_events(s)
    return replay([e for e in rev if e["t"] < t], s["p"], s["r"], t - 1)


def bal_err(rows):
    by = defaultdict(list)
    for y, a in rows:
        by[y].append(a != y)
    return sum(sum(v) / len(v) for v in by.values()) / len(by)


def score(streams, P, counts, labels):
    res, cache = {}, {}
    for t in CHECKPOINTS:
        rows_b, rows_p, agree, per_pair = [], [], 0, defaultdict(list)
        for s in streams:
            y = truth_at(s, t)
            b, _ = bayes_answer(s, t, P, counts, labels, cache)
            q = plugin_answer(s, t, P)
            rows_b.append((y, b)); rows_p.append((y, q)); agree += b == q
            per_pair[s["pair"]].append((y, b, q))
        res[t] = {"bayes": bal_err(rows_b), "plugin": bal_err(rows_p), "agree": agree / len(streams),
                  "per_pair": per_pair}
    return res


def boot_diff(per_pair, n=2000, seed=0):
    pairs = sorted(per_pair)
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        smp = [x for pr in (rng.choice(pairs) for _ in pairs) for x in per_pair[pr]]
        out.append(bal_err([(y, b) for y, b, q in smp]) - bal_err([(y, q) for y, b, q in smp]))
    out.sort()
    return out[int(0.025 * n)], out[int(0.975 * n) - 1]


# ---------------- independence ----------------

NAME = re.compile(r"\b(svc|db|bucket|queue|grp)_[a-z]+\b|T-\d{4}|Event \d{3}|\b\d+\b|security_office|access_admin|ops_console")


def template(c):
    return NAME.sub("#", c)


def independence(P, lines):
    """Per true type: error rate, intraclass correlation of the error indicator within wording templates."""
    by_type = defaultdict(lambda: defaultdict(list))
    for c in lines:
        row = P[c]
        tt = tuple(row["truth"])
        by_type[tt[0]][template(c)].append(0 if row["correct"] else 1)
    out = {}
    for typ, groups in by_type.items():
        gs = [v for v in groups.values() if len(v) >= 2]
        n = sum(len(v) for v in gs)
        if n < 50 or len(gs) < 2:
            continue
        mean = sum(sum(v) for v in gs) / n
        k = len(gs)
        ssb = sum(len(v) * (sum(v) / len(v) - mean) ** 2 for v in gs)
        ssw = sum(sum((x - sum(v) / len(v)) ** 2 for x in v) for v in gs)
        msb, msw = ssb / (k - 1), ssw / (n - k) if n > k else 0.0
        n0 = (n - sum(len(v) ** 2 for v in gs) / n) / (k - 1)
        icc = (msb - msw) / (msb + (n0 - 1) * msw) if (msb + (n0 - 1) * msw) > 0 else float("nan")
        pure = sum(1 for v in gs if sum(v) in (0, len(v))) / len(gs)
        out[typ] = {"n": n, "templates": k, "err": mean, "icc": icc, "pure_templates": pure}
    return out


def main(mode):
    cal = calib_streams()
    for parser in PARSERS:
        path = os.path.join(EXP, f"r3_parse_{parser}.jsonl")
        if not os.path.exists(path):
            continue
        P = load_parses(parser)
        if mode == "cv":
            pairs = sorted({s["pair"] for s in cal})
            half = {pr: (j % 2) for j, pr in enumerate(pairs)}
            folds = [[s for s in cal if half[s["pair"]] == h] for h in (0, 1)]
            agg = {t: {"bayes": [], "plugin": [], "agree": [], "pp": {}} for t in CHECKPOINTS}
            for h in (0, 1):
                counts, labels = fit(folds[1 - h], P)
                res = score(folds[h], P, counts, labels)
                for t in CHECKPOINTS:
                    for k in ("bayes", "plugin", "agree"):
                        agg[t][k].append(res[t][k])
                    agg[t]["pp"].update(res[t]["per_pair"])
            line = f"{parser} CV(cal): "
            for t in CHECKPOINTS:
                lo, hi = boot_diff(agg[t]["pp"])
                b, q = sum(agg[t]["bayes"]) / 2, sum(agg[t]["plugin"]) / 2
                line += f"t={t} bayes {b:.3f} plugin {q:.3f} diff {b - q:+.3f} [{lo:+.3f},{hi:+.3f}] agree {sum(agg[t]['agree']) / 2:.2f}; "
            print(line)
            cal_lines = [c for c, r in P.items() if "cal" in r["sets"]]
            ind = independence(P, cal_lines)
            print("   independence (cal lines):", "; ".join(f"{k}: n={v['n']} err={v['err']:.2f} ICC={v['icc']:.2f} pure={v['pure_templates']:.2f}"
                                                         for k, v in sorted(ind.items())))
        else:
            test, _ = streams_for("full")
            counts, labels = fit(cal, P)
            res = score(test, P, counts, labels)
            line = f"{parser} TEST: "
            for t in CHECKPOINTS:
                lo, hi = boot_diff(res[t]["per_pair"])
                line += f"t={t} bayes {res[t]['bayes']:.3f} plugin {res[t]['plugin']:.3f} diff {res[t]['bayes'] - res[t]['plugin']:+.3f} [{lo:+.3f},{hi:+.3f}] agree {res[t]['agree']:.2f}; "
            print(line)
            test_lines = [c for c, r in P.items() if "test" in r["sets"]]
            ind = independence(P, test_lines)
            print("   independence (test lines):", "; ".join(f"{k}: n={v['n']} err={v['err']:.2f} ICC={v['icc']:.2f} pure={v['pure_templates']:.2f}"
                                                          for k, v in sorted(ind.items())))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "cv")
