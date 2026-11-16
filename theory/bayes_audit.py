#!/usr/bin/env python3
"""Audit of the calibrated Bayes filter (registration theory/REGISTRATION_round5.md, ACT-6).

  (i)   query-blind all-component run of the registered filter B, checked equal to B, with posterior storage
  (ii)  exact joint forward filter over the 10 query-relevant components (J-joint) and the product of its
        marginals (J-prod)
  (iii) memoryless confusion remap followed by the plug-in rule (R)
  (iv)  leave-template-out calibration of B and R
  (v)   family reweighting of the test streams

Usage: python3 bayes_audit.py [parser ...]     (default: all five parsers; results -> bayes_audit_results.json)
"""
import json
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bayes_filter as bf  # noqa: E402
from bayes_filter import (CHECKPOINTS, DIRECT, GROUPS, PRINCIPALS, RANK, RESOURCES, UNIVERSE, answer_from_state,  # noqa: E402
                          content, feature, filter_state, label, posterior, replay_events, summary, template,
                          to_event, true_step, true_step_soft, truth_at, truth_tuple)

PRUNE = 1e-9
TICKET = re.compile(r"^T-\d+$")
FAMILY_LIST = ["restore", "order", "expiry", "priority", "decoy", "implicit", "conj"]


# ---------------- (i) registered filter B, all components, model chosen per line ----------------

def raw_counts_by_template(streams, P):
    """Unaggregated counts of bf.fit, split by the template of the line."""
    by_t = defaultdict(lambda: defaultdict(Counter))
    labels = defaultdict(set)
    for s in streams:
        z = {(kind, tuple(cid.values())): bf.init(kind) for kind, uni in UNIVERSE.items() for cid in uni}
        for i, (line, ev) in enumerate(zip(s["lines"], replay_events(s))):
            tt = truth_tuple(ev)
            c = content(line)
            pt = tuple(P[c]["parsed"])
            w = template(c)
            for kind, uni in UNIVERSE.items():
                for cid in uni:
                    key = (kind, tuple(cid.values()))
                    lab = label(kind, cid, tt)
                    by_t[w][("S", summary(kind, z[key])) + feature(kind, cid, pt)][lab] += 1
                    labels[kind].add(lab)
                    z[key] = true_step(kind, z[key], lab, i, tt[5] if len(tt) == 6 else None)
    return by_t, labels


def sum_counts(by_t, exclude=None):
    tot = defaultdict(Counter)
    for w, cs in by_t.items():
        if w == exclude:
            continue
        for f, c in cs.items():
            tot[f].update(c)
    return tot


def run_components(s, P, model_of, tmax):
    """B's per-component recursion on all 68 records, no query. model_of(content) -> (agg counts, labels, cache).
    Returns {t: state} for t in CHECKPOINTS."""
    comps = [(kind, cid) for kind, uni in UNIVERSE.items() for cid in uni]
    state = {(kind, tuple(cid.values())): ({None: 1.0} if kind == "D" else ({0: 1.0} if kind in "MG" else {(0, 0): 1.0}))
             for kind, cid in comps}
    snaps = {}
    for i, line in enumerate(s["lines"][:tmax]):
        c = content(line)
        pt = tuple(P[c]["parsed"])
        counts, labels, cache = model_of(c)
        vt = pt[5] if pt[0] != "NONE" else None
        for kind, cid in comps:
            key = (kind, tuple(cid.values()))
            f = feature(kind, cid, pt)
            new = defaultdict(float)
            for z, pz in state[key].items():
                post = posterior(counts, labels, ("S", summary(kind, z)) + f, cache)
                for lab, w in post.items():
                    if w > 0:
                        new[true_step_soft(kind, z, lab, i, vt)] += pz * w
            state[key] = {k: v for k, v in new.items() if v > 1e-12}
        if i + 1 in CHECKPOINTS:
            snaps[i + 1] = {k: dict(v) for k, v in state.items()}
    return snaps


def decide(state, p, r, t):
    """The decision part of bf.bayes_answer, on stored marginals."""
    tq = t - 1
    d = state[("D", (p, r))]
    p_direct = sum(v for rec, v in d.items() if rec is not None and rec[0] and (rec[2] is None or rec[2] >= tq))
    none_group = 1.0
    for g in GROUPS:
        none_group *= 1 - state[("M", (p, g))].get(1, 0.0) * state[("G", (g, r))].get(1, 0.0)
    pa = (1 - (1 - p_direct) * none_group) * (1 - state[("W", (r,))].get((1, 0), 0.0))
    return "APPROVE" if pa > 0.5 else "DENY"


def storage(state):
    pts, bits = 0, 0.0
    idx_bits = {"D": math.log2(2 * 3 * 300 + 1), "M": 1, "G": 1, "W": 2}
    for (kind, _), dist in state.items():
        pts += len(dist)
        bits += len(dist) * (32 + idx_bits[kind])
    return pts, bits


# ---------------- (ii) exact joint filter over the query cluster ----------------

def pgroup(pt):
    if pt[0] == "NONE":
        return None
    for x in (pt[4], pt[3], pt[2]):
        if x in GROUPS:
            return x
    return None


def jfeature(pt, p0, r0):
    if pt[0] == "NONE":
        return ("J", "NONE")
    sp = "own" if pt[2] == p0 else ("other" if p0 in (pt[3], pt[4]) else "none")
    sr = "own" if pt[3] == r0 else ("other" if r0 in (pt[2], pt[4]) else "none")
    if sp == "none" and sr == "none":
        return ("J", "UNREL")
    lease = "none" if pt[5] is None else ("idx" if pt[5] < 1000 else "big")
    return ("J", pt[0], sp, sr, pt[1] if pt[1] in RANK else "bad", lease)


def jparent(f):
    if f[0] == "S":
        return f[2:]
    if f[0] == "J" and len(f) == 6:
        return f[:4]
    if f[0] == "J" and len(f) == 4:
        return f[:2]
    if f[0] == "J" and len(f) == 2 and f[1] not in ("ROOT", "NONE", "UNREL"):
        return ("J", "ROOT")
    return None


def jlabel(tt, pt, p0, r0):
    if tt[0] == "NONE":
        return "none"
    k, src, p, r, g, vt = tt
    pg = pgroup(pt)
    if k in DIRECT and (p, r) == (p0, r0):
        return ("D", "pos" if k in ("GRANT", "RECONFIRM") else "neg", src, vt is not None)
    if k in ("ADD", "REMOVE") and p == p0:
        return ("M", "pg" if g == pg else "og", k)
    if k == "GROUP_GRANT" and r == r0:
        return ("G", "pg" if g == pg else "og")
    if k in ("REQ_WINDOW", "OPEN", "CLOSE") and r == r0:
        return ("W", k)
    return "none"


def dsum(d):
    return "none" if d is None else ("allow" if d[0] else "deny")


def jsummary(z, pg):
    gi = GROUPS.index(pg) if pg is not None else None
    return (dsum(z[0]), z[9], z[1 + gi] if gi is not None else "n/a", z[5 + gi] if gi is not None else "n/a")


def japply(z, lab, pt, t):
    """List of (new joint state, weight) for one hypothesized effect."""
    if lab == "none":
        return [(z, 1.0)]
    kind = lab[0]
    if kind == "D":
        return [((true_step_soft("D", z[0], lab[1:], t, pt[5] if len(pt) == 6 else None),) + z[1:], 1.0)]
    if kind == "W":
        need, op = z[9]
        w = (1, op) if lab[1] == "REQ_WINDOW" else (need, 1 if lab[1] == "OPEN" else 0)
        return [(z[:9] + (w,), 1.0)]
    pg = pgroup(pt)
    gs = [pg] if (lab[1] == "pg" and pg is not None) else [g for g in GROUPS if g != pg]
    out = []
    for g in gs:
        j = (1 if kind == "M" else 5) + GROUPS.index(g)
        v = (1 if lab[2] == "ADD" else 0) if kind == "M" else 1
        out.append((z[:j] + (v,) + z[j + 1:], 1.0 / len(gs)))
    return out


def jinit():
    return (None, 0, 0, 0, 0, 0, 0, 0, 0, (0, 0))


def jtrue_state(full, p0, r0):
    return ((full[("D", (p0, r0))],) + tuple(full[("M", (p0, g))] for g in GROUPS)
            + tuple(full[("G", (g, r0))] for g in GROUPS) + (full[("W", (r0,))],))


def jfit(streams, P):
    counts = defaultdict(Counter)
    labels = set()
    keys = [(p, r) for p in PRINCIPALS for r in RESOURCES]
    for s in streams:
        full = {(kind, tuple(cid.values())): bf.init(kind) for kind, uni in UNIVERSE.items() for cid in uni}
        for i, (line, ev) in enumerate(zip(s["lines"], replay_events(s))):
            tt = truth_tuple(ev)
            pt = tuple(P[content(line)]["parsed"])
            pg = pgroup(pt)
            for p0, r0 in keys:
                lab = jlabel(tt, pt, p0, r0)
                counts[("S", jsummary(jtrue_state(full, p0, r0), pg)) + jfeature(pt, p0, r0)][lab] += 1
                labels.add(lab)
            for kind, uni in UNIVERSE.items():
                for cid in uni:
                    key = (kind, tuple(cid.values()))
                    full[key] = true_step(kind, full[key], label(kind, cid, tt), i, tt[5] if len(tt) == 6 else None)
    agg = defaultdict(Counter)
    for g, c in counts.items():
        h = g
        while h is not None:
            agg[h].update(c)
            h = jparent(h)
    return agg, sorted(labels, key=repr)


def jpost_dict(agg, labels, f, cache):
    if f in cache:
        return cache[f]
    c = agg.get(f, Counter())
    par = jparent(f)
    if par is None:
        n = sum(c.values()) + bf.ALPHA * len(labels)
        out = {lab: (c.get(lab, 0) + bf.ALPHA) / n for lab in labels}
    else:
        pp = jpost_dict(agg, labels, par, cache)
        n = sum(c.values()) + bf.BETA
        out = {lab: (c.get(lab, 0) + bf.BETA * pp.get(lab, 0.0)) / n for lab in set(pp) | set(c)}
    cache[f] = out
    return out


def jposterior(agg, labels, f, cache):
    """Hierarchical smoothing as bf.posterior, returned sorted by decreasing weight."""
    key = ("sorted",) + f
    if key not in cache:
        cache[key] = sorted(jpost_dict(agg, labels, f, cache).items(), key=lambda kv: -kv[1])
    return cache[key]


def japprove(z, t):
    tq = t - 1
    d = z[0]
    perm = d is not None and d[0] and (d[2] is None or d[2] >= tq)
    perm = perm or any(z[1 + j] and z[5 + j] for j in range(4))
    return perm and not (z[9][0] and not z[9][1])


def joint_run(s, P, model, tmax):
    agg, labels, cache = model
    p0, r0 = s["p"], s["r"]
    dist = {jinit(): 1.0}
    out, max_support, pruned = {}, 1, 0.0
    for i, line in enumerate(s["lines"][:tmax]):
        pt = tuple(P[content(line)]["parsed"])
        f = jfeature(pt, p0, r0)
        pg = pgroup(pt)
        new = defaultdict(float)
        for z, pz in dist.items():
            for lab, w in jposterior(agg, labels, ("S", jsummary(z, pg)) + f, cache):
                m = pz * w
                if m < PRUNE:
                    pruned += sum(pz * ww for _, ww in jposterior(agg, labels, ("S", jsummary(z, pg)) + f, cache) if pz * ww < PRUNE)
                    break
                for z2, w2 in japply(z, lab, pt, i):
                    new[z2] += m * w2
        tot = sum(new.values())
        dist = {k: v / tot for k, v in new.items()}
        max_support = max(max_support, len(dist))
        if i + 1 in CHECKPOINTS:
            t = i + 1
            pa_joint = sum(v for z, v in dist.items() if japprove(z, t))
            marg = [defaultdict(float) for _ in range(10)]
            for z, v in dist.items():
                for j in range(10):
                    marg[j][z[j]] += v
            tq = t - 1
            pdir = sum(v for d, v in marg[0].items() if d is not None and d[0] and (d[2] is None or d[2] >= tq))
            ng = 1.0
            for j in range(4):
                ng *= 1 - marg[1 + j].get(1, 0.0) * marg[5 + j].get(1, 0.0)
            pa_prod = (1 - (1 - pdir) * ng) * (1 - marg[9].get((1, 0), 0.0))
            out[t] = ("APPROVE" if pa_joint > 0.5 else "DENY", "APPROVE" if pa_prod > 0.5 else "DENY", len(dist))
    return out, max_support, pruned


# ---------------- (iii) memoryless remap ----------------

def vkind(x):
    if x is None:
        return "null"
    if x in PRINCIPALS:
        return "P"
    if x in RESOURCES:
        return "R"
    if x in GROUPS:
        return "G"
    if isinstance(x, str) and TICKET.match(x):
        return "TK"
    return "other"


def rkey(pt):
    if pt[0] == "NONE":
        return ("NONE",)
    lease = "none" if pt[5] is None else ("idx" if pt[5] < 1000 else "big")
    return (pt[0], pt[1] if pt[1] in RANK else "bad", lease, vkind(pt[2]), vkind(pt[3]), vkind(pt[4]))


def rrel(tt, pt):
    if tt[0] == "NONE":
        return ("NONE",)
    k, src, p, r, g, vt = tt
    fields = (("p", pt[2] if pt[0] != "NONE" else None), ("r", pt[3] if pt[0] != "NONE" else None),
              ("g", pt[4] if pt[0] != "NONE" else None))

    def slot(x):
        if x is None:
            return None
        for name, v in fields:
            if v == x:
                return name
        return "missing"
    lease = None if vt is None else ("same" if pt[0] != "NONE" and pt[5] == vt else "other")
    return (k, src, slot(p), slot(r), slot(g), lease)


def remap_counts_by_template(streams, P):
    by_t = defaultdict(lambda: defaultdict(Counter))
    for s in streams:
        for line, ev in zip(s["lines"], replay_events(s)):
            c = content(line)
            pt = tuple(P[c]["parsed"])
            by_t[template(c)][rkey(pt)][rrel(truth_tuple(ev), pt)] += 1
    return by_t


def remap_table(counts):
    table = {}
    for k, c in counts.items():
        if sum(c.values()) < 3:
            continue
        top = c.most_common(2)
        if len(top) == 2 and top[0][1] == top[1][1]:
            continue
        table[k] = top[0][0]
    return table


def apply_remap(pt, table):
    rel = table.get(rkey(pt))
    if rel is None:
        return pt
    if rel == ("NONE",):
        return ("NONE",)
    k, src, sp, sr, sg, lease = rel
    if "missing" in (sp, sr, sg) or lease == "other":
        return ("NONE",)
    get = {"p": pt[2], "r": pt[3], "g": pt[4], None: None}
    return (k, src, get[sp], get[sr], get[sg], pt[5] if lease == "same" else None)


def remap_answer(s, t, P, table_of):
    evs = []
    for i, l in enumerate(s["lines"][:t]):
        c = content(l)
        evs.append(to_event(apply_remap(tuple(P[c]["parsed"]), table_of(c)), i))
    return answer_from_state(filter_state(evs, t - 1), s["p"], s["r"])


# ---------------- statistics ----------------

def werr(rows):
    """rows: (truth, answer, weight). Weighted balanced error."""
    num, den = defaultdict(float), defaultdict(float)
    for y, a, w in rows:
        num[y] += w * (a != y)
        den[y] += w
    return sum(num[y] / den[y] for y in den) / len(den)


def boot(per_pair, m1, m2, weight, n=2000, seed=0):
    """per_pair: pair -> list of dicts with 'y', methods, 'fam'. CI of err(m1) - err(m2)."""
    pairs = sorted(per_pair)
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        smp = [x for pr in (rng.choice(pairs) for _ in pairs) for x in per_pair[pr]]
        out.append(werr([(x["y"], x[m1], weight(x)) for x in smp]) - werr([(x["y"], x[m2], weight(x)) for x in smp]))
    out.sort()
    return out[int(0.025 * n)], out[int(0.975 * n) - 1]


def main(parsers):
    cal = bf.calib_streams()
    test, _ = bf.streams_for("full")
    if os.environ.get("AUDIT_DEBUG"):           # crash test on 6 calibration streams (in sample), no test streams
        test = cal[:6]
    results = {}
    for parser in parsers:
        if not os.path.exists(os.path.join(bf.EXP, f"r3_parse_{parser}.jsonl")):
            continue
        P = bf.load_parses(parser)
        R = {}
        # B, registered fit (bf.fit), query-time function as registered
        counts, labels = bf.fit(cal, P)
        cacheB = {}
        # (i) + (iv) for B: counts by template
        by_t, labels_t = raw_counts_by_template(cal, P)
        std_model = (counts, labels, {})
        lto_models = {}

        def lto_model(c):
            w = template(c)
            if w not in by_t:
                return std_model
            if w not in lto_models:
                lto_models[w] = (bf.aggregate(sum_counts(by_t, exclude=w)), labels_t, {})
            return lto_models[w]
        # (iii) + (iv) for R
        rc = remap_counts_by_template(cal, P)
        tot = defaultdict(Counter)
        for cs in rc.values():
            for k, c in cs.items():
                tot[k].update(c)
        std_table = remap_table(tot)
        lto_tables = {}

        def lto_table(c):
            w = template(c)
            if w not in rc:
                return std_table
            if w not in lto_tables:
                sub = defaultdict(Counter)
                for ww, cs in rc.items():
                    if ww != w:
                        for k, c2 in cs.items():
                            sub[k].update(c2)
                lto_tables[w] = remap_table(sub)
            return lto_tables[w]
        # (ii) joint model
        jagg, jlabels = jfit(cal, P)
        jmodel = (jagg, jlabels, {})

        rows = {t: defaultdict(list) for t in CHECKPOINTS}
        mismatch, stor, jsup, jpruned = 0, {t: [] for t in CHECKPOINTS}, [], []
        for s in test:
            snaps = run_components(s, P, lambda c: std_model, max(CHECKPOINTS))
            snaps_lto = run_components(s, P, lto_model, max(CHECKPOINTS))
            jout, ms, pr = joint_run(s, P, jmodel, max(CHECKPOINTS))
            jsup.append(ms); jpruned.append(pr)
            for t in CHECKPOINTS:
                b, _ = bf.bayes_answer(s, t, P, counts, labels, cacheB)
                qb = decide(snaps[t], s["p"], s["r"], t)
                mismatch += b != qb
                stor[t].append(storage(snaps[t]))
                rows[t][s["pair"]].append({
                    "y": truth_at(s, t), "fam": s["family"], "B": b, "Bqb": qb,
                    "P": bf.plugin_answer(s, t, P),
                    "R": remap_answer(s, t, P, lambda c: std_table),
                    "B_lto": decide(snaps_lto[t], s["p"], s["r"], t),
                    "R_lto": remap_answer(s, t, P, lto_table),
                    "J": jout[t][0], "Jprod": jout[t][1]})
        R["i_mismatch"] = mismatch
        R["i_storage"] = {t: {"points_mean": sum(x[0] for x in v) / len(v), "points_max": max(x[0] for x in v),
                              "kbit_mean": sum(x[1] for x in v) / len(v) / 1000} for t, v in stor.items()}
        R["ii_support_max"] = max(jsup)
        R["ii_pruned_max"] = max(jpruned)
        R["remap_table_size"] = len(std_table)
        uni = lambda x: 1.0  # noqa: E731
        for t in CHECKPOINTS:
            pp = rows[t]
            flat = [x for v in pp.values() for x in v]
            cell = {}
            for m in ("B", "P", "R", "B_lto", "R_lto", "J", "Jprod"):
                cell["err_" + m] = werr([(x["y"], x[m], 1.0) for x in flat])
            for a, b2 in (("J", "Jprod"), ("B", "R"), ("J", "R"), ("B", "P"), ("R", "P"), ("B_lto", "P"),
                          ("B_lto", "R_lto"), ("R_lto", "P")):
                cell[f"ci_{a}-{b2}"] = boot(pp, a, b2, uni)
            cell["agree_J_Jprod"] = sum(x["J"] == x["Jprod"] for x in flat) / len(flat)
            fams = sorted({x["fam"] for x in flat})
            nf = Counter(x["fam"] for x in flat)
            cell["per_family"] = {f: {m: werr([(x["y"], x[m], 1.0) for x in flat if x["fam"] == f])
                                      for m in ("B", "R", "P")} for f in fams}
            weights = {"uniform": lambda x, nf=nf: 1.0 / nf[x["fam"]]}
            for f in FAMILY_LIST:
                weights["x3_" + f] = lambda x, f=f: 3.0 if x["fam"] == f else 1.0
            cell["reweighted"] = {}
            for name, wfun in weights.items():
                cell["reweighted"][name] = {
                    "B-P": werr([(x["y"], x["B"], wfun(x)) for x in flat]) - werr([(x["y"], x["P"], wfun(x)) for x in flat]),
                    "ci_B-P": boot(pp, "B", "P", wfun),
                    "B-R": werr([(x["y"], x["B"], wfun(x)) for x in flat]) - werr([(x["y"], x["R"], wfun(x)) for x in flat]),
                    "ci_B-R": boot(pp, "B", "R", wfun)}
            R[t] = cell
            print(f"{parser} t={t}: " + " ".join(f"{k[4:]} {v:.3f}" for k, v in cell.items() if k.startswith("err_"))
                  + f" | B-R {cell['err_B'] - cell['err_R']:+.3f} {tuple(round(x, 3) for x in cell['ci_B-R'])}"
                  + f" | J-Jprod {cell['err_J'] - cell['err_Jprod']:+.3f} agree {cell['agree_J_Jprod']:.3f}", flush=True)
        print(f"{parser}: (i) mismatches {mismatch}; storage {R['i_storage']}; (ii) max support {R['ii_support_max']}, "
              f"max pruned mass {R['ii_pruned_max']:.2e}; remap entries {len(std_table)}", flush=True)
        results[parser] = R
    out = os.path.join(HERE, "bayes_audit_debug.json" if os.environ.get("AUDIT_DEBUG") else "bayes_audit_results.json")
    old = json.load(open(out)) if os.path.exists(out) else {}
    old.update({p: {str(k): v for k, v in r.items()} for p, r in results.items()})
    json.dump(old, open(out, "w"), indent=1, default=str)


if __name__ == "__main__":
    main(sys.argv[1:] or bf.PARSERS)
