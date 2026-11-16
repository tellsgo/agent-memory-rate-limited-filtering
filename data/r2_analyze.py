#!/usr/bin/env python3
"""R2 analysis against experiments/PREREG.md (R2-B1, R2-B3, R2-C2, R2-D). Missing inputs are skipped.
Balanced accuracy counts a missing decision as wrong; intervals are 2000-resample pair bootstraps."""
import json, os, random, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from fmt import r2  # noqa: E402

POS = {"APPROVE": "APPROVE", "DENY": "DENY", "YES": "YES", "NO": "NO"}


def load(path):
    p = path if os.path.isabs(path) else os.path.join(HERE, path)
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else None


def bal(rs):
    by = defaultdict(list)
    for r in rs:
        by[r["truth"]].append(r.get("decision") == r["truth"])
    return sum(sum(v) / len(v) for v in by.values()) / len(by)


def boot_diff(a, b, n=2000, seed=0):
    """a, b: rows of two methods on the same pairs. Returns (diff, lo, hi) of bal(a) - bal(b)."""
    ga, gb = defaultdict(list), defaultdict(list)
    for r in a:
        ga[r["pair"]].append(r)
    for r in b:
        gb[r["pair"]].append(r)
    pairs = sorted(set(ga) & set(gb))
    rng = random.Random(seed)
    ds = []
    for _ in range(n):
        s = [rng.choice(pairs) for _ in pairs]
        ds.append(bal([r for p in s for r in ga[p]]) - bal([r for p in s for r in gb[p]]))
    ds.sort()
    return bal([r for p in pairs for r in ga[p]]) - bal([r for p in pairs for r in gb[p]]), ds[int(0.025 * n)], ds[int(0.975 * n) - 1]


def fmt_d(d):
    return f"{'+' if d[0] >= 0 else ''}{r2(d[0])} [{r2(d[1])}, {r2(d[2])}]"    # half-up, as everywhere in the paper


def r2a():
    for m in ("qwen2.5-14b", "phi-4"):
        x6, cot = load(f"x6_read_{m}.jsonl"), load(f"r2_cot_{m}.jsonl")
        pf = [r for r in x6 if r["design"] == "full" and r["method"] == "parse_filter:qwen2.5-14b" and r["B"] is None and r["t"] == 256]
        d = boot_diff(pf, cot)
        print(f"R2-A {m}: filter {r2(bal(pf))}, cot4096 {r2(bal(cot))}, no decision {sum(r['decision'] is None for r in cot)}/{len(cot)}, "
              f"diff {fmt_d(d)} {'holds' if d[0] >= 0.10 and d[1] > 0 else 'fails'}")


def r2b():
    for m in ("qwen2.5-14b", "phi-4"):
        x6 = load(f"x6_read_{m}.jsonl")
        pf = [r for r in x6 if r["design"] == "full" and r["method"] == "parse_filter:qwen2.5-14b" and r["B"] is None] if x6 else []
        led = [r for r in x6 if r["design"] == "full" and r["method"] == "ledger"] if x6 else []
        for ph, meth in (("apply", "llm_apply"), ("mem0", "mem0_ops")):
            rows = load(f"r2_{ph}_{m}.jsonl")
            if rows is None:
                print(f"R2-B {meth} {m}: missing"); continue
            for t in (64, 256):
                rt = [r for r in rows if r["t"] == t]
                pft = [r for r in pf if r["t"] == t]
                led_t = [r for r in led if r["t"] == t]
                line = (f"R2-B {meth} {m} t={t}: bal {r2(bal(rt))}, mentions_key {r2(sum(r['mentions_key'] for r in rt) / len(rt))}, "
                        f"tokens {sum(r['memory_tokens'] for r in rt) / len(rt):.0f}")
                if pft:
                    line += f", parse_filter {r2(bal(pft))}, filter - this {fmt_d(boot_diff(pft, rt))}"
                if led_t:
                    line += f", ledger {r2(bal(led_t))}"
                print(line)


def r2c():
    for m in ("qwen2.5-14b", "phi-4"):
        rows = load(f"r2c_{m}.jsonl")
        g = lambda c, T: [r for r in rows if r["cond"] == c and r["T"] == T]
        print(f"R2-C {m}: " + ", ".join(f"{c} {r2(bal(g(c, 64)))}/{r2(bal(g(c, 256)))}" for c in ("raw", "state", "parse_apply"))
              + f", cot4096 {r2(bal(g('cot4096', 256)))}; parse_apply - raw {fmt_d(boot_diff(g('parse_apply', 256), g('raw', 256)))}, "
              f"parse_apply - cot {fmt_d(boot_diff(g('parse_apply', 256), g('cot4096', 256)))}")


def r2c2():
    for m in ("qwen2.5-14b", "phi-4"):
        rows, old = load(f"r2c2_{m}.jsonl"), load(f"r2c_{m}.jsonl")
        if rows is None or old is None:
            print(f"R2-C2 {m}: missing"); continue
        cot = [r for r in old if r["cond"] == "cot4096"]
        for c in ("state_resolved", "parse_apply_resolved"):
            for T in (64, 256):
                rt = [r for r in rows if r["cond"] == c and r["T"] == T]
                fam = {f: r2(bal([r for r in rt if r["family"] == f])) for f in sorted({r["family"] for r in rt})}
                print(f"R2-C2 {m} {c} T={T}: bal {r2(bal(rt))}, tokens {sum(r['memory_tokens'] for r in rt) / len(rt):.0f}, families {fam}")
        st = [r for r in rows if r["cond"] == "state_resolved" and r["T"] == 256 and r["family"] == "expiry"]
        pa = [r for r in rows if r["cond"] == "parse_apply_resolved" and r["T"] == 256]
        d = boot_diff(pa, cot)
        print(f"  H-R2C2a {m}: state expiry T=256 {r2(bal(st))} (>= 0.80?) {'holds' if bal(st) >= 0.80 else 'fails'}")
        print(f"  H-R2C2b {m}: parse_apply_resolved - cot4096 {fmt_d(d)} {'holds' if d[0] >= 0.10 and d[1] > 0 else 'fails'}")


def r2d(base):
    m = "qwen2.5-32b"
    x12 = load(os.path.join(base, f"x12_{m}.jsonl"))
    if x12:
        for c in ("raw_gov", "state_gov"):
            print(f"R2-D X1/X2 {c}: " + ", ".join(f"T={T} {r2(bal([r for r in x12 if r['cond'] == c and r['T'] == T]))}"
                                                  for T in (16, 64, 256)))
    x6 = load(os.path.join(base, f"x6_read_{m}.jsonl"))
    cot = load(os.path.join(base, f"r2_cot_{m}.jsonl"))
    if x6:
        full = [r for r in x6 if r["design"] == "full"]
        for meth, B in sorted({(r["method"], r["B"]) for r in full}, key=str):
            print(f"R2-D X6 {meth} B={B}: " + ", ".join(f"t={t} {r2(bal([r for r in full if r['method'] == meth and r['B'] == B and r['t'] == t]))}"
                                                        for t in sorted({r['t'] for r in full})))
        pf = [r for r in full if r["method"] == "parse_filter:qwen2.5-14b" and r["B"] is None and r["t"] == 256]
        for meth in ("rag_bm25", "recency_relevance"):
            alt = [r for r in full if r["method"] == meth and r["t"] == 256]
            print(f"  filter - {meth} t=256: {fmt_d(boot_diff(pf, alt))}")
        if cot:
            print(f"R2-D cot4096 t=256: {r2(bal(cot))}, no decision {sum(r['decision'] is None for r in cot)}/{len(cot)}, "
                  f"mean trace {sum(r['trace_tokens'] for r in cot) / len(cot):.0f}; filter - cot {fmt_d(boot_diff(pf, cot))}")


if __name__ == "__main__":
    r2a()
    r2b()
    r2c()
    r2c2()
    r2d(sys.argv[1] if len(sys.argv) > 1 else HERE)
