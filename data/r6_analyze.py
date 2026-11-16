#!/usr/bin/env python3
"""R6 analysis (PREREG R6-1, R6-2): Mem0 presence counts, H-R6-1 (mem0_dep minus R5 Mem0 top-10, t=256, B=512, Holm
over the two readers), and report-only contrasts against the R5 dependency retriever and the parsed filter."""
import json, os
from statistics import mean

from fmt import r2
from r2_analyze import bal, fmt_d
from r5_comparators_analyze import boot

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    pres = json.load(open(os.path.join(HERE, "r6_mem0_presence.json")))
    for t in (64, 256):
        p = [x for x in pres if x["t"] == t]
        d = sum(x["dep"] for x in p)
        print(f"Mem0 store t={t}: {d} dependency events; written intact {sum(x['intact'] for x in p)} "
              f"({r2(sum(x['intact'] for x in p) / d)}), returned by top-10 search {sum(x['retrieved'] for x in p)} "
              f"({r2(sum(x['retrieved'] for x in p) / d)}); store size {mean(x['store'] for x in p):.0f} memories, "
              f"{mean(x['no_ticket'] for x in p):.0f} without a ticket")
    fam = []
    for reader in ("qwen2.5-14b", "phi-4"):
        path = os.path.join(HERE, f"r6_read_{reader}.jsonl")
        if not os.path.exists(path):
            continue
        r6 = [json.loads(l) for l in open(path)]
        r5 = [json.loads(l) for l in open(os.path.join(HERE, f"r5_comp_{reader}.jsonl"))]
        x6 = [json.loads(l) for l in open(os.path.join(HERE, f"x6_read_{reader}.jsonl"))]
        get6 = lambda m, t, B: [r for r in r6 if r["method"] == m and r["t"] == t and r["B"] == B]
        get5 = lambda m, t, B: [r for r in r5 if r["method"] == m and r["t"] == t and r["B"] == B]
        filt = lambda t, B: [r for r in x6 if r["design"] == "full" and r["method"] == "parse_filter:qwen2.5-14b"
                             and r["t"] == t and r["B"] == B]
        print(f"\n=== {reader}")
        for m in ("mem0_dep", "mem0_all", "filter_proj"):
            for t in (64, 256):
                for B in (None, 512):
                    rows = get6(m, t, B)
                    if rows:
                        print(f"  {m:12s} t={t:3d} B={str(B):4s} BA {r2(bal(rows))}  tokens {mean(r['memory_tokens'] for r in rows):.0f}")
        d = boot(get6("mem0_dep", 256, 512), get5("mem0", 256, 512))
        fam.append((reader, d))
        print(f"  [registered] mem0_dep - Mem0 top-10 (t=256, 512): {fmt_d(d[:3])} p={d[3]:.4f}")
        for t in (64, 256):
            for lab, a, b in ((f"mem0_dep - retriever (t={t}, uncapped)", get6("mem0_dep", t, None), get5("retriever", t, None)),
                              (f"filter_proj - retriever (t={t}, uncapped)", get6("filter_proj", t, None), get5("retriever", t, None)),
                              (f"filter_proj - full filter state (t={t}, uncapped)", get6("filter_proj", t, None), filt(t, None))):
                if a and b:
                    print(f"  report: {lab}: {fmt_d(boot(a, b)[:3])}")
    if fam:
        print("\nH-R6-1, Holm over", len(fam), "tests:")
        order = sorted(range(len(fam)), key=lambda i: fam[i][1][3])
        stop = False
        for rank, i in enumerate(order):
            reader, d = fam[i]
            thr = 0.05 / (len(fam) - rank)
            rej = (not stop) and d[3] <= thr and d[1] > 0
            stop = stop or not rej
            print(f"  {reader:12s} {fmt_d(d[:3])} p={d[3]:.4f} Holm thr {thr:.4f} {'holds' if rej else 'does not hold'}")


if __name__ == "__main__":
    main()
