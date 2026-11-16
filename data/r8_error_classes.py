"""Report only, written after the R8 results: single-step (ONE) errors on touched components, by update type."""
import sys, os, gzip, json
from collections import defaultdict, Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import r8, domain2 as D
for reader in ("qwen2.5-14b", "phi-4"):
    for dom in (1, 2):
        step = r8.step1 if dom == 1 else r8.step2
        one = defaultdict(dict)
        for rec in map(json.loads, gzip.open(f"r8_one_{reader}_d{dom}.jsonl.gz", "rt")):
            one[rec["stream"]][rec["k"]] = rec
        cat, tot = Counter(), Counter()
        for key, pair, ups, _ in r8.r82_streams(dom):
            rule = {} if dom == 1 else D.new_state()
            for k, (n, u) in enumerate(ups):
                prior = rule; rule = step(rule, n, u); rec = one[key][k]
                if not rec["ok"]: continue
                o = r8.comps(dom, r8.from_lines(dom, rec["state"], n)); rc = r8.comps(dom, rule); pc = r8.comps(dom, prior)
                for c in r8.touched(dom, u):
                    if dom == 1:
                        kind, src = u[0], u[1]
                        cur = pc.get(c) if c[0] == "access" else None
                        if cur is not None and cur[2] is not None and cur[2] < n:
                            cur = None
                        applies = not (cur is not None and r8.RANK[src] < r8.RANK[cur[1]])
                        lab = kind + (" applicable" if applies else " outranked")
                    else:
                        lab = u["op"] + ":" + c[1]
                    tot[lab] += 1
                    if rc.get(c) != o.get(c):
                        if dom == 1:
                            if not applies: sub = "applied although outranked"
                            elif o.get(c) is None: sub = "record not created"
                            else: sub = "wrong value"
                        else:
                            sub = "wrong"
                        cat[(lab, sub)] += 1
        E = sum(cat.values()); N = sum(tot.values())
        print(f"== {reader} d{dom}: {E}/{N} touched wrong")
        for (lab, sub), v in cat.most_common(8):
            print(f"   {lab:40s} {sub:32s} {v:5d} of {tot[lab]:5d} ({v/tot[lab]:.2f})  share {v/E:.2f}")
