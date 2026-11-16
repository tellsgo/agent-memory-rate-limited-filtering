#!/usr/bin/env python3
"""R10, written after the run, report only: split the single-step errors on allowed fields of the field protocol
(ONE_F, and the keyed protocol ONE_K for comparison) from a correct prior state.

1. By position: the first event of a shipment, where the prior state holds no record of that shipment, against
   later events.
2. By the truth at that step: the field's true value changes (the rule's value after the event differs effectively
   from its prior value) or stays (the event is blocked or a no-op on that field). An error on a changing field is
   "left at prior" when the writer's value equals the prior value.
Counts only effective changes, as r9.effective does for the registered error rates. Totals reproduce
r10_step_stats.json. Reads the shipped r10_one*_<reader>.jsonl.gz files and the generator, CPU only.
Run: python3 -B r10_error_split.py > r10_error_split.out
"""
import gzip
import json
import os
import sys
from collections import defaultdict

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "domain2"))
import r8  # noqa: E402
import r9  # noqa: E402
import r10  # noqa: E402
import domain2 as D  # noqa: E402

DOM = 2


def main():
    streams = r10.streams()
    for reader in r10.READERS:
        one = {m: {} for m in ("field", "key")}
        for m in one:
            for rec in map(json.loads, gzip.open(os.path.join(HERE, f"r10_one{m}_{reader}.jsonl.gz"), "rt")):
                one[m][(rec["stream"], rec["k"])] = rec
        z = {m: defaultdict(int) for m in one}
        ops = {m: defaultdict(lambda: [0, 0]) for m in one}
        for key, pair, ups, _ in streams:
            rule = D.new_state()
            for k, (n, u) in enumerate(ups):
                prior = rule
                rule = r8.step2(rule, n, u)
                pos = "first" if u["shipment"] not in prior["shipments"] else "later"
                rc, pc = r8.comps(DOM, rule), r8.comps(DOM, prior)
                for m in one:
                    mc = r8.comps(DOM, r10.merge(m, prior, r10._load_out(m, one[m][(key, k)]), u, n))
                    for c in r8.touched(DOM, u):
                        truth = "change" if rc.get(c) != pc.get(c) and r9.effective(DOM, c, pc.get(c), rc.get(c)) else "stay"
                        z[m][f"fields {pos}"] += 1
                        z[m][f"fields {truth}"] += 1
                        if rc.get(c) == mc.get(c) or not r9.effective(DOM, c, rc.get(c), mc.get(c)):
                            continue
                        z[m]["errors"] += 1
                        z[m][f"errors {pos}"] += 1
                        z[m][f"errors {truth}"] += 1
                        if truth == "change":
                            left = not r9.effective(DOM, c, pc.get(c), mc.get(c))
                            z[m]["errors change, left at prior"] += left
                            ops[m][u["op"]][0 if left else 1] += 1
        for m in ("field", "key"):
            q = z[m]
            print(f"=== {reader}, {m} protocol (ONE_{m[0].upper()}), errors on allowed fields from a correct prior state")
            print(f"  allowed fields {q['fields first'] + q['fields later']}, errors {q['errors']}")
            for g in ("first", "later", "change", "stay"):
                lab = {"first": "first event of a shipment", "later": "later events", "change": "true value changes",
                       "stay": "true value stays"}[g]
                print(f"  {lab:28s} fields {q['fields ' + g]:6d}  errors {q['errors ' + g]:5d}  rate {q['errors ' + g] / q['fields ' + g]:.3f}")
            print(f"  errors on changing fields left at the prior value: {q['errors change, left at prior']} of {q['errors change']}")
            print("  by operation, errors on changing fields (left at prior, other): "
                  + ", ".join(f"{op} {a}/{b}" for op, (a, b) in sorted(ops[m].items())))


if __name__ == "__main__":
    main()
