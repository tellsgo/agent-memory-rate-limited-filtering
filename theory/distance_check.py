#!/usr/bin/env python3
"""Does discriminability at fixed T depend on how far the decisive event is from the query?
Model B (theory.tex Sec. 5) predicts no; a per-intervening-event contraction (Model C) predicts a
strong decay with distance. Uses batch_01 P0 margins (e2_0 = 7B, e2_0b = 14B) and regenerates the
pairs to recover the target slots. Statistic: within-pair ordering, P(margin_DENY > margin_APPROVE)."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
from state_machine import make_pair_v2

V2 = os.path.join(os.path.dirname(__file__), "../../../ideas/quant_memory_spark/batch_01/agent_benchmark/v2")
for name, fn in (("7B", "e2_0_rows.jsonl"), ("14B", "e2_0b_rows.jsonl")):
    rows = [json.loads(l) for l in open(os.path.join(V2, fn))]
    by = {}
    for r in rows:
        if r["prompt"] == "P0":
            by.setdefault((r["pair"], r["T"]), {})[r["truth"]] = r
    for T in (16, 64, 256):
        d_list, ok = [], []
        for (pair, t), v in by.items():
            if t != T or len(v) != 2:
                continue
            ep = make_pair_v2(pair, T, family=v["APPROVE"]["family"])
            assert ep[0]["family"] == v["APPROVE"]["family"]
            d_list.append(T - 1 - max(ep[0]["target_slots"]))
            ok.append(v["DENY"]["margin"] > v["APPROVE"]["margin"])
        d, ok = np.array(d_list), np.array(ok, float)
        cuts = [0, T // 8, T // 2, T]
        parts = []
        for lo, hi in zip(cuts[:-1], cuts[1:]):
            m = (d >= lo) & (d < hi)
            if m.sum():
                p = ok[m].mean()
                parts.append(f"d in [{lo},{hi}): {p:.2f} +- {np.sqrt(p * (1 - p) / m.sum()):.2f} (n={m.sum()})")
        rho = np.corrcoef(d, ok)[0, 1]
        print(f"{name} T={T}: pair ordering {ok.mean():.3f} (n={len(ok)}); " + "; ".join(parts) + f"; corr(d, ok)={rho:+.2f}")
