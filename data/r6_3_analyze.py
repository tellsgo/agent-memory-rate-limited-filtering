#!/usr/bin/env python3
"""R6-3 analysis (PREREG H-R6-3a per seed, H-R6-3b pooled, primary): gpt-oss-120b at 2,048 events, parsed state minus
full log, pair bootstrap 2000 seed 0. Pooled: each pair contributes its rows under both sampling seeds."""
import json, os
from statistics import mean

from fmt import r2
from r2_analyze import bal, boot_diff, fmt_d

HERE = os.path.dirname(os.path.abspath(__file__))


def load(s):
    return [json.loads(l) for l in open(os.path.join(HERE, f"r4_read_gpt-oss-120b_r6t2048s{s}.jsonl"))]


def main():
    pooled = {"log": [], "filter": []}
    for s in (0, 1):
        rows = load(s)
        g = {c: [r for r in rows if r["cond"] == c] for c in ("log", "filter")}
        for c in g:
            pooled[c] += g[c]
        d = boot_diff(g["filter"], g["log"])
        nd = {c: sum(r["decision"] is None for r in g[c]) for c in g}
        cut = {c: sum(r["finish"] == "length" for r in g[c]) for c in g}
        print(f"seed {s}: log {r2(bal(g['log']))} (no decision {nd['log']}, cut {cut['log']}, prompt {mean(r['prompt_tokens'] for r in g['log']):.0f} + gen "
              f"{mean(r['gen_tokens'] for r in g['log']):.0f}), filter {r2(bal(g['filter']))} (no decision {nd['filter']}, prompt "
              f"{mean(r['prompt_tokens'] for r in g['filter']):.0f} + gen {mean(r['gen_tokens'] for r in g['filter']):.0f}); "
              f"filter - log {fmt_d(d)} -> H-R6-3a {'holds' if d[0] >= 0.10 and d[1] > 0 else 'not met'}")
    d = boot_diff(pooled["filter"], pooled["log"])
    print(f"pooled: log {r2(bal(pooled['log']))}, filter {r2(bal(pooled['filter']))}; filter - log {fmt_d(d)} -> H-R6-3b "
          f"{'holds' if d[0] >= 0.10 and d[1] > 0 else 'not met'}")


if __name__ == "__main__":
    main()
