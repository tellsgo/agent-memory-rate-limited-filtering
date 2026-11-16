#!/usr/bin/env python3
"""
R2-C2 (PREREG): the second domain with clearance expiry resolved at render time, matching how the
first domain renders leases. Same 160 episodes as R2-C, the parses from r2c_parse_<model>.jsonl are
reused, domain2/ is unchanged. Conditions: true state and parse + author rule, both rendered with
render_state_resolved. Rows: r2c2_<model>.jsonl in COLAB_OUTPUT_DIR or this directory.
"""
import argparse, json, os, re, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from r2c_domain2 import D, episodes, reader_prompt, yes_no_margin  # noqa: E402
from x3_recursive_memory import load_reader, ntok  # noqa: E402

LIMITED = re.compile(r"^(\S+) has customs clearance through Event (\d+)\.$")


def render_state_resolved(state):
    """The author's rendering, except a limited current clearance is resolved against the last event."""
    L = state["last_event"]
    out = []
    for line in D.render_state(state):
        m = LIMITED.match(line)
        if m:
            s, n = m.group(1), m.group(2)
            line = (f"{s} has customs clearance that expired after Event {n}." if int(n) < L
                    else f"{s} has customs clearance in force through Event {n}.")
        out.append(line)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()
    out_dir = os.environ.get("COLAB_OUTPUT_DIR") or HERE
    parsed = {r["content"]: r["parsed"] for r in map(json.loads, open(os.path.join(out_dir, f"r2c_parse_{args.model}.jsonl")))}

    def parsed_state(e):
        st = D.new_state()
        for i, l in enumerate(e["lines"]):
            u = parsed[l.split(": ", 1)[1]]
            if not isinstance(u, dict):
                st = dict(st, last_event=i + 1)
            else:
                try:
                    st = D.apply_update(st, dict(u, event=i + 1))      # returns a new state
                except Exception:
                    st = dict(st, last_event=i + 1)                    # an unusable parse changes nothing
        return st

    device = "cuda:0"
    tok, model, _ = load_reader(args.model, device)
    w_head = model.lm_head.weight.detach().float()
    eps = episodes()
    conds = {
        "state_resolved": lambda e: D.current_state(e["events"], len(e["lines"])),
        "parse_apply_resolved": parsed_state,
    }
    rows, t0 = [], time.time()
    for name, fn in conds.items():
        mems = ["\n".join(render_state_resolved(fn(e))) for e in eps]
        ms = yes_no_margin(tok, model, w_head, [reader_prompt(tok, m, e["query"], "state") for m, e in zip(mems, eps)], device, bs=16)
        for e, m, mv in zip(eps, mems, ms):
            rows.append({"cond": name, "T": len(e["lines"]), "pair": e["pair"], "family": e["family"],
                         "truth": e["truth"], "margin": mv, "decision": "NO" if mv > 0 else "YES", "memory_tokens": ntok(tok, m)})
        print(name, f"{time.time() - t0:.0f}s", flush=True)
    with open(os.path.join(out_dir, f"r2c2_{args.model}.jsonl"), "w") as f:
        f.write("".join(json.dumps(r) + "\n" for r in rows))
    print("done", flush=True)


if __name__ == "__main__":
    main()
