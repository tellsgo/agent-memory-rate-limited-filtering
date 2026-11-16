#!/usr/bin/env python3
"""
R2-C (PREREG second revision round): the independently specified second domain (domain2/, export
dock release). Conditions per reader: the raw log, the true state (domain2.render_state), and the
log parsed line by line by the reader model into domain2.UPDATE_SCHEMA and applied with the domain
author's apply_update. A reasoning-trace condition (up to 4096 tokens) runs at T=256.
Rows: r2c_<model>.jsonl (+ r2c_parse_<model>.jsonl) in COLAB_OUTPUT_DIR or this directory.
"""
import argparse, json, os, re, sys, time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "domain2"))
sys.path.insert(0, HERE)
import domain2 as D  # noqa: E402
from x3_recursive_memory import chat, generate_batch, load_reader, ntok  # noqa: E402

SPEC = open(os.path.join(HERE, "..", "domain2", "spec.md")).read()
READ_SYS = "You check export shipments against the dock release rules.\n\n" + SPEC
PARSE_SYS = ("You convert one line of an export dock log into a JSON update. Output only JSON matching this schema:\n"
             + json.dumps(D.UPDATE_SCHEMA) + "\n\nThe rules the log follows:\n" + SPEC)


def episodes():
    out = []
    for T in (64, 256):
        for i in range(40):
            seed = 2026180000 + T * 10000 + i
            for e in D.make_pair(seed, T):
                e["pair"] = seed
                out.append(e)
    return out


def few_shot():
    shots, seen = [], set()
    for seed in range(5000, 5100):
        for ep in D.make_pair(seed, 32):
            for line, ev in zip(ep["lines"], ep["events"]):
                op = D.to_update(ev)["op"]
                if op not in seen and len(shots) < 5:
                    seen.add(op)
                    shots.append((line, D.to_update(ev)))
    return "\n\n".join(f"Line: {l}\nJSON: {json.dumps(u)}" for l, u in shots)


@torch.no_grad()
def yes_no_margin(tok, model, w_head, prompts, device, bs=8):
    """log p(" NO") - log p(" YES") after each prompt; label sequences may differ in length."""
    labels = {k: tok.encode(v, add_special_tokens=False) for k, v in (("YES", " YES"), ("NO", " NO"))}
    out = []
    for i in range(0, len(prompts), bs):
        seqs, meta = [], []
        for p in prompts[i:i + bs]:
            ids = tok.encode(p, add_special_tokens=False)
            for k in ("YES", "NO"):
                seqs.append(ids + labels[k]); meta.append(len(labels[k]))
        L = max(map(len, seqs))
        x = torch.full((len(seqs), L), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros_like(x)
        for j, s in enumerate(seqs):
            x[j, L - len(s):] = torch.tensor(s); att[j, L - len(s):] = 1
        x, att = x.to(device), att.to(device)
        pos = (att.cumsum(-1) - 1).clamp_min(0)
        h = model.model(input_ids=x, attention_mask=att, position_ids=pos).last_hidden_state
        lps = []
        for j, n in enumerate(meta):
            lp = torch.log_softmax(h[j, L - n - 1:L - 1].float() @ w_head.T, -1)
            lps.append(float(lp.gather(-1, x[j, L - n:].unsqueeze(-1)).sum()))
        out += [lps[k + 1] - lps[k] for k in range(0, len(lps), 2)]
    return out


def reader_prompt(tok, memory, query, kind):
    tag = "[Log]" if kind == "log" else "[Current state]"
    return chat(tok, READ_SYS, f"{tag}\n{memory}\n\n[Question]\n{query} Answer YES or NO.") + "ANSWER:"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--skip-cot", action="store_true")
    args = ap.parse_args()
    out_dir = os.environ.get("COLAB_OUTPUT_DIR") or HERE
    device = "cuda:0"
    tok, model, _ = load_reader(args.model, device)
    w_head = model.lm_head.weight.detach().float()
    eps = episodes()
    rows = []
    t0 = time.time()
    # parse every distinct line content with this model
    uniq = sorted({l.split(": ", 1)[1] for e in eps for l in e["lines"]})
    truth_up = {}
    for e in eps:
        for l, ev in zip(e["lines"], e["events"]):
            u = dict(D.to_update(ev)); u.pop("event", None)
            truth_up.setdefault(l.split(": ", 1)[1], u)
    shots = few_shot()
    outs = generate_batch(tok, model, [chat(tok, PARSE_SYS, shots + f"\n\nLine: Event 001: {c}\nJSON:") for c in uniq], 96, device, bs=64)
    parsed = {}
    with open(os.path.join(out_dir, f"r2c_parse_{args.model}.jsonl"), "w") as f:
        for c, o in zip(uniq, outs):
            m = re.search(r"\{.*\}", o, re.S)
            try:
                u = json.loads(m.group(0)) if m else None
            except json.JSONDecodeError:
                u = None
            if isinstance(u, dict):
                u.pop("event", None)
            parsed[c] = u
            f.write(json.dumps({"content": c, "parsed": u, "truth": truth_up[c], "correct": u == truth_up[c]}) + "\n")
    print(f"parsed {len(uniq)} lines in {time.time() - t0:.0f}s", flush=True)

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

    conds = {
        "raw": lambda e: ("\n".join(e["lines"]), "log"),
        "state": lambda e: ("\n".join(D.render_state(D.current_state(e["events"], len(e["lines"])))), "state"),
        "parse_apply": lambda e: ("\n".join(D.render_state(parsed_state(e))), "state"),
    }
    for name, fn in conds.items():
        mems = [fn(e) for e in eps]
        prompts = [reader_prompt(tok, m, e["query"], k) for (m, k), e in zip(mems, eps)]
        ms = yes_no_margin(tok, model, w_head, prompts, device, bs=4 if name == "raw" else 16)
        for e, (m, _), mv in zip(eps, mems, ms):
            rows.append({"cond": name, "T": len(e["lines"]), "pair": e["pair"], "family": e["family"],
                         "truth": e["truth"], "margin": mv, "decision": "NO" if mv > 0 else "YES", "memory_tokens": ntok(tok, m)})
        print(name, f"{time.time() - t0:.0f}s", flush=True)
    if not args.skip_cot:
        e256 = [e for e in eps if len(e["lines"]) == 256]
        prompts = [chat(tok, READ_SYS, "[Log]\n" + "\n".join(e["lines"]) + f"\n\n[Question]\n{e['query']}\n\n"
                        "Go through every event that concerns this shipment in order and apply the rules step by step. "
                        "Then end with a final line that is exactly 'ANSWER: YES' or 'ANSWER: NO'.") for e in e256]
        outs = generate_batch(tok, model, prompts, 4096, device, bs=8)
        for e, o in zip(e256, outs):
            m = re.findall(r"ANSWER:\s*(YES|NO)", o.upper())
            rows.append({"cond": "cot4096", "T": 256, "pair": e["pair"], "family": e["family"], "truth": e["truth"],
                         "decision": m[-1] if m else None, "trace_tokens": ntok(tok, o)})
        print("cot", f"{time.time() - t0:.0f}s", flush=True)
    with open(os.path.join(out_dir, f"r2c_{args.model}.jsonl"), "w") as f:
        f.write("".join(json.dumps(r) + "\n" for r in rows))
    print("done", flush=True)


if __name__ == "__main__":
    main()
