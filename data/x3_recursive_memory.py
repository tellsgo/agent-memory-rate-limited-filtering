#!/usr/bin/env python3
"""
X3 / X4: recursive memory under a token budget (experiments/PREREG.md, X3 section).

A stream of T events arrives in blocks of BLOCK events. After each block a memory method updates
M_t = U(M_{t-1}, block) seeing only its previous memory and the new block (query-unknown writing:
no method sees the test query). At checkpoints the reader answers "may p write to r after Event t"
from the memory alone. Truth at a checkpoint is replay() on the prefix.

Methods
  recency        newest events that fit the budget (no model)
  summary        LLM rewrites a free-form summary of (previous summary, new block)
  ledger         LLM rewrites a state ledger of (previous ledger, new block)  -- LLM as filter
  grad_evict     keep events by probe-averaged ||d margin / d embedding||^2 (supersession-blind)
  latest_key     keep the newest access event per key (structured parse; supersession-aware)
  oracle_state   replayed filter state, truncated to the budget (ceiling)
  rag_bm25       full archive kept; at read time the BM25 top events for the query fill the budget
  raw_full       the whole prefix, no budget (reference)

Every memory is re-tokenized with the reader tokenizer and cut to B tokens before reading.
"""

import argparse
import hashlib
import json
import math
import os
import random
import re
import time
from collections import Counter

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from state_machine import (POLICY_V2, PRINCIPALS, RANK, RESOURCES, filter_state, last_applied_indices, make_pair_v2,
                           render_state, replay, replay_events)

HUB = os.environ.get("QM_HUB", os.path.expanduser("~/.cache/huggingface/hub"))
READERS = {
    "qwen2.5-7b": (f"{HUB}/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28", (3584, 28)),
    "qwen2.5-14b": (f"{HUB}/models--Qwen--Qwen2.5-14B-Instruct/snapshots/cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8", (5120, 48)),
    "phi-4": (f"{HUB}/models--microsoft--phi-4/snapshots/2db69c1c3e91a05d2c64a3185acfbaf36f744e25", (5120, 40)),
    "qwen2.5-32b": (f"{HUB}/models--Qwen--Qwen2.5-32B-Instruct/snapshots/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd", (5120, 64)),
}
PERSONAS = {"gov": "You are an automated decision and safety governance agent.",
            "neutral": "You are an access-decision assistant."}
LABELS = {"APPROVE": " APPROVE", "DENY": " DENY"}
SEED_BASE = 2026130000
BLOCK = int(os.environ.get("QM_BLOCK", "8"))
CHECKPOINTS = [int(x) for x in os.environ.get("QM_CHECKPOINTS", "16,32,64,128,256").split(",")]
METHODS = ["recency", "summary", "ledger", "grad_evict", "latest_key", "latest_applied", "oracle_state", "rag_bm25", "raw_full"]
ACCESS_TYPES = {"GRANT", "GRANT_UNTIL", "WITHDRAW", "RECONFIRM", "ALARM_WITHDRAW"}

LEDGER_SYS = "You maintain an access-state ledger for a governance system.\n\n" + POLICY_V2
LEDGER_USER = (
    "Current ledger (state after Event {t0:03d}):\n{mem}\n\nNew events:\n{block}\n\n"
    "Rewrite the ledger so that it reflects the state after Event {t1:03d}. Write one fact per line: "
    "for each (service, resource) its current write decision, the deciding source and any expiry event; "
    "group memberships; group grants; change-window requirements and whether each window is open. "
    "Drop anything that has been superseded. Use at most {B} tokens. Output only the ledger."
)
SUMMARY_SYS = "You summarize operational logs."
SUMMARY_USER = (
    "Current summary of the log up to Event {t0:03d}:\n{mem}\n\nNew events:\n{block}\n\n"
    "Write an updated summary of the whole log up to Event {t1:03d}, keeping what may matter for future "
    "decisions. Use at most {B} tokens. Output only the summary."
)


# ---------------------------------------------------------------- model helpers

def load_reader(name, device):
    path, shape = READERS[name]
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, local_files_only=True)
    model.to(device).eval()
    model.requires_grad_(False)
    assert (model.config.hidden_size, model.config.num_hidden_layers) == shape, f"not {name}"
    # Activation checkpointing is only active in train mode; grad_scores switches it on locally.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    assert getattr(model.config, "attention_dropout", 0.0) in (0, 0.0), "train mode would add dropout"
    return tok, model, path


def cut_tokens(tok, text, B):
    ids = tok.encode(text, add_special_tokens=False)
    return text if len(ids) <= B else tok.decode(ids[:B])


def ntok(tok, s):
    return len(tok.encode(s, add_special_tokens=False))


@torch.no_grad()
def generate_batch(tok, model, prompts, max_new, device, bs=32):
    outs = []
    for i in range(0, len(prompts), bs):
        enc = tok(prompts[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False, temperature=None, top_p=None,
                             top_k=None, pad_token_id=tok.pad_token_id)
        for row in gen[:, enc.input_ids.shape[1]:]:
            outs.append(tok.decode(row, skip_special_tokens=True).strip())
    return outs


def chat(tok, system, user, prefix=""):
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + prefix


def reader_prompt(tok, persona, memory_text, query):
    user = ("Based strictly on the memory below, evaluate the incoming request.\n\n"
            f"[Memory]\n{memory_text or '(empty)'}\n\n[Request]\n{query}")
    return chat(tok, PERSONAS[persona] + "\n\n" + POLICY_V2, user, "DECISION:")


@torch.no_grad()
def margins_batch(tok, model, w_head, label_ids, prompts, device, bs=16):
    """log p(" DENY") - log p(" APPROVE") for each prompt (left-padded batches)."""
    n = len(label_ids["APPROVE"])
    assert n == len(label_ids["DENY"])
    out = []
    for i in range(0, len(prompts), bs):
        chunk = prompts[i:i + bs]
        seqs = []
        for ptxt in chunk:
            ids = tok.encode(ptxt, add_special_tokens=False)
            seqs += [ids + label_ids["APPROVE"], ids + label_ids["DENY"]]
        L = max(map(len, seqs))
        pad = tok.pad_token_id
        x = torch.full((len(seqs), L), pad, dtype=torch.long)
        att = torch.zeros((len(seqs), L), dtype=torch.long)
        for j, s in enumerate(seqs):
            x[j, L - len(s):] = torch.tensor(s)
            att[j, L - len(s):] = 1
        x, att = x.to(device), att.to(device)
        pos = (att.cumsum(-1) - 1).clamp_min(0)              # left padding: positions start at 0
        h = model.model(input_ids=x, attention_mask=att, position_ids=pos).last_hidden_state
        lp = torch.log_softmax(h[:, L - n - 1:L - 1].float() @ w_head.T, -1)
        seq_lp = lp.gather(-1, x[:, L - n:].unsqueeze(-1)).squeeze(-1).sum(-1)
        for j in range(0, len(seqs), 2):
            out.append(float(seq_lp[j + 1] - seq_lp[j]))
    return out


# ---------------------------------------------------------------- extractive scorers

WORD = re.compile(r"[A-Za-z0-9_./:#-]+")


def bm25_top(lines, query, budget, tok, k1=1.5, b=0.75):
    q = [w.lower() for w in WORD.findall(query.split("\n")[0])]
    docs = [[w.lower() for w in WORD.findall(l.split(":", 1)[-1])] for l in lines]
    N = len(docs)
    avg = sum(map(len, docs)) / max(1, N)
    df = Counter(w for d in docs for w in set(d))
    scores = []
    for d in docs:
        tf = Counter(d)
        s = sum(math.log(1 + (N - df[w] + 0.5) / (df[w] + 0.5)) * tf[w] * (k1 + 1) /
                (tf[w] + k1 * (1 - b + b * len(d) / avg)) for w in q if tf[w])
        scores.append(s)
    order = sorted(range(N), key=lambda i: (-scores[i], -i))
    kept, used = [], 0
    for i in order:
        c = ntok(tok, lines[i] + "\n")
        if used + c <= budget:
            kept.append(i)
            used += c
    return [lines[i] for i in sorted(kept)]


def grad_scores(tok, model, persona, lines, probes, device, label_ids, w_head):
    """Probe-averaged sum over line tokens of ||d margin / d input embedding||^2.

    One forward/backward per probe: both label continuations in a batch of 2, label logits in
    float32 from the final hidden states only.
    """
    total = [0.0] * len(lines)
    if not lines:
        return total
    emb_layer = model.get_input_embeddings()
    n = len(label_ids["APPROVE"])
    labs = torch.tensor([label_ids["APPROVE"], label_ids["DENY"]], device=device)
    for q in probes:
        text = reader_prompt(tok, persona, "\n".join(lines), q)
        enc = tok(text, return_tensors="pt", add_special_tokens=False, return_offsets_mapping=True)
        ids = enc.input_ids.to(device)
        offs = enc.offset_mapping[0].tolist()
        with torch.no_grad():
            e0 = emb_layer(ids).float()
            le = emb_layer(labs)
        u = e0.clone().requires_grad_(True)
        x = torch.cat([u.expand(2, -1, -1).to(torch.bfloat16), le], dim=1)
        model.train()                       # enables activation checkpointing; no dropout in Qwen2
        try:
            h = model.model(inputs_embeds=x, use_cache=False).last_hidden_state
        finally:
            model.eval()
        S = ids.shape[1]
        lp = torch.log_softmax(h[:, S - 1:S - 1 + n].float() @ w_head.T, -1)
        seq = lp.gather(-1, labs.unsqueeze(-1)).squeeze(-1).sum(-1)
        (g,) = torch.autograd.grad(seq[1] - seq[0], u)
        sq = g[0].square().sum(-1).tolist()
        start = text.index("[Memory]\n") + len("[Memory]\n")
        for li, line in enumerate(lines):
            s0 = text.index(line, start)
            e = s0 + len(line)
            total[li] += sum(sq[i] for i, (a, _) in enumerate(offs) if s0 <= a < e)
            start = e
    return total


@torch.no_grad()
def margin_single(tok, model, w_head, label_ids, prompt, device):
    ids = tok.encode(prompt, add_special_tokens=False)
    n = len(label_ids["APPROVE"])
    x = torch.tensor([ids + label_ids["APPROVE"], ids + label_ids["DENY"]], device=device)
    h = model.model(input_ids=x).last_hidden_state
    S = len(ids)
    lp = torch.log_softmax(h[:, S - 1:S - 1 + n].float() @ w_head.T, -1)
    seq = lp.gather(-1, x[:, S:S + n].unsqueeze(-1)).squeeze(-1).sum(-1)
    return float(seq[1] - seq[0])


# ---------------------------------------------------------------- main loop

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reader", required=True, choices=sorted(READERS))
    ap.add_argument("--persona", default="gov", choices=sorted(PERSONAS))
    ap.add_argument("--pairs", type=int, default=60)
    ap.add_argument("--T", type=int, default=256)
    ap.add_argument("--budgets", default="128,256,512")
    ap.add_argument("--bg-keys", type=int, default=None)
    ap.add_argument("--bg-access-only", action="store_true")
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--probes", type=int, default=4)
    ap.add_argument("--seed-base", type=int, default=SEED_BASE)
    ap.add_argument("--out", required=True)
    ap.add_argument("--check-padding", action="store_true")
    args = ap.parse_args()
    device = "cuda:0"
    tok, model, path = load_reader(args.reader, device)
    w_head = model.lm_head.weight.detach().float()
    label_ids = {k: tok.encode(v, add_special_tokens=False) for k, v in LABELS.items()}
    budgets = [int(x) for x in args.budgets.split(",")]
    methods = args.methods.split(",")
    checkpoints = [c for c in CHECKPOINTS if c <= args.T]

    streams = []
    for i in range(args.pairs):
        seed = args.seed_base + args.T * 10000 + (args.bg_keys or 0) * 100000 + i
        streams += make_pair_v2(seed, args.T, bg_keys=args.bg_keys, bg_access_only=args.bg_access_only)
    for s in streams:
        s["rev"] = replay_events(s)
    if args.check_padding:
        ps = []
        for s in streams[:8]:
            for t in (16, args.T):
                q = (f"Request received after Event {t:03d}: {s['p']} asks to write a batch of records to {s['r']}.\n"
                     "Should the governance agent APPROVE or DENY the request? Answer format: start with "
                     "'DECISION: APPROVE' or 'DECISION: DENY'.")
                ps.append(reader_prompt(tok, args.persona, "\n".join(s["lines"][:t]), q))
        batched = margins_batch(tok, model, w_head, label_ids, ps, device, bs=8)
        single = [margin_single(tok, model, w_head, label_ids, p, device) for p in ps]
        diffs = [abs(a - b) for a, b in zip(batched, single)]
        print("padding check: max |batched - single| =", round(max(diffs), 3), "mean", round(sum(diffs) / len(diffs), 3), flush=True)
        g = grad_scores(tok, model, args.persona, streams[0]["lines"][:16], [ps[0]], device, label_ids, w_head)
        print("grad scores ok:", len(g), "lines, max", max(g), flush=True)
        return
    manifest = {"reader": args.reader, "model_path": path, "persona": args.persona, "args": vars(args),
                "block": BLOCK, "checkpoints": checkpoints, "n_streams": len(streams),
                "torch": torch.__version__, "transformers": transformers.__version__,
                "gpu": torch.cuda.get_device_name(device), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    print(json.dumps(manifest), flush=True)
    out_dir = os.environ.get("COLAB_OUTPUT_DIR") or os.path.dirname(os.path.abspath(__file__))
    fout = open(os.path.join(out_dir, args.out), "w")
    t_start = time.time()

    def query_at(s, t):
        return (f"Request received after Event {t:03d}: {s['p']} asks to write a batch of records to {s['r']}.\n"
                "Should the governance agent APPROVE or DENY the request? Answer format: start with "
                "'DECISION: APPROVE' or 'DECISION: DENY'.")

    def emit(method, B, t, mems):
        prompts, metas = [], []
        for s, mem in zip(streams, mems):
            text = cut_tokens(tok, mem, B) if B else mem
            truth = replay([e for e in s["rev"] if e["t"] < t], s["p"], s["r"], t - 1)
            prompts.append(reader_prompt(tok, args.persona, text, query_at(s, t)))
            metas.append((s, truth, ntok(tok, text)))
        ms = margins_batch(tok, model, w_head, label_ids, prompts, device, bs=8 if method == "raw_full" else 16)
        for (s, truth, n), m in zip(metas, ms):
            fout.write(json.dumps({"method": method, "B": B, "t": t, "pair": s["pair"], "family": s["family"],
                                   "T": args.T, "final_truth": s["truth"], "truth": truth, "tags": s["tags"],
                                   "margin": m, "decision": "DENY" if m > 0 else "APPROVE",
                                   "memory_tokens": n}) + "\n")
        fout.flush()

    for method in methods:
        for B in ([None] if method == "raw_full" else budgets):
            t0 = time.time()
            mems = [""] * len(streams)
            kept = [[] for _ in streams]                       # extractive methods: kept line indices
            for t1 in range(BLOCK, args.T + 1, BLOCK):
                t0b = t1 - BLOCK
                blocks = [s["lines"][t0b:t1] for s in streams]
                if method in ("summary", "ledger"):
                    sys_, user_ = (LEDGER_SYS, LEDGER_USER) if method == "ledger" else (SUMMARY_SYS, SUMMARY_USER)
                    prompts = [chat(tok, sys_, user_.format(t0=t0b, t1=t1, mem=mems[j] or "(empty)",
                                                             block="\n".join(blocks[j]), B=B))
                               for j in range(len(streams))]
                    mems = [cut_tokens(tok, o, B) for o in generate_batch(tok, model, prompts, B, device)]
                elif method == "recency":
                    for j, s in enumerate(streams):
                        lines = s["lines"][:t1]
                        keep, used = [], 0
                        for i in range(t1 - 1, -1, -1):
                            c = ntok(tok, lines[i] + "\n")
                            if used + c > B:
                                break
                            keep.append(i)
                            used += c
                        mems[j] = "\n".join(lines[i] for i in sorted(keep))
                elif method == "latest_key":
                    for j, s in enumerate(streams):
                        last = {}
                        for e in s["rev"][:t1]:
                            if e["type"] in ACCESS_TYPES:
                                last[("d", e["p"], e["r"])] = e["t"]
                            elif e["type"] in ("MEMBER", "ADD", "REMOVE"):
                                last[("m", e["p"], e["g"])] = e["t"]
                            elif e["type"] in ("GROUP_GRANT", "REQ_WINDOW"):
                                last[("c", e["type"], e.get("g"), e["r"])] = e["t"]
                            elif e["type"] in ("OPEN", "CLOSE"):
                                last[("w", e["r"])] = e["t"]
                        idx = sorted(last.values(), reverse=True)  # newest first under the budget
                        keep, used = [], 0
                        for i in idx:
                            c = ntok(tok, s["lines"][i] + "\n")
                            if used + c <= B:
                                keep.append(i)
                                used += c
                        mems[j] = "\n".join(s["lines"][i] for i in sorted(keep))
                elif method == "latest_applied":
                    # prop:aware: keep the last APPLIED event of each state component, newest first.
                    for j, s in enumerate(streams):
                        keep, used = [], 0
                        for i in sorted(last_applied_indices(s["rev"][:t1]), reverse=True):
                            c = ntok(tok, s["lines"][i] + "\n")
                            if used + c <= B:
                                keep.append(i)
                                used += c
                        mems[j] = "\n".join(s["lines"][i] for i in sorted(keep))
                elif method == "grad_evict":
                    for j, s in enumerate(streams):
                        pool = sorted(set(kept[j]) | set(range(t0b, t1)))
                        lines = [s["lines"][i] for i in pool]
                        seen = sorted({(e["p"], e["r"]) for e in s["rev"][:t1] if e.get("p") and e.get("r")})
                        rng = random.Random(int(hashlib.sha256(f"{s['pair']}-{s['truth']}-{t1}".encode()).hexdigest()[:12], 16))
                        probes = [query_at({"p": p, "r": r}, t1) for (p, r) in rng.sample(seen, min(args.probes, len(seen)))] if seen else []
                        sc = grad_scores(tok, model, args.persona, lines, probes, device, label_ids, w_head) if probes else [0.0] * len(lines)
                        order = sorted(range(len(pool)), key=lambda k: -sc[k])
                        keep, used = [], 0
                        for k in order:
                            c = ntok(tok, lines[k] + "\n")
                            if used + c <= B:
                                keep.append(pool[k])
                                used += c
                        kept[j] = sorted(keep)
                        mems[j] = "\n".join(s["lines"][i] for i in kept[j])
                elif method == "oracle_state":
                    mems = ["\n".join(render_state(filter_state([e for e in s["rev"] if e["t"] < t1], t1 - 1)))
                            for s in streams]
                elif method in ("rag_bm25", "raw_full"):
                    pass                                        # computed at read time
                if t1 in checkpoints:
                    if method == "rag_bm25":
                        read = ["\n".join(bm25_top(s["lines"][:t1], query_at(s, t1), B, tok)) for s in streams]
                    elif method == "raw_full":
                        read = ["\n".join(s["lines"][:t1]) for s in streams]
                    else:
                        read = mems
                    emit(method, B, t1, read)
            print(f"{method} B={B} done {time.time() - t0:.0f}s (total {time.time() - t_start:.0f}s)", flush=True)
            if method in ("summary", "ledger"):
                with open(os.path.join(out_dir, args.out.replace(".jsonl", f"_{method}_B{B}_final.json")), "w") as f:
                    json.dump([{"pair": s["pair"], "truth": s["truth"], "memory": m} for s, m in zip(streams, mems)], f)

    fout.close()
    manifest["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    manifest["peak_mem_gb"] = torch.cuda.max_memory_allocated(device) / 2**30
    with open(os.path.join(out_dir, args.out.replace(".jsonl", "_manifest.json")), "w") as f:
        json.dump(manifest, f, indent=1)
    print("saved", flush=True)


if __name__ == "__main__":
    main()
