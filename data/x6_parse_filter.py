#!/usr/bin/env python3
"""
X6 (PREREG revision round): a buildable filter. A parser LLM reads ONE event line (no query, no
other events) and emits a typed update; the transition rule of replay() applies it; the reader sees
render_state() of the result. Compared on the same streams with the oracle state, BM25 retrieval,
recency-plus-relevance retrieval with time stamps, and (for phi-4) an LLM ledger.

Outputs (COLAB_OUTPUT_DIR or this directory):
  x6_parse_<parser>.jsonl     per unique line: parsed JSON, normalized tuple, truth tuple, correct
  x6_read_<reader>.jsonl      per (method, B, t, stream): margin, decision, truth, memory tokens
"""

import argparse
import hashlib
import json
import math
import os
import re
import time
from collections import Counter

import torch

from state_machine import (RANK, filter_state, make_pair_v2, render_state, replay, replay_events)
from x3_recursive_memory import (LEDGER_SYS, LEDGER_USER, WORD, chat, cut_tokens, generate_batch,
                                 load_reader, margins_batch, ntok, reader_prompt)

TYPES = ["GRANT", "WITHDRAW", "RECONFIRM", "GROUP_GRANT", "ADD", "REMOVE", "REQ_WINDOW", "OPEN", "CLOSE", "NONE"]
PARSE_SYS = (
    "You convert one line of an access-control log into a JSON update. Output only JSON with keys "
    '"type", "source", "principal", "resource", "group", "valid_through". '
    'type is one of GRANT, WITHDRAW, RECONFIRM, GROUP_GRANT, ADD, REMOVE, REQ_WINDOW, OPEN, CLOSE, NONE. '
    "GRANT gives a service write access to a resource (valid_through is the event number if the grant "
    "is limited, else null). WITHDRAW takes a service's write access away (for any reason). RECONFIRM "
    "states again that a service holds write access. GROUP_GRANT gives a group write access. ADD puts a "
    "service into a group (including listing it as a member). REMOVE takes it out. REQ_WINDOW says a "
    "resource accepts writes only during an open change window. OPEN and CLOSE start or end a change "
    "window. NONE is any line that changes no access fact. source is the issuing office "
    "(security_office, access_admin or ops_console). Use null for fields that do not apply."
)
EXAMPLES = [
    ("access_admin granted svc_alpha write access to db_x valid through Event 040, ticket T-1111.",
     {"type": "GRANT", "source": "access_admin", "principal": "svc_alpha", "resource": "db_x", "group": None, "valid_through": 40}),
    ("ops_console started a change window on db_y, reference T-2222.",
     {"type": "OPEN", "source": "ops_console", "principal": None, "resource": "db_y", "group": None, "valid_through": None}),
    ("security_office reported that node 3 finished its scheduled health check in 120 ms.",
     {"type": "NONE", "source": "security_office", "principal": None, "resource": None, "group": None, "valid_through": None}),
]
TYPE_MAP = {"GRANT_UNTIL": "GRANT", "ALARM_WITHDRAW": "WITHDRAW", "MEMBER": "ADD", "FILLER": "NONE"}


def content(line):
    return line.split(": ", 1)[1]


def truth_tuple(ev):
    k = TYPE_MAP.get(ev["type"], ev["type"])
    if k == "NONE":
        return ("NONE",)
    until = ev.get("until")
    return (k, ev.get("src"), ev.get("p"), ev.get("r"), ev.get("g"), None if until is None else until + 1)


def parsed_tuple(js):
    if not isinstance(js, dict):
        return ("NONE",)
    k = str(js.get("type", "NONE")).upper()
    if k not in TYPES or k == "NONE":
        return ("NONE",)
    vt = js.get("valid_through")
    try:
        vt = int(vt) if vt is not None else None
    except (TypeError, ValueError):
        vt = None
    g = lambda key: (js.get(key) or None)
    return (k, g("source"), g("principal"), g("resource"), g("group"), vt)


def parse_json(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def to_event(tup, t):
    """Parsed tuple -> replay event (None if not applicable)."""
    if tup[0] == "NONE":
        return {"t": t, "type": "FILLER"}
    k, src, p, r, g, vt = tup
    need = {"GRANT": (p, r), "WITHDRAW": (p, r), "RECONFIRM": (p, r), "GROUP_GRANT": (g, r), "ADD": (p, g),
            "REMOVE": (p, g), "REQ_WINDOW": (r,), "OPEN": (r,), "CLOSE": (r,)}[k]
    if any(x is None for x in need) or (k in ("GRANT", "WITHDRAW", "RECONFIRM") and src not in RANK):
        return {"t": t, "type": "FILLER"}                # an unusable parse changes no state
    ev = {"t": t, "type": k, "src": src if src in RANK else "ops_console", "p": p, "r": r, "g": g}
    if k == "GRANT" and vt is not None:
        ev["type"] = "GRANT_UNTIL"
        ev["until"] = vt - 1                     # 1-based "valid through Event N" -> replay time
    return ev


def parse_prompts(tok, lines):
    shots = "\n\n".join(f"Line: {a}\nJSON: {json.dumps(b)}" for a, b in EXAMPLES)
    return [chat(tok, PARSE_SYS, shots + f"\n\nLine: {l}\nJSON:") for l in lines]


def bm25_scores(lines, query, k1=1.5, b=0.75):
    q = [w.lower() for w in WORD.findall(query.split("\n")[0].split(":", 1)[-1])]
    docs = [[w.lower() for w in WORD.findall(content(l))] for l in lines]
    N = len(docs)
    avg = sum(map(len, docs)) / max(1, N)
    df = Counter(w for d in docs for w in set(d))
    out = []
    for d in docs:
        tf = Counter(d)
        out.append(sum(math.log(1 + (N - df[w] + 0.5) / (df[w] + 0.5)) * tf[w] * (k1 + 1) /
                       (tf[w] + k1 * (1 - b + b * len(d) / avg)) for w in q if tf[w]))
    return out


def select(lines, scores, B, tok):
    order = sorted(range(len(lines)), key=lambda i: (-scores[i], -i))
    keep, used = [], 0
    for i in order:
        c = ntok(tok, lines[i] + "\n")
        if used + c <= B:
            keep.append(i)
            used += c
    return "\n".join(lines[i] for i in sorted(keep))


def query_at(s, t):
    return (f"Request received after Event {t:03d}: {s['p']} asks to write a batch of records to {s['r']}.\n"
            "Should the governance agent APPROVE or DENY the request? Answer format: start with "
            "'DECISION: APPROVE' or 'DECISION: DENY'.")


def streams_for(design):
    out = []
    if design == "full":
        for i in range(40):
            out += make_pair_v2(2026130000 + 256 * 10000 + i, 256)
        return out, [32, 64, 128, 256]
    K, T = design
    for i in range(40):
        out += make_pair_v2(2026160000 + T * 10000 + K * 100000 + i, T, bg_keys=K, bg_access_only=True)
    return out, [T]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="parser and reader model for this run")
    ap.add_argument("--parse-only", action="store_true")
    ap.add_argument("--parsers", default="qwen2.5-14b,phi-4", help="parsers whose outputs the reader uses")
    ap.add_argument("--ledger", action="store_true", help="also run an LLM ledger with this model (B=256)")
    ap.add_argument("--parse-designs", default="all", help="'all' or 'full': which streams to parse")
    ap.add_argument("--read-designs", default="all", help="'all' or 'full': which streams to read")
    args = ap.parse_args()
    out_dir = os.environ.get("COLAB_OUTPUT_DIR") or os.path.dirname(os.path.abspath(__file__))
    stop = os.environ.get("COLAB_STOP_FILE", "")
    device = "cuda:0"
    tok, model, path = load_reader(args.model, device)
    w_head = model.lm_head.weight.detach().float()
    label_ids = {k: tok.encode(v, add_special_tokens=False) for k, v in {"APPROVE": " APPROVE", "DENY": " DENY"}.items()}
    designs = ["full", (3, 64), (15, 64), (3, 256), (15, 256)]
    if args.read_designs == "full" and not args.parse_only:
        designs = ["full"]
    all_streams = {d: streams_for(d) for d in designs}
    for d, (ss, _) in all_streams.items():
        for s in ss:
            s["rev"] = replay_events(s)

    # 1. Parse every unique line content with this model.
    parse_path = os.path.join(out_dir, f"x6_parse_{args.model}.jsonl")
    if args.parse_only:
        uniq = {}
        for d, (ss, _) in all_streams.items():
            if args.parse_designs == "full" and d != "full":
                continue
            for s in ss:
                for line, ev in zip(s["lines"], s["rev"]):
                    uniq.setdefault(content(line), truth_tuple(ev))
        have = set()
        if os.path.exists(parse_path):                         # resume: periodic chunks already written
            have = {json.loads(l)["content"] for l in open(parse_path)}
        keys = [k for k in sorted(uniq) if k not in have]
        t0 = time.time()
        with open(parse_path, "a") as f:
            for a in range(0, len(keys), 2048):
                chunk = keys[a:a + 2048]
                outs = generate_batch(tok, model, parse_prompts(tok, chunk), 96, device, bs=64)
                for c, o in zip(chunk, outs):
                    pt = parsed_tuple(parse_json(o))
                    f.write(json.dumps({"content": c, "raw": o, "parsed": list(pt), "truth": list(uniq[c]),
                                        "correct": list(pt) == list(uniq[c])}) + "\n")
                f.flush()
                if stop and os.path.exists(stop):
                    print("stop file seen during parsing; partial parse saved", flush=True)
                    return
        print(f"parsed {len(keys)} new unique lines in {time.time() - t0:.0f}s", flush=True)
    if args.parse_only:
        return

    parsed = {}
    for p in args.parsers.split(","):
        pp = os.path.join(out_dir, f"x6_parse_{p}.jsonl")
        if os.path.exists(pp):
            parsed[p] = {r["content"]: tuple(r["parsed"]) for r in map(json.loads, open(pp))}
    print("parsers available:", list(parsed), flush=True)

    # 2. Read.
    read_path = os.path.join(out_dir, f"x6_read_{args.model}.jsonl")
    done = set()
    if os.path.exists(read_path):
        for r in map(json.loads, open(read_path)):
            done.add((str(r["design"]), r["method"], r["B"]))
    f = open(read_path, "a")
    for d in designs:
        ss, cps = all_streams[d]
        methods = [("oracle_state", None), ("rag_bm25", 512), ("recency_relevance", 512)]
        covered = [p for p in parsed if all(content(l) in parsed[p] for s in ss for l in s["lines"])]
        methods += [(f"parse_filter:{p}", None) for p in covered] + [(f"parse_filter:{p}", 512) for p in covered]
        if args.ledger and d == "full":
            methods.append(("ledger", 256))
        for method, B in methods:
            if (str(d), method, B) in done:
                continue
            if stop and os.path.exists(stop):
                print("stop file seen", flush=True)
                return
            t0 = time.time()
            mems = {t: [] for t in cps}
            if method == "ledger":
                ledger = [""] * len(ss)
                for t1 in range(16, 257, 16):
                    prompts = [chat(tok, LEDGER_SYS, LEDGER_USER.format(t0=t1 - 16, t1=t1, mem=ledger[j] or "(empty)",
                                                                        block="\n".join(s["lines"][t1 - 16:t1]), B=B))
                               for j, s in enumerate(ss)]
                    ledger = [cut_tokens(tok, o, B) for o in generate_batch(tok, model, prompts, B, device, bs=40)]
                    if t1 in cps:
                        mems[t1] = list(ledger)
            else:
                for t in cps:
                    for s in ss:
                        if method == "oracle_state":
                            m = "\n".join(render_state(filter_state([e for e in s["rev"] if e["t"] < t], t - 1)))
                        elif method.startswith("parse_filter:"):
                            P = parsed[method.split(":", 1)[1]]
                            evs = [to_event(P[content(l)], i) for i, l in enumerate(s["lines"][:t])]
                            m = "\n".join(render_state(filter_state(evs, t - 1)))
                        elif method == "rag_bm25":
                            lines = s["lines"][:t]
                            m = select(lines, bm25_scores(lines, query_at(s, t)), B, tok)
                        else:  # recency_relevance: normalized BM25 plus exponential recency, half-life 32 events
                            lines = s["lines"][:t]
                            sc = bm25_scores(lines, query_at(s, t))
                            mx = max(sc) or 1.0
                            m = select(lines, [x / mx + 0.5 ** ((t - 1 - i) / 32) for i, x in enumerate(sc)], B, tok)
                        mems[t].append(cut_tokens(tok, m, B) if B else m)
            for t in cps:
                prompts = [reader_prompt(tok, "gov", m, query_at(s, t)) for s, m in zip(ss, mems[t])]
                ms = margins_batch(tok, model, w_head, label_ids, prompts, device, bs=16)
                for s, m, mv in zip(ss, mems[t], ms):
                    truth = replay([e for e in s["rev"] if e["t"] < t], s["p"], s["r"], t - 1)
                    f.write(json.dumps({"design": str(d), "method": method, "B": B, "t": t, "pair": s["pair"],
                                        "family": s["family"], "truth": truth, "margin": mv,
                                        "decision": "DENY" if mv > 0 else "APPROVE", "memory_tokens": ntok(tok, m)}) + "\n")
            f.flush()
            print(f"{d} {method} B={B} {time.time() - t0:.0f}s", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
