#!/usr/bin/env python3
"""
R5-3 (PREREG, ACT-3 of reviews/REFEREE_v2.md): comparator memories on the 80 full-background X6 streams, checkpoints
t in {64, 256}. Every memory is written without the query; only the retriever and Mem0's search see the query, at
read time.

  --phase retriever   CPU. Query-time dependency retriever: from the raw archive, every line naming the queried
                      pair (p, r), p's membership lines, group-grant lines on r and r's window lines, in time order
                      with their event numbers. Uncapped, and capped at 512 reader tokens (newest whole lines kept).
  --phase statemem    GPU (vLLM). StateMem reimplemented from arXiv 2608.19652 (Sec. 5 and the App. E.1/E.3
                      description; prompt wording ours, no code was released): one encoder call per event into typed
                      state units, encoder-flagged supersession, a deterministic dependency re-check, and a state
                      block for the reader. Two variants: native (generic task context) and policy (the access
                      policy of App. A as task context). Encoder: Qwen2.5-14B-Instruct, greedy.
  --phase mem0        GPU server + mem0ai OSS library with default prompts and a compatible vLLM server
                      (Qwen2.5-14B-Instruct), a local embedder and a local vector store; one add() per event line,
                      search(query, limit=10) at read time.
  --phase read        GPU. Reader margins (likelihood, gov prompt, as everywhere) for every memory file present.

Outputs: r5_mem_<method>.jsonl (stream, t, memory, cost fields) and r5_comp_<reader>.jsonl (reader rows).
"""
import argparse, json, os, re, sys, time
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from state_machine import GROUPS, POLICY_V2, replay, replay_events  # noqa: E402
from x6_parse_filter import query_at, streams_for  # noqa: E402

CPS = (64, 256)
HUBS = [os.path.expanduser("~/.cache/huggingface/hub")]


def streams():
    test, _ = streams_for("full")
    for s in test:
        s["rev"] = replay_events(s)
        s["key"] = f"{s['pair']}-{s['truth']}"
    return test


def truth_at(s, t):
    return replay([e for e in s["rev"] if e["t"] < t], s["p"], s["r"], t - 1)


def write_mem(method, rows):
    with open(os.path.join(HERE, f"r5_mem_{method}.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


# ---------------------------------------------------------------- dependency retriever
def dependency_lines(s, t):
    p, r = s["p"], s["r"]
    out = []
    for i, line in enumerate(s["lines"][:t]):
        has_p, has_r = re.search(rf"\b{re.escape(p)}\b", line), re.search(rf"\b{re.escape(r)}\b", line)
        has_g = any(g in line for g in GROUPS)
        if (has_p and has_r) or (has_p and has_g) or (has_r and has_g) or (has_r and "window" in line.lower()):
            out.append(line)
    return out


def phase_retriever():
    rows = []
    for s in streams():
        for t in CPS:
            rows.append({"stream": s["key"], "pair": s["pair"], "truth": truth_at(s, t), "t": t,
                         "lines": dependency_lines(s, t), "write_calls": 0, "write_tokens": 0})
    write_mem("retriever", rows)
    n = [len(r["lines"]) for r in rows if r["t"] == 256]
    print(f"retriever: {len(rows)} memories, lines at t=256 mean {sum(n) / len(n):.1f} max {max(n)}")


# ---------------------------------------------------------------- StateMem
SM_SYSTEM = (
    "You are a state maintenance system for a task-executing agent. You read the latest turn of an ongoing record "
    "together with the current state, a set of state units, and you update the state. Later turns may supersede or "
    "update earlier information. Track hard constraints, soft preferences, the status of entities (active, closed or "
    "superseded) and dependencies between units.\n\n"
    "Possible actions: add new units; supersede existing units (give the unit id, a reason and a replacement unit); "
    "track progress on multi-item requests; link dependencies (coupled_with, derived_from, triggers); or make no "
    "change.\n\n"
    "Watch actively for supersession cues such as \"actually\", \"instead\", \"no longer\", \"switched to\", "
    "\"moved off\", \"cancelled\", \"revoked\", \"withdrawn\", \"restored\", \"expired\".\n\n"
    "Rules: store only information that is relevant to future decisions. Do not duplicate existing units. Use priority "
    "\"hard\" for verified facts, binding commitments and remaining obligations, and \"soft\" for preferences. Keep each "
    "unit to one sentence. Add coupled_with, derived_from or triggers only when a real dependency exists.\n\n"
    "Output a single JSON object and nothing else:\n"
    "{\"add\": [{\"content\": str, \"priority\": \"hard\"|\"soft\", \"source_type\": \"user\"|\"action\"|\"tool_result\", "
    "\"type\": str, \"scope\": str, \"coupled_with\": [unit ids], \"derived_from\": [unit ids], \"triggers\": "
    "[{\"kind\": \"date_passed\"|\"entity_closed\"|\"supersession_announced\"|\"cascade\", \"detail\": str}]}],\n"
    " \"supersede\": [{\"unit_id\": str, \"reason\": str, \"replacement\": {same fields as an add item}}]}\n"
    "If nothing changes, output {\"add\": [], \"supersede\": []}.")
SM_TASK = {
    "native": "Task context: an operations log of an access-governance system for service accounts, read event by event.",
    "policy": "Task context: an operations log of an access-governance system for service accounts, read event by event. "
              "The governance policy that decides access is:\n" + POLICY_V2,
}
SM_GUIDE = ("Recompute guidance: hard constraints take priority over recent soft preferences. A unit under NEEDS RECHECK "
            "is stale: do not use its stored value; re-derive it from the current units on the path to the question.")


class StateStore:
    def __init__(self):
        self.units, self.n = {}, 0

    def new(self, u, t):
        self.n += 1
        uid = f"U{self.n}"
        self.units[uid] = {"content": str(u.get("content", ""))[:400], "priority": "soft" if u.get("priority") == "soft" else "hard",
                           "source": t, "deps": [d for d in (u.get("coupled_with") or []) + (u.get("derived_from") or []) if isinstance(d, str)],
                           "status": "active", "trigger": None}
        return uid

    def apply(self, out, t):
        changed = []
        for s in out.get("supersede") or []:
            uid = s.get("unit_id") if isinstance(s, dict) else None
            if uid in self.units and self.units[uid]["status"] != "superseded":
                self.units[uid]["status"] = "superseded"
                changed.append((uid, str(s.get("reason", ""))[:200]))
                if isinstance(s.get("replacement"), dict) and s["replacement"].get("content"):
                    self.new(s["replacement"], t)
        for u in out.get("add") or []:
            if isinstance(u, dict) and u.get("content"):
                self.new(u, t)
        for uid, reason in changed:                     # deterministic re-check over dependency links, no LLM call
            for v, w in self.units.items():
                if w["status"] == "active" and uid in w["deps"]:
                    w["status"], w["trigger"] = "needs_recheck", f"{uid} superseded ({reason})"

    def active(self):
        return [(k, v) for k, v in self.units.items() if v["status"] != "superseded"]

    def encoder_view(self):
        a = self.active()
        if not a:
            return "(no units yet)"
        return "\n".join(f"[{k}] ({v['priority']}{', needs_recheck' if v['status'] == 'needs_recheck' else ''}, turn {v['source']}) {v['content']}" for k, v in a)

    def block(self):
        a = self.active()
        hard = [f"- [{k}] (turn {v['source']}) {v['content']}" for k, v in a if v["status"] == "active" and v["priority"] == "hard"]
        soft = [f"- [{k}] (turn {v['source']}) {v['content']}" for k, v in a if v["status"] == "active" and v["priority"] == "soft"]
        rec = [f"- [{k}] (turn {v['source']}) {v['content']} -> trigger: {v['trigger']}" for k, v in a if v["status"] == "needs_recheck"]
        return "\n".join(["BINDING, THIS RECORD:"] + (hard or ["(none)"]) + ["PREFERENCES (yield if necessary):"] + (soft or ["(none)"])
                         + ["NEEDS RECHECK:"] + (rec or ["(none)"]) + ["", SM_GUIDE])


def parse_json(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        o = json.loads(m.group(0))
        return o if isinstance(o, dict) else None
    except json.JSONDecodeError:
        return None


def snapshot(name):
    import glob
    org, n = name.split("/")
    for h in HUBS:
        s = sorted(glob.glob(f"{h}/models--{org}--{n}/snapshots/*"))
        if s:
            return s[0]
    raise FileNotFoundError(name)


def phase_statemem():
    from vllm import LLM, SamplingParams
    llm = LLM(model=snapshot("Qwen/Qwen2.5-14B-Instruct"), max_model_len=32768, gpu_memory_utilization=float(os.environ.get("VLLM_UTIL", "0.85")),
              seed=0, enforce_eager=True, enable_prefix_caching=True)
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0.0, max_tokens=1536)
    fail_log = open(os.path.join(HERE, "r5_statemem_failures.jsonl"), "w")
    ss = streams()
    jobs = [(v, s, StateStore()) for v in ("native", "policy") for s in ss]
    cost = defaultdict(lambda: [0, 0, 0])               # calls, prompt tokens, generated tokens
    bad = defaultdict(int)
    snaps = {v: [] for v in SM_TASK}
    t0 = time.time()
    for t in range(256):
        prompts = []
        for v, s, st in jobs:
            user = (f"{SM_TASK[v]}\n\nCurrent state:\n{st.encoder_view()}\n\nLatest turn (turn {t}):\n{s['lines'][t]}\n\n"
                    "Output the JSON object only.")
            prompts.append(tok.apply_chat_template([{"role": "system", "content": SM_SYSTEM}, {"role": "user", "content": user}],
                                                   tokenize=False, add_generation_prompt=True))
        res = llm.generate(prompts, sp, use_tqdm=False)
        outs = [parse_json(r.outputs[0].text) for r in res]
        retry = [i for i, o in enumerate(outs) if o is None]       # one retry with a stricter instruction
        if retry:
            rp = [prompts[i] + "{" for i in retry]
            rr = llm.generate(rp, sp, use_tqdm=False)
            for i, r in zip(retry, rr):
                fail_log.write(json.dumps({"t": t, "variant": jobs[i][0], "stream": jobs[i][1]["key"],
                                           "first": res[i].outputs[0].text[-600:], "finish": res[i].outputs[0].finish_reason,
                                           "retry": r.outputs[0].text[-300:]}) + "\n")
                outs[i] = parse_json("{" + r.outputs[0].text)
                c = cost[(jobs[i][0], jobs[i][1]["key"])]
                c[0] += 1
                c[1] += len(r.prompt_token_ids)
                c[2] += len(r.outputs[0].token_ids)
        for (v, s, st), r, o in zip(jobs, res, outs):
            c = cost[(v, s["key"])]
            c[0] += 1
            c[1] += len(r.prompt_token_ids)
            c[2] += len(r.outputs[0].token_ids)
            if o is None:
                bad[v] += 1
                continue
            st.apply(o, t)
        if t + 1 in CPS:
            for v, s, st in jobs:
                c = cost[(v, s["key"])]
                snaps[v].append({"stream": s["key"], "pair": s["pair"], "truth": truth_at(s, t + 1), "t": t + 1,
                                 "memory": st.block(), "units_active": len(st.active()), "units_total": len(st.units),
                                 "write_calls": c[0], "write_tokens": c[1] + c[2], "parse_failures": bad[v]})
        if t % 16 == 15:
            print(f"statemem t={t + 1} {time.time() - t0:.0f}s failures {dict(bad)}", flush=True)
    for v in SM_TASK:
        write_mem(f"statemem_{v}", snaps[v])


# ---------------------------------------------------------------- Mem0 (native OSS library)
def phase_mem0():
    from concurrent.futures import ThreadPoolExecutor
    from mem0 import Memory
    base = os.environ.get("MEM0_LLM_URL", "http://127.0.0.1:8011/v1")
    root = os.environ.get("MEM0_STORE", "mem0_store")
    os.makedirs(root, exist_ok=True)
    ss = streams()

    def config(key):
        return {"llm": {"provider": os.environ["MEM0_PROVIDER"], "config": {"model": os.environ.get("MEM0_MODEL", "qwen2.5-14b"), os.environ["MEM0_BASE_URL_KEY"]: base,
                                                         "api_key": "local", "temperature": 0.0, "max_tokens": 1500}},
                "embedder": {"provider": "huggingface", "config": {"model": os.environ.get("MEM0_EMBED", "BAAI/bge-small-en-v1.5")}},
                "vector_store": {"provider": "qdrant", "config": {"collection_name": f"s{key}", "path": os.path.join(root, key),
                                                                  "embedding_model_dims": 384, "on_disk": True}},
                "history_db_path": os.path.join(root, f"{key}.db")}

    def run(s):
        key = s["key"].replace("-", "_")
        m = Memory.from_config(config(key))
        out, t0 = [], time.time()
        for t in range(256):
            m.add([{"role": "user", "content": s["lines"][t]}], user_id=key)
            if t + 1 in CPS:
                hits = m.search(query_at(s, t + 1), filters={"user_id": key}, top_k=10)     # mem0 2.2.1 API
                hits = hits.get("results", hits) if isinstance(hits, dict) else hits
                allm = m.get_all(filters={"user_id": key}, top_k=100000)
                allm = allm.get("results", allm) if isinstance(allm, dict) else allm
                out.append({"stream": s["key"], "pair": s["pair"], "truth": truth_at(s, t + 1), "t": t + 1,
                            "memory": "\n".join(h["memory"] for h in hits), "stored": len(allm), "seconds": time.time() - t0})
        return out

    if os.environ.get("MEM0_STREAM"):                 # one stream per process: Mem0's local Qdrant allows one client per folder
        i = int(os.environ["MEM0_STREAM"])
        with open(os.path.join(HERE, "r5_mem0_parts", f"s{i:03d}.jsonl"), "w") as f:
            for r in run(ss[i]):
                f.write(json.dumps(r) + "\n")
        print(f"mem0 stream {i} done", flush=True)
        return
    rows = [json.loads(l) for p in sorted(os.listdir(os.path.join(HERE, "r5_mem0_parts")))
            for l in open(os.path.join(HERE, "r5_mem0_parts", p))]
    print(f"mem0 merge: {len(rows)} memories from {len(os.listdir(os.path.join(HERE, 'r5_mem0_parts')))} streams")
    write_mem("mem0", rows)


# ---------------------------------------------------------------- reader
def phase_read(reader):
    from x3_recursive_memory import cut_tokens, load_reader, margins_batch, ntok, reader_prompt
    tok, model, _ = load_reader(reader, "cuda:0")
    w_head = model.lm_head.weight.detach().float()
    label_ids = {k: tok.encode(v, add_special_tokens=False) for k, v in {"APPROVE": " APPROVE", "DENY": " DENY"}.items()}
    qs = {s["key"]: s for s in streams()}
    out_path = os.path.join(HERE, f"r5_comp_{reader}.jsonl")
    done = {(r["method"], r["B"]) for r in map(json.loads, open(out_path))} if os.path.exists(out_path) else set()
    for method in ("retriever", "statemem_native", "statemem_policy", "mem0"):
        path = os.path.join(HERE, f"r5_mem_{method}.jsonl")
        if not os.path.exists(path):
            continue
        mems = [json.loads(l) for l in open(path)]
        for B in (None, 512):
            if (method, B) in done:
                continue
            texts = []
            for m in mems:
                if method == "retriever":
                    lines = m["lines"]
                    if B:
                        keep, used = [], 0
                        for line in reversed(lines):            # newest whole lines that fit, shown in time order
                            c = ntok(tok, line) + 1
                            if used + c > B:
                                break
                            keep.append(line)
                            used += c
                        lines = keep[::-1]
                    texts.append("\n".join(lines))
                else:
                    texts.append(cut_tokens(tok, m["memory"], B) if B else m["memory"])
            prompts = [reader_prompt(tok, "gov", x, query_at(qs[m["stream"]], m["t"])) for m, x in zip(mems, texts)]
            ms = margins_batch(tok, model, w_head, label_ids, prompts, "cuda:0", bs=int(os.environ.get("R5_READ_BS", "8")))   # 32B on long StateMem memories: 2
            with open(out_path, "a") as f:
                for m, x, mg in zip(mems, texts, ms):
                    f.write(json.dumps({"method": method, "B": B, "t": m["t"], "pair": m["pair"], "stream": m["stream"],
                                        "truth": m["truth"], "margin": mg, "decision": "DENY" if mg > 0 else "APPROVE",
                                        "memory_tokens": ntok(tok, x)}) + "\n")
            print(f"{reader} {method} B={B} done", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["retriever", "statemem", "mem0", "read"], required=True)
    ap.add_argument("--reader", default="")
    a = ap.parse_args()
    {"retriever": phase_retriever, "statemem": phase_statemem, "mem0": phase_mem0}.get(a.phase, lambda: phase_read(a.reader))()
