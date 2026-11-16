#!/usr/bin/env python3
"""
R7-2 and R7-3 (PREREG, fresh blind review R3-M1, R1-M3/R3-M2): StateMem with the components the R5-3 adapter dropped,
and a repair-only replay of the R5-3 adapter.

R7-2, faithful StateMem on the first 100 pairs of R7-1 domain 1 (200 streams, 256 events), paired with R7-1 arms A and E.
Changes from the R5-3 adapter (r5_comparators.StateStore), each taken from the StateMem description (arXiv 2608.19652,
Sec. 5 and App. E):
  - every unit keeps what the encoder writes: content (no length cut), priority, source type, type, scope, the two
    typed link lists (coupled with, derived from) and the triggers; the encoder and the reader see type, scope, links
    and triggers with each unit
  - a PolicyEncoder call turns the policy text into units (content, priority, type, scope) once per scenario, before
    the first event; the scenario and its policy are the same for every stream, so the call is made once and its units
    start every store; the turn encoder's task context is the generic one (no policy text), as in the original
  - a deterministic syntax repair before parsing (bare unit ids quoted, trailing commas dropped), then the one retry as
    before; every raw output is logged in full
  - re-check: one hop over both link types, no model call, as in R5-3; triggers are stored and shown, none is executed
    (the description executes none)
  - read uncut, and cut to 512 reader tokens with binding units listed newest first
R7-3, repair-only replay: the R5-3 adapter, prompts, streams (the 80 X6 streams) and readers unchanged, with only the
syntax repair added before parsing; native and policy forms; every raw output logged. Batch nondeterminism of vLLM can
change outputs that were valid in R5-3 as well, and R5-3 did not log its valid outputs, so the replay reports the paired
change in accuracy and the failures that remain, not which updates changed.
  --phase policy   GPU (vLLM). The PolicyEncoder call -> r7_n2_policy_units.json
  --phase encode   GPU (vLLM). R7-2 turn encoder over the 200 streams -> r7_n2_mem.jsonl, r7_n2_events.jsonl.gz
  --phase replay   GPU (vLLM). R7-3 -> r7_n3_mem_statemem_{native,policy}.jsonl, r7_n3_events.jsonl.gz
  --phase read     GPU. Margins for R7-2 (uncut, 512 newest first) and R7-3 (uncut, 512 as R5-3) -> r7_n23_read_<reader>.jsonl
  --phase analyze  CPU -> r7_n23_analyze.out
"""
import argparse, gzip, json, os, re, sys, time
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r5_comparators as R  # noqa: E402
from state_machine import POLICY_V2  # noqa: E402

N_PAIRS = 100
PFX = ""
if os.environ.get("R7_SMOKE"):                      # with r7_n1 smoke seeds; R7-3 replays the first 3 X6 streams
    N_PAIRS, PFX = 2, "smoke_"
READERS = ("qwen2.5-14b", "phi-4")
POLICY_SYS = (
    "You are the policy encoder of a state maintenance system for a task-executing agent. Read the policy document "
    "and turn each rule into one state unit. Output a single JSON object and nothing else:\n"
    "{\"units\": [{\"content\": str, \"priority\": \"hard\"|\"soft\", \"type\": str, \"scope\": str}]}\n"
    "Use priority \"hard\" for rules that must hold. Keep each unit to one or two sentences and keep every condition of "
    "the rule.")


def repair(text):
    t = re.sub(r'(?<=[\[,:\s])(U\d+)(?=\s*[,\]\}])', r'"\1"', text)
    return re.sub(r',\s*([\]\}])', r'\1', t)


def parse(text):
    return R.parse_json(repair(text))


class FullStore:
    """StateMem store that keeps every field the encoder writes."""

    def __init__(self, policy_units):
        self.units, self.n = {}, 0
        for u in policy_units:
            self.new(dict(u, source_type="policy"), -1)

    def new(self, u, t):
        self.n += 1
        uid = f"U{self.n}"
        lst = lambda k: [d for d in (u.get(k) or []) if isinstance(d, str)]
        trig = [x for x in (u.get("triggers") or []) if isinstance(x, dict)]
        self.units[uid] = {"content": str(u.get("content", "")), "priority": "soft" if u.get("priority") == "soft" else "hard",
                           "source_type": str(u.get("source_type") or ""), "type": str(u.get("type") or ""),
                           "scope": str(u.get("scope") or ""), "coupled_with": lst("coupled_with"),
                           "derived_from": lst("derived_from"), "triggers": trig, "source": t, "status": "active", "recheck": None}
        return uid

    def apply(self, out, t):
        changed = []
        for s in out.get("supersede") or []:
            uid = s.get("unit_id") if isinstance(s, dict) else None
            if uid in self.units and self.units[uid]["status"] != "superseded":
                self.units[uid]["status"] = "superseded"
                changed.append((uid, str(s.get("reason", ""))))
                if isinstance(s.get("replacement"), dict) and s["replacement"].get("content"):
                    self.new(s["replacement"], t)
        for u in out.get("add") or []:
            if isinstance(u, dict) and u.get("content"):
                self.new(u, t)
        for uid, reason in changed:                  # one hop over both link types, no model call
            for w in self.units.values():
                if w["status"] == "active" and uid in w["coupled_with"] + w["derived_from"]:
                    w["status"], w["recheck"] = "needs_recheck", f"{uid} superseded ({reason})"

    def active(self):
        return [(k, v) for k, v in self.units.items() if v["status"] != "superseded"]

    @staticmethod
    def line(k, v, with_turn=True):
        meta = ", ".join(x for x in (v["priority"], v["type"] and f"type {v['type']}", v["scope"] and f"scope {v['scope']}",
                                     v["source_type"] and f"from {v['source_type']}") if x)
        links = "".join(f"; {n.replace('_', ' ')} {', '.join(v[n])}" for n in ("coupled_with", "derived_from") if v[n])
        trig = "".join(f"; trigger {x.get('kind', '')}: {x.get('detail', '')}" for x in v["triggers"])
        turn = ("policy" if v["source"] < 0 else f"turn {v['source']}") if with_turn else ""
        return f"[{k}] ({meta}{', ' + turn if turn else ''}) {v['content']}{links}{trig}"

    def encoder_view(self):
        a = self.active()
        return "\n".join(self.line(k, v) + (" [needs recheck]" if v["status"] == "needs_recheck" else "") for k, v in a) or "(no units yet)"

    def block(self, newest_first=False):
        a = self.active()
        hard = [(k, v) for k, v in a if v["status"] == "active" and v["priority"] == "hard"]
        soft = [(k, v) for k, v in a if v["status"] == "active" and v["priority"] == "soft"]
        rec = [(k, v) for k, v in a if v["status"] == "needs_recheck"]
        if newest_first:
            hard, soft, rec = hard[::-1], soft[::-1], rec[::-1]
        L = lambda xs, f: [f"- {self.line(k, v)}" + f(v) for k, v in xs]
        return "\n".join(["BINDING, THIS RECORD:"] + (L(hard, lambda v: "") or ["(none)"]) + ["PREFERENCES (yield if necessary):"]
                         + (L(soft, lambda v: "") or ["(none)"]) + ["NEEDS RECHECK:"]
                         + (L(rec, lambda v: f" -> recheck: {v['recheck']}") or ["(none)"]) + ["", R.SM_GUIDE])


def r7_streams():
    import r7_n1
    return [s for s in r7_n1.streams_d1() if s["pair"] < r7_n1.BASE1 + r7_n1.T * 10000 + N_PAIRS]


def llm_14b():
    from vllm import LLM
    return LLM(model=R.snapshot("Qwen/Qwen2.5-14B-Instruct"), max_model_len=32768, seed=0, enforce_eager=True,
               enable_prefix_caching=True, gpu_memory_utilization=float(os.environ.get("VLLM_UTIL", "0.85")))


def phase_policy():
    from vllm import SamplingParams
    llm = llm_14b()
    tok = llm.get_tokenizer()
    p = tok.apply_chat_template([{"role": "system", "content": POLICY_SYS},
                                 {"role": "user", "content": "Policy document:\n" + POLICY_V2 + "\n\nOutput the JSON object only."}],
                                tokenize=False, add_generation_prompt=True)
    r = llm.generate([p], SamplingParams(temperature=0.0, max_tokens=1536), use_tqdm=False)[0]
    o = parse(r.outputs[0].text)
    units = [u for u in (o or {}).get("units", []) if isinstance(u, dict) and u.get("content")]
    json.dump({"raw": r.outputs[0].text, "units": units, "prompt_tokens": len(r.prompt_token_ids),
               "gen_tokens": len(r.outputs[0].token_ids)}, open(os.path.join(HERE, PFX + "r7_n2_policy_units.json"), "w"), indent=1)
    print("policy units", len(units))
    assert units, "PolicyEncoder produced no units"


def run_encoder(jobs, task_of, log_path, snap_t=(256,)):
    """jobs: [(variant, stream, store)]. Shared loop of R5-3 with the syntax repair and full logging."""
    from vllm import SamplingParams
    llm = llm_14b()
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0.0, max_tokens=1536)
    cost = defaultdict(lambda: [0, 0, 0])
    bad = defaultdict(int)
    snaps = defaultdict(list)
    log = gzip.open(log_path, "wt")
    t0 = time.time()
    for t in range(256):
        prompts = []
        for v, s, st in jobs:
            user = (f"{task_of(v)}\n\nCurrent state:\n{st.encoder_view()}\n\nLatest turn (turn {t}):\n{s['lines'][t]}\n\n"
                    "Output the JSON object only.")
            prompts.append(tok.apply_chat_template([{"role": "system", "content": R.SM_SYSTEM}, {"role": "user", "content": user}],
                                                   tokenize=False, add_generation_prompt=True))
        res = llm.generate(prompts, sp, use_tqdm=False)
        first = [r.outputs[0].text for r in res]
        outs = [parse(x) for x in first]
        rtxt = {}
        retry = [i for i, o in enumerate(outs) if o is None]
        if retry:
            rr = llm.generate([prompts[i] + "{" for i in retry], sp, use_tqdm=False)
            for i, r in zip(retry, rr):
                rtxt[i] = r.outputs[0].text
                outs[i] = parse("{" + r.outputs[0].text)
                c = cost[(jobs[i][0], jobs[i][1]["key"])]
                c[0] += 1
                c[1] += len(r.prompt_token_ids)
                c[2] += len(r.outputs[0].token_ids)
        for i, ((v, s, st), r, o) in enumerate(zip(jobs, res, outs)):
            c = cost[(v, s["key"])]
            c[0] += 1
            c[1] += len(r.prompt_token_ids)
            c[2] += len(r.outputs[0].token_ids)
            log.write(json.dumps({"t": t, "variant": v, "stream": s["key"], "first": first[i], "finish": r.outputs[0].finish_reason,
                                  "retry": rtxt.get(i), "ok": o is not None, "out": o}) + "\n")
            if o is None:
                bad[(v, s["key"])] += 1
                continue
            st.apply(o, t)
        if t + 1 in snap_t:
            for v, s, st in jobs:
                c = cost[(v, s["key"])]
                snaps[v].append({"stream": s["key"], "pair": s["pair"], "truth": R.truth_at(s, t + 1), "t": t + 1,
                                 "memory": st.block(), "memory_newest_first": st.block(newest_first=True) if isinstance(st, FullStore) else None,
                                 "units_active": len(st.active()), "units_total": len(st.units), "write_calls": c[0],
                                 "write_tokens": c[1] + c[2], "failed_events": bad[(v, s["key"])]})
        if t % 16 == 15:
            print(f"t={t + 1} {time.time() - t0:.0f}s failed {sum(bad.values())}", flush=True)
    log.close()
    return snaps


def phase_encode():
    pol = json.load(open(os.path.join(HERE, PFX + "r7_n2_policy_units.json")))["units"]
    jobs = [("faithful", s, FullStore(pol)) for s in r7_streams()]
    snaps = run_encoder(jobs, lambda v: R.SM_TASK["native"], os.path.join(HERE, PFX + "r7_n2_events.jsonl.gz"))
    with open(os.path.join(HERE, PFX + "r7_n2_mem.jsonl"), "w") as f:
        for r in snaps["faithful"]:
            f.write(json.dumps(r) + "\n")


def phase_replay():
    ss = R.streams()[:3] if PFX else R.streams()
    jobs = [(v, s, R.StateStore()) for v in ("native", "policy") for s in ss]
    snaps = run_encoder(jobs, lambda v: R.SM_TASK[v], os.path.join(HERE, PFX + "r7_n3_events.jsonl.gz"), snap_t=R.CPS)
    for v in ("native", "policy"):
        with open(os.path.join(HERE, PFX + f"r7_n3_mem_statemem_{v}.jsonl"), "w") as f:
            for r in snaps[v]:
                f.write(json.dumps(r) + "\n")


def phase_read(reader):
    from x3_recursive_memory import cut_tokens, load_reader, margins_batch, ntok, reader_prompt
    from x6_parse_filter import query_at
    tok, model, _ = load_reader(reader, "cuda:0")
    w_head = model.lm_head.weight.detach().float()
    label_ids = {k: tok.encode(v, add_special_tokens=False) for k, v in {"APPROVE": " APPROVE", "DENY": " DENY"}.items()}
    qs = {s["key"]: s for s in r7_streams()} | {s["key"]: s for s in R.streams()}
    items = []
    if os.path.exists(os.path.join(HERE, PFX + "r7_n2_mem.jsonl")):
        for m in map(json.loads, open(os.path.join(HERE, PFX + "r7_n2_mem.jsonl"))):
            items.append(("R7-2", "statemem_faithful", None, m, m["memory"]))
            items.append(("R7-2", "statemem_faithful", 512, m, cut_tokens(tok, m["memory_newest_first"], 512)))
    for v in ("native", "policy"):
        p = os.path.join(HERE, PFX + f"r7_n3_mem_statemem_{v}.jsonl")
        if os.path.exists(p):
            for m in map(json.loads, open(p)):
                for B in (None, 512):
                    items.append(("R7-3", f"statemem_{v}", B, m, cut_tokens(tok, m["memory"], B) if B else m["memory"]))
    prompts = [reader_prompt(tok, "gov", x, query_at(qs[m["stream"]], m["t"])) for _, _, _, m, x in items]
    ms = margins_batch(tok, model, w_head, label_ids, prompts, "cuda:0", bs=int(os.environ.get("R7_READ_BS", "2")))
    with open(os.path.join(HERE, PFX + f"r7_n23_read_{reader}.jsonl"), "w") as f:
        for (row, meth, B, m, x), mg in zip(items, ms):
            f.write(json.dumps({"row": row, "method": meth, "B": B, "t": m["t"], "pair": m["pair"], "stream": m["stream"],
                                "truth": m["truth"], "margin": mg, "decision": "DENY" if mg > 0 else "APPROVE",
                                "memory_tokens": ntok(tok, x)}) + "\n")
    print(reader, "read", len(items), flush=True)


def phase_analyze():
    from fmt import r2
    from r2_analyze import bal
    from r5_comparators_analyze import boot
    sg = lambda x: ("+" if x >= 0 else "") + r2(x)
    out = []
    for rd in READERS:
        rr = [json.loads(l) for l in open(os.path.join(HERE, PFX + f"r7_n23_read_{rd}.jsonl"))]
        n1 = [r for r in map(json.loads, open(os.path.join(HERE, PFX + f"r7_n1_read_{rd}.jsonl"))) if r["domain"] == 1]
        f2 = {B: [r for r in rr if r["row"] == "R7-2" and r["B"] == B] for B in (None, 512)}
        if f2[None]:
            pairs = {r["pair"] for r in f2[None]}
            for B in (None, 512):
                out.append(f"{rd} R7-2 faithful StateMem, {'uncut' if B is None else '512 newest first'}: BA {r2(bal(f2[B]))}, "
                           f"memory tokens {sum(r['memory_tokens'] for r in f2[B]) / len(f2[B]):.0f}")
                for arm in ("A", "E"):
                    d = boot([r for r in n1 if r["arm"] == arm and r["pair"] in pairs], f2[B])
                    out.append(f"  R7-1 arm {arm} - faithful StateMem: {sg(d[0])} [{r2(d[1])}, {r2(d[2])}]")
        old = [json.loads(l) for l in open(os.path.join(HERE, f"r5_comp_{rd}.jsonl"))]
        for v in ("native", "policy"):
            for B in (None, 512):
                new = [r for r in rr if r["row"] == "R7-3" and r["method"] == f"statemem_{v}" and r["B"] == B and r["t"] == 256]
                if new:
                    o = [r for r in old if r["method"] == f"statemem_{v}" and r["B"] == B and r["t"] == 256]
                    d = boot(new, o)
                    out.append(f"{rd} R7-3 {v} {'uncut' if B is None else '512'}: replay {r2(bal(new))} against R5-3 {r2(bal(o))}, "
                               f"difference {sg(d[0])} [{r2(d[1])}, {r2(d[2])}]")
    for name in ("r7_n2_events.jsonl.gz", "r7_n3_events.jsonl.gz"):
        p = os.path.join(HERE, PFX + name)
        if os.path.exists(p):
            c = defaultdict(lambda: [0, 0, 0])
            for e in map(json.loads, gzip.open(p, "rt")):
                x = c[e["variant"]]
                x[0] += 1
                x[1] += e["retry"] is not None
                x[2] += not e["ok"]
            for v, (n, rt, bad) in c.items():
                out.append(f"{name} {v}: {n} events, {rt} retried, {bad} without a valid update after the repair and the retry")
    open(os.path.join(HERE, PFX + "r7_n23_analyze.out"), "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["policy", "encode", "replay", "read", "analyze"], required=True)
    ap.add_argument("--reader", default="")
    a = ap.parse_args()
    {"policy": phase_policy, "encode": phase_encode, "replay": phase_replay, "read": lambda: phase_read(a.reader),
     "analyze": phase_analyze}[a.phase]()
