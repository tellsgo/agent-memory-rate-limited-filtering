#!/usr/bin/env python3
"""
R8 (PREREG, second fresh blind review, independent referee): who executes the transition, with semantics, parse,
representation, selection and reader held fixed.

Fresh paired streams that no earlier analysis touched, 256 events:
  domain 1  make_pair_v2(2026700000 + 256*10000 + i, 256), i < 200
  domain 2  D.make_pair(2026710000 + 256*10000 + i, 256), frozen R5-4 wording, i < 100
Changes shared by every arm (registered):
  glossary    one sentence per operation type, shown to every reader and writer (GLOSS_D1, GLOSS_D2)
  renderer v2 every component the history ever touched is written with its current value, including a removed
              membership ("is not a member") and a resource without a window rule; after selection, the queried
              pair's missing records are written as absent (no direct record, not a member of a granting group,
              no window rule). One renderer for every resolved state, whoever produced it.
  selection   the R5-3 dependency rule on domain 1 (r5_comparators.dependency_lines), the queried shipment's lines
              on domain 2, identical in every arm
R8-1 (200 + 100 pairs, Qwen2.5-14B parses every line with the frozen prompts):
  A      raw dependency lines
  Dp     the parser's typed updates of the same lines, key=value
  Enat   the filter's state (parsed events, exact rule) as renderer-v2 sentences, query rows
  Etyp   the same state as key=value records, query rows
  C      A, and the reader first writes a summary of at most 250 words (as R7-1)
  H8-1: resolution main effect 1/2[(Enat - A) + (Etyp - Dp)] >= 0.05, lower limit > 0, Holm over 4 cells.
R8-1c (reuse, report only): the R7-2 StateMem stores rebuilt from their logged encoder outputs and rendered in the
  StateMem paper's own read format (App. E.3: "- [type] content", four sections, triggers, the verbatim recompute
  guidance in the system message), whole and projected to the query.
R8-2 (first 100 pairs of each domain; the generator's true events as the correct parse; null events skipped):
  RULE   the exact rule
  ONE    at every update the writer gets the rule's state before it, the update, the policy and the glossary, and
         writes the complete next state in the canonical record format (teacher forced, all steps independent)
  FREE   the same prompt, but the prior state is the writer's own previous output, from the empty state
  INJ    CPU: the rule, but wherever ONE's state differs from the rule's after a step, ONE's values are taken
  Each final state is rendered by renderer v2, selected and read by the writer model.
  H8-2: BA(RULE) - BA(FREE) >= 0.10, lower limit > 0, Holm over 4 cells (2 writers x 2 domains).
Phases (vLLM for generation, HF margins for reading, as in R7):
  parse | build | gen --reader R | read --reader R | sm_build | sm_read --reader R | one --reader R --domain D |
  free --reader R --domain D | inj | read2 --reader R | analyze
"""
import argparse, gzip, json, os, random, re, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "domain2"))
from state_machine import GROUPS, OVERRIDE_RULE_V2, POLICY_V2, RANK, make_pair_v2, replay, replay_events  # noqa: E402

N1, N2, N_PAIRS2 = 200, 100, 100
BASE1, BASE2, T = 2026700000, 2026710000, 256
PFX, SMOKE_STEPS = "", None
if os.environ.get("R8_SMOKE"):                      # R8-2 smoke streams stop after 12 updates
    SMOKE_STEPS = 12
if os.environ.get("R8_SMOKE"):                      # smoke test on other seeds, a few pairs, separate files
    N1, N2, N_PAIRS2, BASE1, BASE2, PFX = 3, 3, 2, 2026800000, 2026810000, "smoke_"
READERS = ("qwen2.5-14b", "phi-4")
ARMS1 = ("A", "Dp", "Enat", "Etyp", "C")
WORDS, GEN_MAX, W_MAX = 250, 450, 1500
SUMMARY_D1 = (f"Before deciding, write a summary of at most {WORDS} words of the information in the memory that is relevant "
              "to the request. Then end with the decision line.")
SUMMARY_D2 = (f"Before answering, write a summary of at most {WORDS} words of the information in the log that is relevant "
              "to the question. Then end with the answer line.")

GLOSS_D1 = (
    "Meaning of typed updates and state records:\n"
    "- type=GRANT: the source gives the service write access to the resource. With valid_through=Event N the grant "
    "applies through Event N and no longer applies after it.\n"
    "- type=WITHDRAW: the source removes the service's write access to the resource.\n"
    "- type=RECONFIRM: the source restates that the service may write to the resource. It acts as a grant from that "
    "source with no time limit.\n"
    "- type=GROUP_GRANT: the group holds write access to the resource.\n"
    "- type=ADD: the service becomes a member of the group. type=REMOVE: the service is no longer a member.\n"
    "- type=REQ_WINDOW: from now on the resource accepts writes only while its change window is open.\n"
    "- type=OPEN and type=CLOSE: the resource's change window opens or closes.\n"
    "- A GRANT, WITHDRAW or RECONFIRM applies only if its source ranks at least as high as the source of the current "
    "decision on that service and resource. A grant whose last valid event is before the current event has expired "
    "and holds no rank, so any later access event applies. A later applicable event replaces an earlier one.\n"
    "- In state records, access gives the current decision on a service and resource (allowed=yes or no, the deciding "
    "source, and valid_through if the grant is limited; expired=yes means a limited grant has run out), member=no means "
    "the service is not in the group, record=none means nothing is recorded, and requires_window=no means the "
    "resource accepts writes without a window.")
GLOSS_D2 = (
    "Meaning of typed updates and state records:\n"
    "- op=grant_clearance: customs grants a clearance, through Event N if through=N is given, otherwise with no limit. "
    "It replaces any current clearance and erases any remembered one.\n"
    "- op=revoke_clearance: the current clearance is removed and remembered. With no current clearance nothing changes.\n"
    "- op=reinstate_clearance: the remembered clearance comes back with its original limit and the memory is erased. "
    "With nothing remembered nothing changes.\n"
    "- op=confirm_payment and op=reverse_payment: payment becomes confirmed or not confirmed.\n"
    "- op=place_hold with source=compliance or source=floor: that side's hold is placed (once per side).\n"
    "- op=lift_hold: a Compliance lift removes every hold, a floor lift removes only the floor hold.\n"
    "- In state records, clearance and remembered are none, no_limit or through:N, payment is confirmed or "
    "not_confirmed, and holds lists the sides holding the shipment or none.")
FMT_D1 = (
    "State format, one record per line, sorted:\n"
    "access service=<service> resource=<resource> allowed=<yes|no> source=<source> valid_through=<Event N|none>\n"
    "member service=<service> group=<group> member=<yes|no>\n"
    "group_grant group=<group> resource=<resource>\n"
    "window resource=<resource> requires_window=<yes|no> open=<yes|no>\n"
    "An access event that applies sets the access record of its service and resource, creating it if needed, so a "
    "withdrawal that applies writes allowed=no with its source. ADD and REMOVE set the member record, GROUP_GRANT adds a "
    "group_grant record, and REQ_WINDOW, OPEN and CLOSE set the window record of the resource (a new window record "
    "starts with requires_window=no and open=no before the event is applied). "
    "Keep a record once it exists, also when it says allowed=no or member=no. Output the complete state after the "
    "update, every record changed or not, between a line STATE and a line END, and nothing else.")
FMT_D2 = (
    "State format, one record per shipment, sorted:\n"
    "shipment=<id> clearance=<none|no_limit|through:N> remembered=<none|no_limit|through:N> "
    "payment=<confirmed|not_confirmed> holds=<none|compliance|floor|compliance+floor>\n"
    "A shipment has a record once any update has touched it, starting from clearance=none remembered=none "
    "payment=not_confirmed holds=none. Output the complete state after the update, every record "
    "changed or not, between a line STATE and a line END, and nothing else.")


# ------------------------------------------------------------------ streams
def streams_d1(n=None):
    out = []
    for i in range(N1 if n is None else n):
        for s in make_pair_v2(BASE1 + T * 10000 + i, T):
            s["rev"] = replay_events(s)
            s["key"] = f"{s['pair']}-{s['truth']}"
            out.append(s)
    return out


def episodes_d2(n=None):
    import r5_domain2 as R
    import domain2 as D
    R.use_templates()
    out = []
    for i in range(N2 if n is None else n):
        seed = BASE2 + T * 10000 + i
        for e in D.make_pair(seed, T):
            e["pair"] = seed
            e["key"] = f"{seed}-{e['truth']}"
            out.append(e)
    return out


def truth_d1(s):
    return replay([e for e in s["rev"] if e["t"] < T], s["p"], s["r"], T - 1)


def dep_idx_d1(s):
    from r5_comparators import dependency_lines
    keep = set(dependency_lines(s, T))
    return [i for i, l in enumerate(s["lines"][:T]) if l in keep]


def dep_idx_d2(e):
    return [i for i, l in enumerate(e["lines"]) if re.search(rf"\b{re.escape(e['entity'])}\b", l)]


# ------------------------------------------------------------------ domain 1: canonical state and the exact rule
TYPEMAP = {"GRANT_UNTIL": "GRANT", "ALARM_WITHDRAW": "WITHDRAW", "MEMBER": "ADD"}


def true_updates_d1(s):
    """[(event number, (type, src, p, r, g, valid_through))] for the non-null true events before the query."""
    out = []
    for ev in s["rev"]:
        if ev["t"] >= T or ev["type"] == "FILLER":
            continue
        k = TYPEMAP.get(ev["type"], ev["type"])
        vt = ev.get("until")
        out.append((ev["t"] + 1, (k, ev.get("src"), ev.get("p"), ev.get("r"), ev.get("g"), None if vt is None else vt + 1)))
    return out


def parsed_updates_d1(s, P):
    from x6_parse_filter import content, to_event
    out = []
    for i, l in enumerate(s["lines"][:T]):
        ev = to_event(P[content(l)], i)
        if ev["type"] == "FILLER":
            continue
        vt = ev.get("until")
        out.append((i + 1, (TYPEMAP.get(ev["type"], ev["type"]), ev.get("src"), ev.get("p"), ev.get("r"), ev.get("g"),
                            None if vt is None else vt + 1)))
    return out


def step1(st, n, u):
    """The exact rule on the canonical state. st: component -> value, n: event number (1-based)."""
    st = dict(st)
    k, src, p, r, g, vt = u
    if k in ("GRANT", "WITHDRAW", "RECONFIRM"):
        cur = st.get(("access", p, r))
        if cur is not None and cur[2] is not None and cur[2] < n:
            cur = None                                   # an expired grant holds no rank
        if cur is None or RANK[src] >= RANK[cur[1]]:
            st[("access", p, r)] = (k != "WITHDRAW", src, vt if k == "GRANT" else None)
    elif k == "GROUP_GRANT":
        st[("gg", g, r)] = True
    elif k in ("ADD", "REMOVE"):
        st[("member", p, g)] = k == "ADD"
    elif k in ("REQ_WINDOW", "OPEN", "CLOSE"):
        req, op = st.get(("window", r), (False, False))
        st[("window", r)] = (True, op) if k == "REQ_WINDOW" else (req, k == "OPEN")
    return st


def run1(ups):
    st = {}
    for n, u in ups:
        st = step1(st, n, u)
    return st


def answer1(st, p, r):
    a = st.get(("access", p, r))
    perm = a is not None and a[0] and (a[2] is None or a[2] >= T)
    perm = perm or any(st.get(("member", p, g)) and st.get(("gg", g, r)) for g in GROUPS)
    w = st.get(("window", r))
    if w and w[0] and not w[1]:
        perm = False
    return "APPROVE" if perm else "DENY"


yn = lambda b: "yes" if b else "no"


def lines1(st):
    """Canonical record lines (unresolved), as the writer reads and writes them."""
    out = []
    for c in sorted(st, key=lambda c: (("access", "member", "gg", "window").index(c[0]),) + c[1:]):
        v = st[c]
        if c[0] == "access":
            out.append(f"access service={c[1]} resource={c[2]} allowed={yn(v[0])} source={v[1]} "
                       f"valid_through={'none' if v[2] is None else f'Event {v[2]:03d}'}")
        elif c[0] == "member":
            out.append(f"member service={c[1]} group={c[2]} member={yn(v)}")
        elif c[0] == "gg":
            out.append(f"group_grant group={c[1]} resource={c[2]}")
        else:
            out.append(f"window resource={c[1]} requires_window={yn(v[0])} open={yn(v[1])}")
    return out


R_ACC = re.compile(r"access\s+service=(\S+)\s+resource=(\S+)\s+allowed=(yes|no)\s+source=(\S+)\s+valid_through=(?:Event\s*)?(\d+|none)", re.I)
R_MEM = re.compile(r"member\s+service=(\S+)\s+group=(\S+)\s+member=(yes|no)", re.I)
R_GG = re.compile(r"group_grant\s+group=(\S+)\s+resource=(\S+)", re.I)
R_WIN = re.compile(r"window\s+resource=(\S+)\s+requires_window=(yes|no)\s+open=(yes|no)", re.I)


def parse1(text):
    """Writer output -> canonical state, or None if no record parses."""
    st = {}
    for line in text.splitlines():
        line = line.strip().lstrip("-* ").strip()
        m = R_ACC.match(line)
        if m and m.group(4) in RANK:
            st[("access", m.group(1), m.group(2))] = (m.group(3).lower() == "yes", m.group(4),
                                                     None if m.group(5).lower() == "none" else int(m.group(5)))
            continue
        m = R_MEM.match(line)
        if m:
            st[("member", m.group(1), m.group(2))] = m.group(3).lower() == "yes"
            continue
        m = R_GG.match(line)
        if m:
            st[("gg", m.group(1), m.group(2))] = True
            continue
        m = R_WIN.match(line)
        if m:
            st[("window", m.group(1))] = (m.group(2).lower() == "yes", m.group(3).lower() == "yes")
    return st or None


def typed1(n, u):
    k, src, p, r, g, vt = u
    parts = [f"type={k}"] + [f"{a}={b}" for a, b in (("source", src), ("service", p), ("resource", r), ("group", g),
                                                    ("valid_through", None if vt is None else f"Event {vt:03d}")) if b is not None]
    return f"Event {n:03d}: " + " ".join(parts)


def query_rows1(st, p, r, typed):
    """Renderer v2 on the canonical state at the query, then the dependency selection, then explicit absence."""
    from r5_comparators import dependency_lines
    lines = []
    for c in sorted(st, key=lambda c: (("access", "member", "gg", "window").index(c[0]),) + c[1:]):
        v = st[c]
        if c[0] == "access":
            exp = v[0] and v[2] is not None and v[2] < T
            if typed:
                lines.append(f"access service={c[1]} resource={c[2]} allowed={yn(v[0] and not exp)} source={v[1]} "
                             f"valid_through={'none' if v[2] is None else f'Event {v[2]:03d}'} expired={yn(exp)}")
            else:
                what = (f"write access expired after Event {v[2]:03d}" if exp else
                        "write access granted" + (f", valid through Event {v[2]:03d}" if v[2] is not None else "") if v[0]
                        else "write access withdrawn")
                lines.append(f"State: {c[1]} on {c[2]}: {what} (decided by {v[1]}).")
        elif c[0] == "member":
            lines.append(f"member service={c[1]} group={c[2]} member={yn(v)}" if typed else
                         f"State: {c[1]} is {'a member' if v else 'not a member'} of group {c[2]}.")
        elif c[0] == "gg":
            lines.append(f"group_grant group={c[1]} resource={c[2]}" if typed else f"State: group {c[1]} holds write access to {c[2]}.")
        else:
            req, op = v
            if typed:
                lines.append(f"window resource={c[1]} requires_window={yn(req)} open={yn(op)}")
            elif req:
                lines.append(f"State: {c[1]} accepts writes only while its change window is open; the window is {'open' if op else 'closed'}.")
            else:
                lines.append(f"State: {c[1]} does not require a change window.")
    rows = dependency_lines({"p": p, "r": r, "lines": lines}, len(lines))
    if ("access", p, r) not in st:
        rows.append(f"access service={p} resource={r} record=none" if typed else
                    f"State: no grant or withdrawal is recorded for {p} on {r}.")
    for g in GROUPS:
        if st.get(("gg", g, r)) and ("member", p, g) not in st:
            rows.append(f"member service={p} group={g} member=no" if typed else f"State: {p} is not a member of group {g}.")
    if ("window", r) not in st:
        rows.append(f"window resource={r} requires_window=no open=no" if typed else f"State: {r} does not require a change window.")
    return rows


def query_comps1(st, p, r):
    """The queried components' resolved values (for field exactness)."""
    a = st.get(("access", p, r))
    acc = None if a is None else (a[0] and (a[2] is None or a[2] >= T), a[1])
    return (acc, tuple(bool(st.get(("member", p, g))) for g in GROUPS), tuple(bool(st.get(("gg", g, r))) for g in GROUPS),
            tuple(st.get(("window", r), (False, False))))


# ------------------------------------------------------------------ domain 2: canonical state and the exact rule
def true_updates_d2(e):
    import domain2 as D
    return [(ev["t"], D.to_update(ev)) for ev in sorted(e["events"], key=lambda x: x["t"]) if ev["type"] != "note" and ev["t"] <= T]


def parsed_updates_d2(e, P):
    import r5_domain2 as R
    return [(u["event"], u) for u in R.parsed_updates(e, P) if u.get("op") not in (None, "no_change") and u.get("shipment")]


def step2(st, n, u):
    import domain2 as D
    try:
        return D.apply_update(st, dict(u, event=n))
    except Exception:
        return dict(st, last_event=n)


def run2(ups):
    import domain2 as D
    st = D.new_state()
    for n, u in ups:
        st = step2(st, n, u)
    return st


def lim(c):
    return "none" if c is None else ("no_limit" if c["through"] is None else f"through:{c['through']}")


def lines2(st):
    return [f"shipment={s} clearance={lim(v['clearance'])} remembered={lim(v['remembered'])} "
            f"payment={'confirmed' if v['paid'] else 'not_confirmed'} holds={'+'.join(v['holds']) or 'none'}"
            for s, v in sorted(st["shipments"].items())]


R_SH = re.compile(r"shipment=(\S+)\s+clearance=(none|no_limit|through:\d+)\s+remembered=(none|no_limit|through:\d+)\s+"
                  r"payment=(confirmed|not_confirmed)\s+holds=(\S+)", re.I)


def parse2(text, last_event):
    def L(x):
        x = x.lower()
        return None if x == "none" else {"through": None} if x == "no_limit" else {"through": int(x.split(":")[1])}
    sh = {}
    for line in text.splitlines():
        m = R_SH.search(line)
        if not m:
            continue
        h = m.group(5).lower()
        holds = [] if h == "none" else sorted({x for x in h.split("+") if x in ("compliance", "floor")})
        sh[m.group(1)] = {"clearance": L(m.group(2)), "remembered": L(m.group(3)), "paid": m.group(4).lower() == "confirmed",
                          "holds": holds}
    return {"last_event": last_event, "shipments": sh} if sh else None


def typed2(n, u):
    parts = [f"{k}={u[k]}" for k in ("op", "shipment", "source", "through") if u.get(k) is not None]
    return f"Event {n:03d}: " + " ".join(parts)


def query_rows2(st, s, typed):
    from r2c2_resolved import render_state_resolved
    st = dict(st, last_event=T)
    v = st["shipments"].get(s, {"clearance": None, "remembered": None, "paid": False, "holds": []})
    if typed:
        c = v["clearance"]
        cl = "none" if c is None else ("in_force" if c["through"] is None or c["through"] >= T else "expired")
        lm = "none" if c is None or c["through"] is None else "Event %03d" % c["through"]
        return [f"shipment={s} clearance={cl} limit={lm} "
                f"remembered={lim(v['remembered'])} payment={'confirmed' if v['paid'] else 'not_confirmed'} "
                f"holds={'+'.join(v['holds']) or 'none'}"]
    one = {"last_event": T, "shipments": {s: v}}
    return [l for l in render_state_resolved(one) if re.search(rf"\b{re.escape(s)}\b", l)]


def query_comps2(st, s):
    v = st["shipments"].get(s, {"clearance": None, "remembered": None, "paid": False, "holds": []})
    c = v["clearance"]
    return (c is not None and (c["through"] is None or c["through"] >= T), v["paid"], tuple(v["holds"]),
            None if v["remembered"] is None else v["remembered"]["through"])


def answer2(st, s):
    import domain2 as D
    return "YES" if D.may_ship(dict(st, last_event=T), s) else "NO"


# ------------------------------------------------------------------ models
def snapshot(name):
    import glob
    org, n = name.split("/")
    for h in (os.path.expanduser("~/.cache/huggingface/hub"),):
        s = sorted(glob.glob(f"{h}/models--{org}--{n}/snapshots/*"))
        if s:
            return s[0]
    raise FileNotFoundError(name)


def vllm(reader, max_len=8192):
    from vllm import LLM
    from x3_recursive_memory import READERS as RP
    return LLM(model=RP[reader][0], max_model_len=max_len, seed=0, enforce_eager=True, enable_prefix_caching=True,
               gpu_memory_utilization=float(os.environ.get("VLLM_UTIL", "0.85")))


def sys_reader(dom):
    from x3_recursive_memory import PERSONAS
    from r2c_domain2 import READ_SYS
    return (PERSONAS["gov"] + "\n\n" + POLICY_V2 + OVERRIDE_RULE_V2 + "\n\n" + GLOSS_D1) if dom == 1 else (READ_SYS + "\n\n" + GLOSS_D2)


def prompt_parts(tok, row, instr=None):
    """(prompt before any generated text, decision cue). Every arm of a domain has the same system prompt."""
    from x3_recursive_memory import chat
    if row["domain"] == 1:
        user = ("Based strictly on the memory below, evaluate the incoming request.\n\n"
                f"[Memory]\n{row['memory'] or '(empty)'}\n\n[Request]\n{row['query']}")
        cue = "DECISION:"
    else:
        tag = {"A": "[Log]", "Dp": "[Typed updates]"}.get(row["arm"], "[Current state]")
        user = f"{tag}\n{row['memory']}\n\n[Question]\n{row['query']} Answer YES or NO."
        cue = "ANSWER:"
    if instr:
        user += "\n\n" + instr
    return chat(tok, sys_reader(row["domain"]), user), cue


def clip(text):
    m = re.search(r"\b(DECISION|ANSWER)\s*:", text)
    return (text[:m.start()] if m else text).rstrip(" *#_\n\t")


def read_items(reader, items, out_path):
    """items: [(row dict, prompt with cue)] -> rows with margins and decisions, written to out_path."""
    from x3_recursive_memory import load_reader, margins_batch, ntok
    from r2c_domain2 import yes_no_margin
    tok, model, _ = load_reader(reader, "cuda:0")
    w_head = model.lm_head.weight.detach().float()
    bs = int(os.environ.get("R8_READ_BS", "8"))
    out = []
    for dom in (1, 2):
        its = [x for x in items if x[0]["domain"] == dom]
        if not its:
            continue
        if dom == 1:
            ms = margins_batch(tok, model, w_head, {k: tok.encode(v, add_special_tokens=False) for k, v in
                                                    (("APPROVE", " APPROVE"), ("DENY", " DENY"))}, [x[1] for x in its], "cuda:0", bs=bs)
        else:
            ms = yes_no_margin(tok, model, w_head, [x[1] for x in its], "cuda:0", bs=bs)
        for (r, p), m in zip(its, ms):
            neg, pos = ("DENY", "APPROVE") if dom == 1 else ("NO", "YES")
            o = {k: v for k, v in r.items() if k not in ("memory", "query", "block")}
            o.update(margin=m, decision=neg if m > 0 else pos, memory_tokens=ntok(tok, r.get("memory") or ""),
                     prompt_tokens=ntok(tok, p))
            out.append(o)
    with open(out_path, "w") as f:
        for o in out:
            f.write(json.dumps(o) + "\n")
    print(reader, "read", len(out), "max prompt tokens", max(o["prompt_tokens"] for o in out), flush=True)


# ------------------------------------------------------------------ R8-1
def phase_parse():
    from vllm import SamplingParams
    from r2c_domain2 import PARSE_SYS as PS2, few_shot
    from x6_parse_filter import PARSE_SYS as PS1, EXAMPLES, content
    llm = vllm("qwen2.5-14b", 4096)
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0.0, max_tokens=96)
    chat = lambda sys_, user: tok.apply_chat_template([{"role": "system", "content": sys_}, {"role": "user", "content": user}],
                                                      tokenize=False, add_generation_prompt=True)
    shots1 = "\n\n".join(f"Line: {a}\nJSON: {json.dumps(b)}" for a, b in EXAMPLES)
    u1 = sorted({content(l) for s in streams_d1() for l in s["lines"]})
    shots2 = few_shot()
    u2 = sorted({l.split(": ", 1)[1] for e in episodes_d2() for l in e["lines"]})
    for name, uniq, prompts in (("d1", u1, [chat(PS1, shots1 + f"\n\nLine: {c}\nJSON:") for c in u1]),
                                ("d2", u2, [chat(PS2, shots2 + f"\n\nLine: Event 001: {c}\nJSON:") for c in u2])):
        res = llm.generate(prompts, sp, use_tqdm=False)
        with open(os.path.join(HERE, PFX + f"r8_parse_{name}.jsonl"), "w") as f:
            for c, r in zip(uniq, res):
                m = re.search(r"\{.*\}", r.outputs[0].text, re.S)
                try:
                    u = json.loads(m.group(0)) if m else None
                except json.JSONDecodeError:
                    u = None
                f.write(json.dumps({"content": c, "parsed": u, "prompt_tokens": len(r.prompt_token_ids),
                                    "gen_tokens": len(r.outputs[0].token_ids)}) + "\n")
        print(name, "parsed", len(uniq), flush=True)


def phase_build():
    from x6_parse_filter import content, parsed_tuple, query_at
    P1 = {r["content"]: parsed_tuple(r["parsed"]) for r in map(json.loads, open(os.path.join(HERE, PFX + "r8_parse_d1.jsonl")))}
    P2 = {r["content"]: r["parsed"] for r in map(json.loads, open(os.path.join(HERE, PFX + "r8_parse_d2.jsonl")))}
    rows, out = [], []
    ok_rule = ok_dep = n = 0
    for s in streams_d1():
        truth = truth_d1(s)
        ok_rule += answer1(run1(true_updates_d1(s)), s["p"], s["r"]) == truth
        idx = dep_idx_d1(s)
        ok_dep += replay([e for e in s["rev"] if e["t"] < T and e["t"] in set(idx)], s["p"], s["r"], T - 1) == truth
        st = run1(parsed_updates_d1(s, P1))
        n += 1
        base = {"domain": 1, "pair": s["pair"], "stream": s["key"], "family": s.get("family"), "truth": truth,
                "query": query_at(s, T), "n_dep": len(idx)}
        rows.append(dict(base, arm="A", memory="\n".join(s["lines"][i] for i in idx)))
        from r7_n1 import typed_d1
        rows.append(dict(base, arm="Dp", memory="\n".join(typed_d1(i, P1[content(s["lines"][i])]) for i in idx)))
        rows.append(dict(base, arm="Enat", memory="\n".join(query_rows1(st, s["p"], s["r"], False))))
        rows.append(dict(base, arm="Etyp", memory="\n".join(query_rows1(st, s["p"], s["r"], True))))
    out.append(f"domain 1: {n} streams; the canonical rule on the true events gives the replay answer in {ok_rule}; "
               f"replaying the dependency lines alone is exact in {ok_dep}")
    ok_rule = ok_dep = n = 0
    import domain2 as D
    from r7_n1 import typed_d2
    for e in episodes_d2():
        ok_rule += answer2(run2(true_updates_d2(e)), e["entity"]) == e["truth"]
        idx = dep_idx_d2(e)
        ok_dep += ("YES" if D.may_ship(D.current_state([ev for ev in e["events"] if ev["t"] - 1 in set(idx)], T), e["entity"])
                   else "NO") == e["truth"]
        st = run2(parsed_updates_d2(e, P2))
        n += 1
        base = {"domain": 2, "pair": e["pair"], "stream": e["key"], "family": e["family"], "truth": e["truth"],
                "query": e["query"], "n_dep": len(idx)}
        rows.append(dict(base, arm="A", memory="\n".join(e["lines"][i] for i in idx)))
        rows.append(dict(base, arm="Dp", memory="\n".join(typed_d2(i, P2.get(e["lines"][i].split(": ", 1)[1])) for i in idx)))
        rows.append(dict(base, arm="Enat", memory="\n".join(query_rows2(st, e["entity"], False))))
        rows.append(dict(base, arm="Etyp", memory="\n".join(query_rows2(st, e["entity"], True))))
    out.append(f"domain 2: {n} streams; the canonical rule on the true events gives the replay answer in {ok_rule}; "
               f"replaying the shipment's lines alone is exact in {ok_dep}")
    with open(os.path.join(HERE, PFX + "r8_mem.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    open(os.path.join(HERE, PFX + "r8_build.out"), "w").write("\n".join(out) + "\n")
    print("\n".join(out))


def phase_gen(reader):
    from vllm import SamplingParams
    llm = vllm(reader)
    tok = llm.get_tokenizer()
    rows = [r for r in map(json.loads, open(os.path.join(HERE, PFX + "r8_mem.jsonl"))) if r["arm"] == "A"]
    prompts = [prompt_parts(tok, r, SUMMARY_D1 if r["domain"] == 1 else SUMMARY_D2)[0] for r in rows]
    res = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=GEN_MAX), use_tqdm=False)
    with open(os.path.join(HERE, PFX + f"r8_gen_{reader}.jsonl"), "w") as f:
        for r, x in zip(rows, res):
            f.write(json.dumps({"domain": r["domain"], "stream": r["stream"], "arm": "C", "text": x.outputs[0].text,
                                "gen_tokens": len(x.outputs[0].token_ids), "prompt_tokens": len(x.prompt_token_ids),
                                "finish": x.outputs[0].finish_reason}) + "\n")
    print(reader, "generated", len(rows), flush=True)


def phase_read(reader):
    from transformers import AutoTokenizer
    from x3_recursive_memory import READERS as RP
    tok = AutoTokenizer.from_pretrained(RP[reader][0], local_files_only=True)
    mem = [json.loads(l) for l in open(os.path.join(HERE, PFX + "r8_mem.jsonl"))]
    gen = {(g["domain"], g["stream"]): g for g in map(json.loads, open(os.path.join(HERE, PFX + f"r8_gen_{reader}.jsonl")))}
    items = []
    for r in mem:
        p, cue = prompt_parts(tok, r)
        items.append((r, p + cue))
        if r["arm"] == "A":
            p, cue = prompt_parts(tok, r, SUMMARY_D1 if r["domain"] == 1 else SUMMARY_D2)
            items.append((dict(r, arm="C"), p + clip(gen[(r["domain"], r["stream"])]["text"]) + "\n" + cue))
    read_items(reader, items, os.path.join(HERE, PFX + f"r8_read_{reader}.jsonl"))


# ------------------------------------------------------------------ R8-1c, StateMem in its native read format (reuse)
SM_GUIDE_NATIVE = (
    "When answering, work through each relevant piece of state. Specifically:\n"
    " - Hard constraints (medical, scope-bound) take priority over recent soft preferences.\n"
    " - A unit flagged NEEDS RECHECK is STALE: an input it was derived\n"
    "   from has changed, so its baked-in value is now wrong. Do NOT use\n"
    "   that number. RECOMPUTE it: take its stated derivation (e.g.\n"
    "   'X = A * 2'), substitute the CURRENT value of each input from the\n"
    "   active state above, and evaluate. If the result feeds another\n"
    "   NEEDS RECHECK unit, recompute that one from the value you just\n"
    "   derived — propagate through the chain in order before answering.\n"
    "   Only recompute units on the path to the question; leave others.\n"
    " - For cascade-coupled units, surface the downstream implication explicitly.")


def native_block(units, keep=None):
    """StateMem App. E.3 layout. units: FullStore.units; keep: unit ids to show (None = all)."""
    a = [(k, v) for k, v in units.items() if v["status"] != "superseded" and (keep is None or k in keep)]
    L = lambda v: f"- [{v['type'] or 'note'}] {v['content']}"
    conv = [L(v) for k, v in a if v["status"] == "active" and v["priority"] == "hard" and v["source_type"] != "policy"]
    pol = [L(v) for k, v in a if v["status"] == "active" and v["source_type"] == "policy"]
    pref = [L(v) for k, v in a if v["status"] == "active" and v["priority"] == "soft" and v["source_type"] != "policy"]
    rec = [L(v) + f"\n -> trigger: {v['recheck']}" for k, v in a if v["status"] == "needs_recheck"]
    return "\n".join(["## BINDING — THIS CONVERSATION"] + conv + ["## BINDING — POLICY RULES"] + pol +
                     ["## PREFERENCES (yield if necessary)"] + pref + ["## ! NEEDS RECHECK — may be stale due to recent events"] + rec)


def project(units, p, r):
    """Units that name the service or the resource, or a group named together with either; policy units always."""
    has = lambda txt, x: re.search(rf"\b{re.escape(x)}\b", txt) is not None
    act = {k: v for k, v in units.items() if v["status"] != "superseded"}
    gs = {g for v in act.values() for g in GROUPS if g in v["content"] and (has(v["content"], p) or has(v["content"], r))}
    return {k for k, v in act.items() if v["source_type"] == "policy" or has(v["content"], p) or has(v["content"], r)
            or any(g in v["content"] for g in gs)}


def phase_sm_build():
    """Rebuild the R7-2 stores from their logged encoder outputs and check them against the stored blocks."""
    import r7_n2
    pol = json.load(open(os.path.join(HERE, "r7_n2_policy_units.json")))["units"]
    stores = {}
    for rec in map(json.loads, gzip.open(os.path.join(HERE, "r7_n2_events.jsonl.gz"), "rt")):
        st = stores.setdefault(rec["stream"], r7_n2.FullStore(pol))
        if rec["ok"]:
            st.apply(rec["out"], rec["t"])
    mem = {m["stream"]: m for m in map(json.loads, open(os.path.join(HERE, "r7_n2_mem.jsonl")))}
    import r7_n1
    ss = {s["key"]: s for s in r7_n1.streams_d1() if s["key"] in mem}
    from x6_parse_filter import query_at
    rows, same = [], 0
    for k, st in stores.items():
        same += st.block() == mem[k]["memory"]
        s = ss[k]
        for view, keep in (("whole", None), ("projected", project(st.units, s["p"], s["r"]))):
            rows.append({"domain": 1, "arm": f"sm_native_{view}", "pair": s["pair"], "stream": k, "family": s.get("family"),
                         "truth": mem[k]["truth"], "query": query_at(s, T), "block": native_block(st.units, keep)})
    with open(os.path.join(HERE, PFX + "r8_sm_mem.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    msg = f"R8-1c: {len(stores)} R7-2 stores rebuilt from the event log, identical to the stored block in {same}"
    open(os.path.join(HERE, PFX + "r8_sm_build.out"), "w").write(msg + "\n")
    print(msg)


def phase_sm_read(reader):
    from transformers import AutoTokenizer
    from x3_recursive_memory import PERSONAS, READERS as RP, chat
    tok = AutoTokenizer.from_pretrained(RP[reader][0], local_files_only=True)
    items = []
    for r in map(json.loads, open(os.path.join(HERE, PFX + "r8_sm_mem.jsonl"))):
        sys_ = PERSONAS["gov"] + "\n\n" + POLICY_V2 + "\n\n" + r["block"] + "\n\n" + SM_GUIDE_NATIVE
        user = "Evaluate the incoming request.\n\n[Request]\n" + r["query"]
        items.append((dict(r, memory=r["block"]), chat(tok, sys_, user) + "DECISION:"))
    read_items(reader, items, os.path.join(HERE, PFX + f"r8_sm_read_{reader}.jsonl"))


# ------------------------------------------------------------------ R8-2
def r82_streams(dom):
    if dom == 1:
        out = [(s["key"], s["pair"], true_updates_d1(s), s) for s in streams_d1(N_PAIRS2)]
    else:
        out = [(e["key"], e["pair"], true_updates_d2(e), e) for e in episodes_d2(N_PAIRS2)]
    return [(k, p, u[:SMOKE_STEPS], s) for k, p, u, s in out] if SMOKE_STEPS else out


def writer_sys(dom):
    from r2c_domain2 import SPEC
    if dom == 1:
        return ("You maintain the current access state of a governance agent. Apply each update to the state by this "
                "policy.\n\n" + POLICY_V2 + OVERRIDE_RULE_V2 + "\n\n" + GLOSS_D1 + "\n\n" + FMT_D1)
    return ("You maintain the current state of an export dock. Apply each update to the state by these rules.\n\n" + SPEC
            + "\n\n" + GLOSS_D2 + "\n\n" + FMT_D2)


def writer_prompt(tok, dom, prev_n, prior_lines, n, u):
    upd = typed1(n, u) if dom == 1 else typed2(n, u)
    user = (f"Current state after Event {prev_n:03d}:\nSTATE\n" + ("\n".join(prior_lines) or "(no records)") + "\nEND\n\n"
            f"Update:\n{upd}\n\nOutput the complete state after Event {n:03d}.")
    return tok.apply_chat_template([{"role": "system", "content": writer_sys(dom)}, {"role": "user", "content": user}],
                                   tokenize=False, add_generation_prompt=True)


def to_lines(dom, st):
    return lines1(st) if dom == 1 else lines2(st)


def last_block(text):
    """The text after the last line that reads STATE, up to END; the whole text if there is no STATE line."""
    lines = text.splitlines()
    idx = [i for i, l in enumerate(lines) if l.strip().strip("`*# ").upper() == "STATE"]
    if not idx:
        return text
    out = []
    for l in lines[idx[-1] + 1:]:
        if l.strip().strip("`*# ").upper() == "END":
            break
        out.append(l)
    return "\n".join(out)


def parse_out(dom, text, n):
    text = last_block(text)
    return parse1(text) if dom == 1 else parse2(text, n)


def gen_states(llm, tok, dom, jobs):
    """jobs: [(prompt, n)] -> [(text, retry text, parsed or None, tokens)], one retry with the STATE line prefilled."""
    from vllm import SamplingParams
    sp = SamplingParams(temperature=0.0, max_tokens=W_MAX)
    res = llm.generate([p for p, _ in jobs], sp, use_tqdm=False)
    out = []
    for (p, n), r in zip(jobs, res):
        out.append([r.outputs[0].text, None, parse_out(dom, r.outputs[0].text, n),
                    len(r.prompt_token_ids) + len(r.outputs[0].token_ids), r.outputs[0].finish_reason])
    bad = [i for i, o in enumerate(out) if o[2] is None]
    if bad:
        rr = llm.generate([jobs[i][0] + "STATE\n" for i in bad], sp, use_tqdm=False)
        for i, r in zip(bad, rr):
            out[i][1] = r.outputs[0].text
            out[i][2] = parse_out(dom, r.outputs[0].text, jobs[i][1])
            out[i][3] += len(r.prompt_token_ids) + len(r.outputs[0].token_ids)
    return out


def phase_one(reader, dom):
    """Teacher forced: every step from the rule's prior state."""
    llm = vllm(reader)
    tok = llm.get_tokenizer()
    step = step1 if dom == 1 else step2
    import domain2 as D
    meta, jobs = [], []
    for key, pair, ups, _ in r82_streams(dom):
        st, prev = ({} if dom == 1 else D.new_state()), 0
        for k, (n, u) in enumerate(ups):
            jobs.append((writer_prompt(tok, dom, prev, to_lines(dom, st), n, u), n))
            meta.append((key, k, n))
            st, prev = step(st, n, u), n
    res = gen_states(llm, tok, dom, jobs)
    with gzip.open(os.path.join(HERE, PFX + f"r8_one_{reader}_d{dom}.jsonl.gz"), "wt") as f:
        for (key, k, n), (t1, t2, o, ntk, fin) in zip(meta, res):
            f.write(json.dumps({"stream": key, "k": k, "n": n, "first": t1, "retry": t2, "ok": o is not None,
                                "state": None if o is None else to_lines(dom, o), "tokens": ntk, "finish": fin}) + "\n")
    print(reader, f"d{dom}", "ONE steps", len(jobs), "invalid after retry", sum(o is None for _, _, o, _, _ in res), flush=True)


def phase_free(reader, dom):
    """Recursive: every step from the writer's own previous state, from the empty state."""
    llm = vllm(reader)
    tok = llm.get_tokenizer()
    import domain2 as D
    S = r82_streams(dom)
    cur = {key: ({} if dom == 1 else D.new_state()) for key, _, _, _ in S}
    prev = {key: 0 for key, _, _, _ in S}
    f = gzip.open(os.path.join(HERE, PFX + f"r8_free_{reader}_d{dom}.jsonl.gz"), "wt")
    final, bad = {}, 0
    for k in range(max(len(ups) for _, _, ups, _ in S)):
        act = [(key, ups[k]) for key, _, ups, _ in S if k < len(ups)]
        jobs = [(writer_prompt(tok, dom, prev[key], to_lines(dom, cur[key]), n, u), n) for key, (n, u) in act]
        res = gen_states(llm, tok, dom, jobs)
        for (key, (n, u)), (t1, t2, o, ntk, fin) in zip(act, res):
            if o is None:
                bad += 1
                nxt = cur[key] if dom == 1 else dict(cur[key], last_event=n)
            else:
                nxt = o
            f.write(json.dumps({"stream": key, "k": k, "n": n, "first": t1, "retry": t2, "ok": o is not None,
                                "state": to_lines(dom, nxt), "tokens": ntk, "finish": fin}) + "\n")
            cur[key], prev[key] = nxt, n
        if k % 10 == 0:
            print(reader, f"d{dom}", "FREE step", k, flush=True)
    f.close()
    with open(os.path.join(HERE, PFX + f"r8_free_final_{reader}_d{dom}.jsonl"), "w") as g:
        for key in cur:
            g.write(json.dumps({"stream": key, "state": to_lines(dom, cur[key])}) + "\n")
    print(reader, f"d{dom}", "FREE done, invalid after retry", bad, flush=True)


def from_lines(dom, lines, n=T):
    o = parse_out(dom, "\n".join(lines), n)
    if o is None:
        import domain2 as D
        return {} if dom == 1 else D.new_state()
    return o


def comps(dom, st):
    """component -> value, for comparing states."""
    if dom == 1:
        return dict(st)
    return {(s, f): (lim(v[f]) if f in ("clearance", "remembered") else tuple(v[f]) if f == "holds" else v[f])
            for s, v in st["shipments"].items() for f in ("clearance", "remembered", "paid", "holds")}


def set_comp(dom, st, c, val):
    """Write one component of ONE's state into the rule's state (val None = absent)."""
    st = dict(st) if dom == 1 else {"last_event": st["last_event"], "shipments": {k: dict(v) for k, v in st["shipments"].items()}}
    if dom == 1:
        if val is None:
            st.pop(c, None)
        else:
            st[c] = val
        return st
    s, f = c
    rec = st["shipments"].setdefault(s, {"clearance": None, "remembered": None, "paid": False, "holds": []})
    if val is None:
        rec[f] = {"clearance": None, "remembered": None, "paid": False, "holds": []}[f]
    elif f in ("clearance", "remembered"):
        rec[f] = None if val == "none" else {"through": None} if val == "no_limit" else {"through": int(val.split(":")[1])}
    elif f == "holds":
        rec[f] = list(val)
    else:
        rec[f] = val
    return st


def touched(dom, u):
    if dom == 1:
        k, src, p, r, g, vt = u
        return {("access", p, r)} if k in ("GRANT", "WITHDRAW", "RECONFIRM") else {("gg", g, r)} if k == "GROUP_GRANT" else \
            {("member", p, g)} if k in ("ADD", "REMOVE") else {("window", r)}
    f = {"grant_clearance": ("clearance", "remembered"), "revoke_clearance": ("clearance", "remembered"),
         "reinstate_clearance": ("clearance", "remembered"), "confirm_payment": ("paid",), "reverse_payment": ("paid",),
         "place_hold": ("holds",), "lift_hold": ("holds",)}[u["op"]]
    return {(u["shipment"], x) for x in f}


def phase_inj():
    """CPU: ONE's single-step errors injected into the rule's run; one-step error statistics."""
    import domain2 as D
    out, stats = [], []
    for dom in (1, 2):
        S = r82_streams(dom)
        step = step1 if dom == 1 else step2
        for reader in READERS:
            one = defaultdict(dict)
            for rec in map(json.loads, gzip.open(os.path.join(HERE, PFX + f"r8_one_{reader}_d{dom}.jsonl.gz"), "rt")):
                one[rec["stream"]][rec["k"]] = rec
            n_touch = e_touch = n_other = e_other = n_inv = n_steps = 0
            for key, pair, ups, _ in S:
                rule = inj = {} if dom == 1 else D.new_state()
                for k, (n, u) in enumerate(ups):
                    rule, inj = step(rule, n, u), step(inj, n, u)
                    rec = one[key][k]
                    n_steps += 1
                    if not rec["ok"]:
                        n_inv += 1
                        continue                          # an invalid output keeps the prior state: not injected
                    o = comps(dom, from_lines(dom, rec["state"], n))
                    rc = comps(dom, rule)
                    tc = touched(dom, u)
                    for c in set(rc) | set(o):
                        if c in tc:
                            n_touch += 1
                            e_touch += rc.get(c) != o.get(c)
                        else:
                            n_other += 1
                            e_other += rc.get(c) != o.get(c)
                        if rc.get(c) != o.get(c):
                            inj = set_comp(dom, inj, c, o.get(c))
                out.append({"domain": dom, "writer": reader, "stream": key, "state": to_lines(dom, inj)})
            stats.append({"domain": dom, "writer": reader, "steps": n_steps, "invalid": n_inv, "touched": n_touch,
                          "touched_wrong": e_touch, "untouched": n_other, "untouched_changed": e_other})
    with open(os.path.join(HERE, PFX + "r8_inj_final.jsonl"), "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    json.dump(stats, open(os.path.join(HERE, PFX + "r8_one_stats.json"), "w"), indent=1)
    print(json.dumps(stats, indent=1))


def r82_rows(reader):
    """Memories of RULE, INJ and FREE for one writer, rendered by renderer v2 and selected."""
    from x6_parse_filter import query_at
    inj = {(r["domain"], r["stream"]): r["state"] for r in map(json.loads, open(os.path.join(HERE, PFX + "r8_inj_final.jsonl")))
           if r["writer"] == reader}
    rows = []
    for dom in (1, 2):
        free = {r["stream"]: r["state"] for r in map(json.loads, open(os.path.join(HERE, PFX + f"r8_free_final_{reader}_d{dom}.jsonl")))}
        for key, pair, ups, s in r82_streams(dom):
            rule = run1(ups) if dom == 1 else run2(ups)
            for arm, st in (("RULE", rule), ("INJ", from_lines(dom, inj[(dom, key)])), ("FREE", from_lines(dom, free[key]))):
                if dom == 1:
                    mem, qc, qr = query_rows1(st, s["p"], s["r"], False), query_comps1(st, s["p"], s["r"]), query_comps1(rule, s["p"], s["r"])
                    row = {"domain": 1, "truth": truth_d1(s), "family": s.get("family"), "query": query_at(s, T),
                           "state_answer": answer1(st, s["p"], s["r"])}
                else:
                    mem, qc, qr = query_rows2(st, s["entity"], False), query_comps2(st, s["entity"]), query_comps2(rule, s["entity"])
                    row = {"domain": 2, "truth": s["truth"], "family": s["family"], "query": s["query"],
                           "state_answer": answer2(st, s["entity"])}
                rows.append(dict(row, arm=arm, writer=reader, pair=pair, stream=key, memory="\n".join(mem), query_exact=qc == qr,
                                 n_components=len(comps(dom, st)), n_components_rule=len(comps(dom, rule))))
    return rows


def phase_read2(reader):
    from transformers import AutoTokenizer
    from x3_recursive_memory import READERS as RP
    tok = AutoTokenizer.from_pretrained(RP[reader][0], local_files_only=True)
    items = []
    for r in r82_rows(reader):
        p, cue = prompt_parts(tok, dict(r, arm="Enat"))
        items.append((r, p + cue))
    read_items(reader, items, os.path.join(HERE, PFX + f"r8_read2_{reader}.jsonl"))


# ------------------------------------------------------------------ analysis
def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    rej, m = [False] * len(ps), len(ps)
    for j, i in enumerate(order):
        if ps[i] <= 0.05 / (m - j):
            rej[i] = True
        else:
            break
    return rej


def boot_combo(rows, weights, n=2000, seed=0):
    """Pair bootstrap of sum_w w * BA(arm). rows carry pair, arm, truth, decision. (est, lo, hi, one-sided p)."""
    from r2_analyze import bal
    g = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["arm"] in weights:
            g[r["pair"]][r["arm"]].append(r)
    pairs = sorted(p for p in g if all(a in g[p] for a in weights))
    stat = lambda ps: sum(w * bal([r for p in ps for r in g[p][a]]) for a, w in weights.items())
    rng = random.Random(seed)
    ds = sorted(stat([rng.choice(pairs) for _ in pairs]) for _ in range(n))
    return stat(pairs), ds[int(0.025 * n)], ds[int(0.975 * n) - 1], sum(x <= 0 for x in ds) / n


def fmt(x):
    return f"{x[0]:+.2f} [{x[1]:.2f}, {x[2]:.2f}]"


def phase_analyze():
    from r2_analyze import bal
    out = [open(os.path.join(HERE, PFX + "r8_build.out")).read().strip()]
    # R8-1
    R = {rd: [json.loads(l) for l in open(os.path.join(HERE, PFX + f"r8_read_{rd}.jsonl"))] for rd in READERS}
    cells = [(rd, d) for d in (1, 2) for rd in READERS]
    for rd, d in cells:
        rs = [r for r in R[rd] if r["domain"] == d]
        out.append(f"R8-1 {rd} d{d}: " + ", ".join(f"{a} {bal([r for r in rs if r['arm'] == a]):.2f}" for a in ARMS1))
    out.append("H8-1 (resolution with format fixed): 1/2[(Enat - A) + (Etyp - Dp)] >= 0.05, lower limit > 0, Holm over 4")
    res = {c: boot_combo([r for r in R[c[0]] if r["domain"] == c[1]], {"Enat": .5, "A": -.5, "Etyp": .5, "Dp": -.5}) for c in cells}
    rej = holm([res[c][3] for c in cells])
    for c, h in zip(cells, rej):
        ok = h and res[c][0] >= 0.05 and res[c][1] > 0
        out.append(f"  {c[0]} d{c[1]}: {fmt(res[c])} p={res[c][3]:.4f} Holm reject {h} -> {'holds' if ok else 'does not hold'}")
    out.append("secondary, report only")
    for lab, w in (("format effect 1/2[(Dp - A) + (Etyp - Enat)]", {"Dp": .5, "A": -.5, "Etyp": .5, "Enat": -.5}),
                   ("interaction 1/2[(Etyp - Dp) - (Enat - A)]", {"Etyp": .5, "Dp": -.5, "Enat": -.5, "A": .5}),
                   ("Enat - C (two-sided)", {"Enat": 1, "C": -1}), ("Enat - A", {"Enat": 1, "A": -1}),
                   ("Etyp - Dp", {"Etyp": 1, "Dp": -1}), ("Dp - A", {"Dp": 1, "A": -1})):
        for rd, d in cells:
            out.append(f"  {lab} {rd} d{d}: {fmt(boot_combo([r for r in R[rd] if r['domain'] == d], w))}")
    for rd, d in cells:
        rs = [r for r in R[rd] if r["domain"] == d]
        for fam in sorted({r["family"] for r in rs}):
            out.append(f"  family {rd} d{d} {fam}: " + " ".join(f"{a} {bal([r for r in rs if r['arm'] == a and r['family'] == fam]):.2f}" for a in ARMS1))
    # R8-1c
    out.append("R8-1c StateMem native read format (reused R7-2 stores, report only); E = the R7-1 resolved rows on the same pairs")
    for rd in READERS:
        sm = [json.loads(l) for l in open(os.path.join(HERE, PFX + f"r8_sm_read_{rd}.jsonl"))]
        e7 = [dict(r, arm="E7") for r in map(json.loads, open(os.path.join(HERE, f"r7_n1_read_{rd}.jsonl"))) if r["domain"] == 1 and r["arm"] == "E"]
        pairs = {r["pair"] for r in sm}
        e7 = [r for r in e7 if r["pair"] in pairs]
        for v in ("whole", "projected"):
            x = [r for r in sm if r["arm"] == f"sm_native_{v}"]
            d = boot_combo(e7 + x, {"E7": 1, f"sm_native_{v}": -1})
            out.append(f"  {rd} {v}: BA {bal(x):.2f}, mean tokens {sum(r['memory_tokens'] for r in x) / len(x):.0f}, E - StateMem {fmt(d)}"
                       + (" -> within 0.10 of E" if d[0] < 0.10 else ""))
    # R8-2
    out.append("H8-2 (who executes the transition): BA(RULE) - BA(FREE) >= 0.10, lower limit > 0, Holm over 4")
    R2 = {rd: [json.loads(l) for l in open(os.path.join(HERE, PFX + f"r8_read2_{rd}.jsonl"))] for rd in READERS}
    res = {c: boot_combo([r for r in R2[c[0]] if r["domain"] == c[1]], {"RULE": 1, "FREE": -1}) for c in cells}
    rej = holm([res[c][3] for c in cells])
    for c, h in zip(cells, rej):
        rs = [r for r in R2[c[0]] if r["domain"] == c[1]]
        ok = h and res[c][0] >= 0.10 and res[c][1] > 0
        tot = res[c][0]
        si = boot_combo(rs, {"RULE": 1, "INJ": -1})
        rc = boot_combo(rs, {"INJ": 1, "FREE": -1})
        case = ("3, withdraw the update-map claim" if not ok else "1, recursion" if rc[0] >= tot / 2 else
                "2, single steps" if si[0] >= tot / 2 else "neither half")
        out.append(f"  {c[0]} d{c[1]}: BA RULE {bal([r for r in rs if r['arm'] == 'RULE']):.2f} INJ {bal([r for r in rs if r['arm'] == 'INJ']):.2f} "
                   f"FREE {bal([r for r in rs if r['arm'] == 'FREE']):.2f}; RULE - FREE {fmt(res[c])} p={res[c][3]:.4f} Holm reject {h} "
                   f"-> {'holds' if ok else 'does not hold'}; RULE - INJ {fmt(si)}, INJ - FREE {fmt(rc)}; case {case}")
        for arm in ("RULE", "INJ", "FREE"):
            x = [r for r in rs if r["arm"] == arm]
            out.append(f"    {arm}: queried components exact {sum(r['query_exact'] for r in x)}/{len(x)}, state answer correct "
                       f"{sum(r['state_answer'] == r['truth'] for r in x)}/{len(x)}, components {sum(r['n_components'] for r in x) / len(x):.1f} "
                       f"(rule {sum(r['n_components_rule'] for r in x) / len(x):.1f})")
    for s in json.load(open(os.path.join(HERE, PFX + "r8_one_stats.json"))):
        out.append(f"  ONE {s['writer']} d{s['domain']}: {s['steps']} steps, {s['invalid']} invalid after retry, touched components wrong "
                   f"{s['touched_wrong']}/{s['touched']} ({s['touched_wrong'] / max(1, s['touched']):.3f}), untouched changed "
                   f"{s['untouched_changed']}/{s['untouched']} ({s['untouched_changed'] / max(1, s['untouched']):.4f})")
    open(os.path.join(HERE, PFX + "r8_analyze.out"), "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True)
    ap.add_argument("--reader")
    ap.add_argument("--domain", type=int)
    a = ap.parse_args()
    {"parse": phase_parse, "build": phase_build, "gen": lambda: phase_gen(a.reader), "read": lambda: phase_read(a.reader),
     "sm_build": phase_sm_build, "sm_read": lambda: phase_sm_read(a.reader), "one": lambda: phase_one(a.reader, a.domain),
     "free": lambda: phase_free(a.reader, a.domain), "inj": phase_inj, "read2": lambda: phase_read2(a.reader),
     "analyze": phase_analyze}[a.phase]()
