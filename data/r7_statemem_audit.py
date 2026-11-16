#!/usr/bin/env python3
"""StateMem format failures at the decision level, and an interface check of the adapter (blind review round 2, R2-3 /
R3-3). Report only, CPU, no model call, no rereading.

Failure audit. An event's update is lost when its first output and its one retry both fail to parse. The R5-3 failure
log (r5_statemem_failures.jsonl) keeps every retried event but only the last 300 characters of the retry, so the 63
native and 7 policy final failures cannot be read off it. We bound them instead:
  1. a retry whose logged text is complete (under 300 characters) and parses is proven valid;
  2. a retried event off the query's dependency lines (the retriever's selection rule) cannot change the true answer;
  3. a retried event that left a unit stamped with its own turn in the stored memory at 64 or 256 events was applied,
     since a failed event writes nothing.
What remains is the direct exposure of the deciding lines. It is not a bound on accuracy: the encoder sees every
active unit and decides every supersession, so a lost update on any line can change later updates of the deciding
units (fresh blind review, R1-M3; the replay R7-3 measures the effect).

Adapter check. Hand-written encoder outputs on short streams, fed to the R5-3 StateStore: adds, supersessions,
replacements, one-hop re-checks and the block layout must follow the outputs exactly; the 512-token cut keeps the head.
Writes r7_statemem_audit.out and r7_statemem_audit.json."""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r5_comparators as R  # noqa: E402


def audit(out):
    ss = {s["key"]: s for s in R.streams()}
    dep = {k: set(i for i, l in enumerate(s["lines"][:256]) if l in set(R.dependency_lines(s, 256))) for k, s in ss.items()}
    recs = [json.loads(l) for l in open(os.path.join(HERE, "r5_statemem_failures.jsonl"))]
    res = {}
    for v in ("native", "policy"):
        mem = {(m["stream"], m["t"]): m["memory"] for m in map(json.loads, open(os.path.join(HERE, f"r5_mem_statemem_{v}.jsonl")))}
        final = max(m["parse_failures"] for m in map(json.loads, open(os.path.join(HERE, f"r5_mem_statemem_{v}.jsonl"))))
        rv = [r for r in recs if r["variant"] == v]
        open_ = [r for r in rv if len(r["retry"]) >= 300 or R.parse_json("{" + r["retry"]) is None]
        on_dep = [r for r in open_ if r["t"] in dep[r["stream"]]]
        stamped = lambda r: f"(turn {r['t']})" in mem[(r["stream"], 256)] or (r["t"] < 64 and f"(turn {r['t']})" in mem[(r["stream"], 64)])
        possible = [r for r in on_dep if not stamped(r)]
        streams = sorted({r["stream"] for r in possible})
        res[v] = {"retried": len(rv), "final_failures": final, "not_proven_valid": len(open_), "on_dependency_lines": len(on_dep),
                  "proven_by_unit": len(on_dep) - len(possible), "possible_decisive": [(r["stream"], r["t"]) for r in possible],
                  "streams": streams}
        out.append(f"{v}: {len(rv)} retried events, {final} final failures; {len(open_)} retries not proven valid from the log, "
                   f"{len(on_dep)} of them on dependency lines, {len(on_dep) - len(possible)} applied (a unit with their turn survives), "
                   f"{len(possible)} possibly unapplied events on dependency lines, in {len(streams)} streams {streams} (direct exposure "
                   f"only, not a bound on accuracy)")
    return res


def adapter(out):
    def check(name, cond):
        out.append(f"{'PASS' if cond else 'FAIL'}  {name}")
        return cond

    ok = []
    st = R.StateStore()
    st.apply({"add": [{"content": "A grants p write on r.", "priority": "hard"}]}, 0)                         # U1
    st.apply({"add": [{"content": "p may write r today.", "priority": "soft", "derived_from": ["U1"]}]}, 1)    # U2 depends on U1
    st.apply({"add": [{"content": "unrelated replica note", "priority": "hard"}]}, 2)                         # U3
    ok.append(check("adds create active units with the given priority", [st.units[k]["priority"] for k in ("U1", "U2", "U3")] == ["hard", "soft", "hard"]))
    st.apply({"supersede": [{"unit_id": "U1", "reason": "withdrawn", "replacement": {"content": "S withdraws p on r.", "priority": "hard"}}]}, 3)
    ok.append(check("a supersession marks its target superseded", st.units["U1"]["status"] == "superseded"))
    ok.append(check("the replacement is added as a new active unit", st.units["U4"]["status"] == "active" and "withdraws" in st.units["U4"]["content"]))
    ok.append(check("a unit derived from the superseded one is flagged needs_recheck with the reason",
                    st.units["U2"]["status"] == "needs_recheck" and "U1" in st.units["U2"]["trigger"]))
    ok.append(check("an unrelated unit is untouched", st.units["U3"]["status"] == "active"))
    b = st.block()
    ok.append(check("the block omits superseded units", "A grants p write on r." not in b))
    ok.append(check("the block lists the re-check unit with its trigger", "p may write r today." in b.split("NEEDS RECHECK:")[1]))
    ok.append(check("the block puts binding units before preferences", b.index("S withdraws") < b.index("PREFERENCES")))
    st.apply({"supersede": [{"unit_id": "U9", "reason": "no such unit"}]}, 4)
    ok.append(check("a supersession of an unknown id changes nothing", len(st.units) == 4))
    st.apply({"supersede": [{"unit_id": "U1", "reason": "again"}]}, 5)
    ok.append(check("a second supersession of the same unit changes nothing", len(st.units) == 4))
    st.apply({"add": [], "supersede": []}, 6)
    ok.append(check("an empty update changes nothing", len(st.units) == 4))
    from transformers import AutoTokenizer
    from x3_recursive_memory import READERS, cut_tokens
    tok = AutoTokenizer.from_pretrained(READERS["qwen2.5-14b"][0], local_files_only=True)
    long = "\n".join(f"- [U{i}] (turn {i}) unit number {i} of a long block" for i in range(200))
    cut = cut_tokens(tok, long, 512)
    ok.append(check("the 512-token cut keeps the head of the block", long.startswith(cut) and "[U0]" in cut and "[U199]" not in cut))
    return {"checks": len(ok), "passed": sum(ok)}


def main():
    out = []
    res = {"audit": audit(out), "adapter": adapter(out)}
    open(os.path.join(HERE, "r7_statemem_audit.out"), "w").write("\n".join(out) + "\n")
    json.dump(res, open(os.path.join(HERE, "r7_statemem_audit.json"), "w"), indent=1)
    print("\n".join(out))


if __name__ == "__main__":
    main()
