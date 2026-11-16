#!/usr/bin/env python3
"""Final comparison table (blind review round 2, editor's request): every memory of the 256-event comparison on one
row, with accuracy on both domains, the paired difference to the dependency retriever and the resource ledger. All
cells are recomputed from the per-row reading files and r6_ledger.json; nothing is reread. Memories are shown uncut
(the 512-token cut of Table 2 is a separate condition). Pair bootstrap, 2000 resamples, seed 0, as everywhere.
Writes ../paper/generated/tab_final.tex and r7_final.out."""
import json, os

from fmt import r2
from r2_analyze import bal
from r5_comparators_analyze import boot

HERE = os.path.dirname(os.path.abspath(__file__))
RD = ("qwen2.5-14b", "phi-4")
J = lambda f: [json.loads(l) for l in open(os.path.join(HERE, f))]


def rows_d1(reader):
    x6 = J(f"x6_read_{reader}.jsonl")
    r5 = J(f"r5_comp_{reader}.jsonl")
    r6 = J(f"r6_read_{reader}.jsonl")
    f5 = lambda m: [r for r in r5 if r["method"] == m and r["t"] == 256 and r["B"] is None]
    f6 = lambda m: [r for r in r6 if r["method"] == m and r["t"] == 256 and r["B"] is None]
    return {"full log": None,
            "dependency retriever": f5("retriever"),
            "parsed filter, whole state": [r for r in x6 if r["design"] == "full" and r["method"] == "parse_filter:qwen2.5-14b"
                                           and r["t"] == 256 and r["B"] is None],
            "parsed filter, query rows": f6("filter_proj"),
            "StateMem-style, native": f5("statemem_native"),
            "StateMem-style, policy": f5("statemem_policy"),
            "Mem0, top-10 search": f5("mem0"),
            "Mem0, dependency selection": f6("mem0_dep")}


def rows_d2(reader):
    r5 = J(f"r5_d2_read_{reader}.jsonl")
    r6 = J(f"r6_d2_read_{reader}.jsonl")
    g = lambda rs, c: [r for r in rs if r["cond"] == c and r["T"] == 256]
    return {"full log": g(r5, "log"), "dependency retriever": g(r5, "retriever"), "parsed filter, whole state": g(r5, "parsed"),
            "parsed filter, query rows": g(r6, "filter_proj")}


STATUS = {"full log": "R5-4", "dependency retriever": "R5-3", "parsed filter, whole state": "X6", "parsed filter, query rows": "R6-2",
          "StateMem-style, native": "R5-3", "StateMem-style, policy": "R5-3", "Mem0, top-10 search": "R5-3",
          "Mem0, dependency selection": "R6-1"}
LEDGER = {"full log": "full log", "dependency retriever": "dependency retriever", "parsed filter, whole state": "parsed filter",
          "parsed filter, query rows": "parsed filter, query rows", "StateMem-style, native": "StateMem, native",
          "StateMem-style, policy": "StateMem, policy", "Mem0, top-10 search": "Mem0", "Mem0, dependency selection": "Mem0"}


def main():
    led = json.load(open(os.path.join(HERE, "r6_ledger.json")))["t256"]
    d1 = {rd: rows_d1(rd) for rd in RD}
    d2 = {rd: rows_d2(rd) for rd in RD}
    r6 = {rd: J(f"r6_read_{rd}.jsonl") for rd in RD}
    mem0_dep_read = {rd: sum(r["memory_tokens"] for r in r6[rd] if r["method"] == "mem0_dep" and r["t"] == 256 and r["B"] is None)
                     / 80 for rd in RD}
    num = lambda x: f"{x:,.0f}".replace(",", "{,}")
    sg = lambda x: ("+" if x >= 0 else "") + r2(x)
    diff = lambda d: f"${sg(d[0])}$ $[{r2(d[1])},{r2(d[2])}]$"
    out, L = [], [r"\begin{tabular}{@{}llrrllrrrrrr@{}}", r"\toprule",
                  r" & & \multicolumn{4}{c}{first domain, 256 events} & \multicolumn{2}{c}{second domain} & \multicolumn{4}{c}{resources per stream} \\",
                  r"\cmidrule(lr){3-6}\cmidrule(lr){7-8}\cmidrule(l){9-12}",
                  r"memory & row & Qwen & phi-4 & Qwen $-$ retr. & phi-4 $-$ retr. & Qwen & phi-4 & store kB & calls & write tokens & read tokens \\",
                  r"\midrule"]
    for name in STATUS:
        cells = []
        for rd in RD:
            rs = d1[rd][name]
            cells.append(r2(bal(rs)) if rs else "n.r.")
        for rd in RD:
            rs, ret = d1[rd][name], d1[rd]["dependency retriever"]
            if rs and name != "dependency retriever":
                d = boot(rs, ret)
                cells.append(diff(d))
                out.append(f"{rd} {name} - retriever (D1, 256, uncut): {d[0]:+.3f} [{d[1]:.3f}, {d[2]:.3f}] BA {bal(rs):.3f}")
            else:
                cells.append("")
        for rd in RD:
            rs = d2[rd].get(name)
            cells.append(r2(bal(rs)) if rs else "n.r.")
            if rs and name not in ("dependency retriever",):
                d = boot(rs, d2[rd]["dependency retriever"])
                out.append(f"{rd} {name} - retriever (D2, 256): {d[0]:+.3f} [{d[1]:.3f}, {d[2]:.3f}] BA {bal(rs):.3f}")
        lg = led[LEDGER[name]]
        w = num(lg["wtok"]) + (f" ({num(lg['wtok_cached'])})" if "wtok_cached" in lg else "")
        read = num(lg["read"]) if name != "Mem0, dependency selection" else num(mem0_dep_read["qwen2.5-14b"])   # Qwen tokens, as the ledger
        cells += [f"{lg['store'] / 1000:.1f}", num(lg["calls"]), w, read]
        L.append(f"{name} & {STATUS[name]} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    gen = os.path.join(HERE, "..", "paper", "generated")
    os.makedirs(gen, exist_ok=True)
    open(os.path.join(gen, "tab_final.tex"), "w").write("\n".join(L) + "\n")
    open(os.path.join(HERE, "r7_final.out"), "w").write("\n".join(out) + "\n")
    print("\n".join(out))
    print("\n".join(L))


if __name__ == "__main__":
    main()
