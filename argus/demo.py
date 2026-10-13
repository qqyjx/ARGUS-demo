"""End-to-end demo: python -m argus.demo [--case KEY] [--semantics gr|pr|st]."""
from __future__ import annotations

import argparse

from .af import extensions, status
from .cases import CASES, Case
from .llm import MockLLM
from .ops import CostModel, EvidenceUpdate, Op
from .pipeline import ArgGraph, build_graph, evidence_update, repair_graph
from .verify import verify

COSTS = (CostModel("uniform"), CostModel("structure", w=2.0), CostModel("confidence"))


def _fmt(s) -> str:
    return "{" + ", ".join(sorted(s)) + "}"


def describe_op(o: Op, units: dict) -> str:
    c = lambda a: repr(units[a].claim)
    return {
        "add_arg": f"{o}: add claim {c(o.x)}",
        "del_arg": f"{o}: remove claim {c(o.x)}",
        "add_att": f"{o}: {c(o.x)} now attacks {c(o.y)}" if o.kind == "add_att" else "",
        "del_att": f"{o}: {c(o.x)} no longer attacks {c(o.y)}" if o.kind == "del_att" else "",
    }[o.kind]


def show_framework(g: ArgGraph, title: str) -> None:
    print(f"  {title}")
    for a in sorted(g.af.args):
        u = g.units[a]
        mark = " [target]" if a == g.target else ""
        print(f"    {a}{mark}: {u.claim!r}  (conf {u.confidence:.2f}, {u.origin})")
    print(f"    attacks : {', '.join(f'{x}->{y}' for x, y in sorted(g.af.attacks)) or '(none)'}")
    print(f"    supports: {', '.join(f'{x}=>{y}' for x, y in sorted(g.supports)) or '(none)'}  (not part of R)")


def show_acceptance(g: ArgGraph, sigma: str) -> None:
    exts = extensions(g.af, sigma)
    print(f"    [{sigma}] extensions: {', '.join(_fmt(e) for e in exts) or '(none)'}")
    print("    [%s] accepted: %s" % (sigma, _fmt(a for a in g.af.args if status(g.af, a, sigma) == "in") or "{}"))
    v = verify(g.af, g.target, sigma)
    print(f"    [{sigma}] verification of {g.target}: {v.verdict.upper()} -> "
          f"{'PASS' if v.passes else 'FAIL'}")
    if v.defense_sets:
        print(f"    [{sigma}] defense sets: {', '.join(_fmt(d) for d in v.defense_sets)}")
    elif v.attackers:
        print(f"    [{sigma}] attackers of {g.target}: {_fmt(v.attackers)}")


def run_case(case: Case, semantics: str | None = None) -> None:
    sigma = semantics or case.semantics
    llm = MockLLM(case.kb)
    print("=" * 78)
    print(f"{case.title}   [SYNTHETIC toy data]")
    print("=" * 78)
    print(f"  Q: {case.question}")
    print(f"  Explanation: {case.explanation}")
    g0 = build_graph(llm, case.question, case.answer, case.explanation)
    show_framework(g0, "Argumentation framework F0 (extracted from the explanation)")
    for s in ("gr", "pr"):
        show_acceptance(g0, s)
    g1, delta = g0, None
    if case.evidence:
        g1, delta = evidence_update(llm, g0, case.evidence)
        print(f"  Evidence: {' | '.join(case.evidence)}")
        print(f"  Update Delta: add args {_fmt(delta.args_add)}, add attacks "
              f"{', '.join(f'{x}->{y}' for x, y in sorted(delta.atts_add)) or '(none)'}")
        show_framework(g1, "Framework F1 after the update")
        for s in ("gr", "pr"):
            show_acceptance(g1, s)
    else:
        delta = EvidenceUpdate()
    print(f"  Minimal-change repair of {g0.target} to 'in' under {sigma} (exact search):")
    for cost in COSTS:
        out = repair_graph(llm, g1, delta, g0, cost, semantics=sigma)
        r = out.result
        out_units = {**g1.units, **out.pool_units}
        if not r.feasible:
            print(f"    - {cost.describe()}: unrepairable within budget")
            continue
        ops = ", ".join(str(o) for o in r.ops) or "(no edits needed)"
        print(f"    - {cost.describe()}: cost {r.cost:g}, edits {{{ops}}}; "
              f"{len(r.optimal_sets)} optimal set(s)")
        for o in r.ops:
            print(f"        {describe_op(o, out_units)}")
        if r.ops:
            changed = out.added or out.removed
            print("        repaired explanation: " + (" / ".join(out.explanation) if changed else "(claims unchanged)"))
    print()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="ARGUS toy demo (synthetic data, mock LLM)")
    ap.add_argument("--case", choices=[c.key for c in CASES], help="run one case only")
    ap.add_argument("--semantics", choices=("gr", "pr", "st"), help="semantics for the repair")
    args = ap.parse_args(argv)
    print("ARGUS demo: toy inputs are SYNTHETIC; the LLM is a deterministic mock.\n")
    for c in CASES:
        if args.case in (None, c.key):
            run_case(c, args.semantics)


if __name__ == "__main__":
    main()
