"""ARGUS pipeline glue: extraction, relation discovery, evidence, repair.

Sections 3.1-3.4 of the paper. The graph keeps, next to the AF, the claim,
confidence and support links of every argument so a repaired AF can be turned
back into a repaired explanation.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

from .af import AF
from .llm import CONTRADICTION, ENTAILMENT, LLMClient
from .ops import ADD_ARG, DEL_ARG, CostModel, EvidenceUpdate, Op, apply_update
from .repair import RepairProblem, RepairResult, optimal_repair


@dataclass(frozen=True)
class Unit:
    id: str
    claim: str
    confidence: float
    origin: str = "explanation"  # 'explanation' | 'evidence' | 'counterargument'


@dataclass(frozen=True)
class ArgGraph:
    """Argument graph G = (A, R) plus per-argument records and support links."""

    af: AF
    units: dict[str, Unit]
    supports: frozenset[tuple[str, str]]  # (premise, claim); not part of R
    target: str

    def claim(self, a: str) -> str:
        return self.units[a].claim


def _relations(llm: LLMClient, new: Sequence[Unit], old: Sequence[Unit]):
    """Attacks and supports for every ordered pair that involves a unit from `new`."""
    atts, sups = set(), set()
    allu = list(old) + list(new)
    new_ids = {u.id for u in new}
    for ui in allu:
        for uj in allu:
            if ui.id == uj.id or (ui.id not in new_ids and uj.id not in new_ids):
                continue
            rel = llm.classify(ui.claim, uj.claim)
            if rel == CONTRADICTION:
                atts.add((ui.id, uj.id))
            elif rel == ENTAILMENT:
                sups.add((ui.id, uj.id))
    return atts, sups


def build_graph(llm: LLMClient, question: str, answer: str, explanation: str) -> ArgGraph:
    """Structured extraction + relation discovery -> AF (Sections 3.1-3.2)."""
    ex = llm.extract_units(question, answer, explanation)
    units = [Unit(f"a{i + 1}", u.claim, u.confidence) for i, u in enumerate(ex.units)]
    atts, sups = _relations(llm, units, [])
    for i, u in enumerate(ex.units):
        sups |= {(units[j].id, units[i].id) for j in u.premises}
    return ArgGraph(
        AF.of((u.id for u in units), atts),
        {u.id: u for u in units},
        frozenset(sups),
        units[ex.target].id,
    )


def evidence_update(
    llm: LLMClient, graph: ArgGraph, evidence: Sequence[str]
) -> tuple[ArgGraph, EvidenceUpdate]:
    """Turn evidence statements into Delta = (A+, R+, {}, {}) and the updated graph."""
    new = []
    for i, text in enumerate(evidence):
        c = llm.extract_units("", "", text).units[0]
        new.append(Unit(f"e{i + 1}", c.claim, c.confidence, "evidence"))
    atts, sups = _relations(llm, new, list(graph.units.values()))
    delta = EvidenceUpdate(frozenset(u.id for u in new), frozenset(atts))
    units = {**graph.units, **{u.id: u for u in new}}
    return replace(graph, af=apply_update(graph.af, delta), units=units, supports=graph.supports | sups), delta


@dataclass(frozen=True)
class RepairOutcome:
    problem: RepairProblem
    result: RepairResult
    explanation: tuple[str, ...]  # repaired claims; empty if unrepairable
    added: tuple[str, ...]        # claims of arguments the repair added
    removed: tuple[str, ...]      # claims of arguments the repair deleted
    pool_units: dict[str, Unit]   # candidate counterargument arguments, by id


def repair_graph(
    llm: LLMClient,
    graph: ArgGraph,
    delta: EvidenceUpdate,
    graph_before: ArgGraph,
    cost: CostModel,
    semantics: str = "gr",
    k: int | None = None,
    max_ops: int = 4,
) -> RepairOutcome:
    """Minimal-change repair of the target to status 'in' (Section 3.4).

    Candidate arguments come from `llm.counter`; a candidate attack is admitted
    only if the relation classifier labels it a contradiction.
    """
    pool_units, seen = [], {u.claim.lower() for u in graph.units.values()}
    for u in list(graph.units.values()):
        for claim in llm.counter(u.claim, [x.claim for x in graph.units.values() if x.origin == "evidence"]):
            if claim.lower() not in seen:
                seen.add(claim.lower())
                pool_units.append(Unit(f"r{len(pool_units) + 1}", claim, 0.8, "counterargument"))
    atts, _ = _relations(llm, pool_units, list(graph.units.values()))
    conf = {u.id: u.confidence for u in graph.units.values()}
    prob = RepairProblem(
        af=graph_before.af,
        target=graph.target,
        desired="in",
        semantics=semantics,
        delta=delta,
        cost=CostModel(cost.name, cost.w, conf if cost.name == "confidence" else {}),
        pool=frozenset(u.id for u in pool_units),
        candidate_attacks=frozenset(atts),
        k=k,
        max_ops=max_ops,
    )
    res = optimal_repair(prob)
    pool = {u.id: u for u in pool_units}
    units = {**graph.units, **pool}
    if not res.feasible:
        return RepairOutcome(prob, res, (), (), (), pool)
    keep = sorted(res.repaired.args, key=lambda a: (units[a].origin == "counterargument", a))
    return RepairOutcome(
        prob,
        res,
        tuple(units[a].claim for a in keep),
        tuple(units[o.x].claim for o in res.ops if o.kind == ADD_ARG),
        tuple(units[o.x].claim for o in res.ops if o.kind == DEL_ARG),
        pool,
    )
