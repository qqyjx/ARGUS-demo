"""Minimal-change repair (Definition 4 and Algorithm 1 of the paper).

Exact search replaces the paper's ASP encoding: operation sets are enumerated
by increasing size and the search stops as soon as no larger set can be
cheaper (every operation costs at least `min_cost`). Returns every optimal
repair and one chosen by a fixed tie-break.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable

from .af import AF, status
from .ops import ADD_ARG, ADD_ATT, DEL_ARG, DEL_ATT, CostModel, EvidenceUpdate, Op, apply, apply_update

EPS = 1e-9


@dataclass(frozen=True)
class RepairProblem:
    """Inputs of the minimal-change repair problem.

    pool: candidate new arguments (add_arg candidates).
    candidate_attacks: attacks that may be added; None allows every ordered pair
    over the current arguments plus the pool (no self-attacks).
    k: if set, only operations inside the undirected k-neighbourhood of the
    target are considered. max_ops: search budget.
    """

    af: AF
    target: str
    desired: str = "in"                      # 's' in {in, out}
    semantics: str = "gr"
    delta: EvidenceUpdate = EvidenceUpdate()
    cost: CostModel = CostModel()
    pool: frozenset[str] = frozenset()
    candidate_attacks: frozenset[tuple[str, str]] | None = None
    k: int | None = None
    max_ops: int = 4


@dataclass(frozen=True)
class RepairResult:
    feasible: bool                           # False: unrepairable within budget
    ops: tuple[Op, ...]
    cost: float
    optimal_sets: tuple[tuple[Op, ...], ...]  # all minimum-cost valid repairs
    repaired: AF | None
    evaluated: int                           # operation sets checked


def is_valid(p: RepairProblem, ops: Iterable[Op]) -> bool:
    """A repair is valid iff the target exists in AF' and has the desired status."""
    try:
        af2 = apply(p.af, p.delta, tuple(ops))
    except ValueError:
        return False
    return p.target in af2.args and status(af2, p.target, p.semantics) == p.desired


def _neighbourhood(p: RepairProblem, af1: AF, cand_atts: set[tuple[str, str]]) -> set[str]:
    nodes = set(af1.args) | set(p.pool)
    edges = set(af1.attacks) | cand_atts
    seen, frontier = {p.target}, {p.target}
    for _ in range(p.k or 0):
        frontier = {y for x, y in edges if x in frontier} | {x for x, y in edges if y in frontier}
        frontier = (frontier & nodes) - seen
        seen |= frontier
    return seen


def candidate_ops(p: RepairProblem) -> list[Op]:
    """Edit operations the search may use, in a fixed order."""
    af1 = apply_update(p.af, p.delta)
    pool = set(p.pool) - af1.args
    universe = sorted(af1.args | pool)
    if p.candidate_attacks is None:
        cand = {(x, y) for x in universe for y in universe if x != y}
    else:
        cand = set(p.candidate_attacks)
    cand = {e for e in cand if e[0] in universe and e[1] in universe and e not in af1.attacks}
    keep = set(universe) if p.k is None else _neighbourhood(p, af1, cand)
    ops = [Op(DEL_ARG, a) for a in sorted(af1.args & keep) if a != p.target]
    ops += [Op(DEL_ATT, x, y) for x, y in sorted(af1.attacks) if x in keep and y in keep]
    ops += [Op(ADD_ARG, a) for a in sorted(pool & keep)]
    ops += [Op(ADD_ATT, x, y) for x, y in sorted(cand) if x in keep and y in keep]
    return ops


def _tie_key(ops: tuple[Op, ...]) -> tuple:
    """Prefer repairs that keep more structure: fewer argument deletions, then
    fewer deletions, then fewer operations, then lexicographic order."""
    return (
        sum(o.kind == DEL_ARG for o in ops),
        sum(o.is_deletion for o in ops),
        len(ops),
        sorted(str(o) for o in ops),
    )


def optimal_repair(p: RepairProblem) -> RepairResult:
    """Exact minimum-cost valid repair (or 'unrepairable within budget')."""
    ops_all = candidate_ops(p)
    min_c = min((p.cost.cost(o) for o in ops_all), default=1.0)
    best = float("inf")
    found: list[tuple[float, tuple[Op, ...]]] = []
    evaluated = 0
    for r in range(p.max_ops + 1):
        if r * min_c > best + EPS:  # no r-set can beat the best cost found
            break
        for combo in combinations(ops_all, r):
            evaluated += 1
            if not _structurally_ok(combo):
                continue
            if is_valid(p, combo):
                c = p.cost.total(combo)
                found.append((c, combo))
                best = min(best, c)
    if not found:
        return RepairResult(False, (), float("inf"), (), None, evaluated)
    optimal = sorted((o for c, o in found if c <= best + EPS), key=_tie_key)
    chosen = optimal[0]
    return RepairResult(True, chosen, best, tuple(optimal), apply(p.af, p.delta, chosen), evaluated)


def _structurally_ok(combo: tuple[Op, ...]) -> bool:
    """Drop sets with an attack edit on an argument the same set deletes."""
    deleted = {o.x for o in combo if o.kind == DEL_ARG}
    return not any(o.kind in (ADD_ATT, DEL_ATT) and (o.x in deleted or o.y in deleted) for o in combo)
