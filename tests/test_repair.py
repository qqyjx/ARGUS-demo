"""Minimality of the repair, checked against an independent exhaustive search."""
import random
from itertools import combinations

from argus.af import AF, status
from argus.ops import CostModel, EvidenceUpdate, Op, apply
from argus.repair import RepairProblem, candidate_ops, is_valid, optimal_repair

# Running example: F0, then evidence a5 attacking a3 (so a0 reinstates and defeats a4).
F0 = AF.of(["a0", "a1", "a2", "a3", "a4"], [("a0", "a4"), ("a3", "a0")])
DELTA = EvidenceUpdate(frozenset({"a5"}), frozenset({("a5", "a3")}))
AUG = (Op("add_arg", "a6"), Op("add_att", "a6", "a5"))


def problem(cost, **kw):
    kw.setdefault("pool", frozenset({"a6"}))
    kw.setdefault("candidate_attacks", frozenset({("a6", "a5")}))
    return RepairProblem(F0, "a4", "in", "gr", DELTA, cost, **kw)


def brute_force_min(p, max_size=4):
    """Independent reference: try every subset of every candidate op, no pruning."""
    ops = candidate_ops(p)
    best, sets = float("inf"), []
    for r in range(max_size + 1):
        for c in combinations(ops, r):
            if is_valid(p, c):
                cost = p.cost.total(c)
                if cost < best - 1e-9:
                    best, sets = cost, [c]
                elif abs(cost - best) <= 1e-9:
                    sets.append(c)
    return best, sets


def test_update_breaks_target():
    assert status(apply(F0, DELTA, ()), "a4", "gr") == "out"
    assert status(F0, "a4", "gr") == "in"


def test_uniform_known_answer_cost_one():
    r = optimal_repair(problem(CostModel("uniform")))
    assert r.feasible and r.cost == 1
    assert {frozenset(s) for s in r.optimal_sets} == {
        frozenset({Op("del_arg", "a5")}), frozenset({Op("del_att", "a5", "a3")}),
        frozenset({Op("del_arg", "a0")}), frozenset({Op("del_att", "a0", "a4")}),
    }


def test_structure_preserving_w2_ties_and_prefers_augmentation():
    r = optimal_repair(problem(CostModel("structure", w=2)))
    assert r.cost == 2 and len(r.optimal_sets) == 5
    assert set(r.ops) == set(AUG)  # tie-break keeps structure


def test_structure_preserving_w3_unique_augmentation():
    r = optimal_repair(problem(CostModel("structure", w=3)))
    assert r.cost == 2 and [set(s) for s in r.optimal_sets] == [set(AUG)]
    assert status(r.repaired, "a4", "gr") == "in"
    assert {"a0", "a1", "a2", "a3", "a5"} <= r.repaired.args  # inclusion: nothing deleted


def test_confidence_weighted_prefers_cheap_deletion_of_low_confidence_claim():
    r = optimal_repair(problem(CostModel("confidence", confidence={"a5": 0.9, "a0": 0.4})))
    assert r.ops == (Op("del_arg", "a0"),) and abs(r.cost - 0.4) < 1e-9


def test_search_matches_exhaustive_reference_on_running_example():
    for cm in (CostModel("uniform"), CostModel("structure", w=2), CostModel("structure", w=3),
               CostModel("confidence", confidence={"a5": 0.9, "a0": 0.4, "a3": 0.75})):
        p = problem(cm)
        best, sets = brute_force_min(p)
        r = optimal_repair(p)
        assert abs(r.cost - best) < 1e-9
        assert {frozenset(s) for s in r.optimal_sets} == {frozenset(s) for s in sets}


def test_search_matches_exhaustive_reference_on_random_afs():
    rng = random.Random(0)
    names = ["t", "x", "y", "z"]
    for _ in range(40):
        atts = {(a, b) for a in names for b in names if rng.random() < 0.3}
        p = RepairProblem(AF.of(names, atts), "t", rng.choice(["in", "out"]), rng.choice(["gr", "pr"]),
                          cost=rng.choice([CostModel("uniform"), CostModel("structure", w=2.5)]), max_ops=3)
        best, _ = brute_force_min(p, 3)
        r = optimal_repair(p)
        assert r.feasible == (best < float("inf"))
        if r.feasible:
            assert abs(r.cost - best) < 1e-9


def test_repair_is_irredundant():
    r = optimal_repair(problem(CostModel("structure", w=3)))
    for i in range(len(r.ops)):
        assert not is_valid(problem(CostModel("uniform")), r.ops[:i] + r.ops[i + 1:])


def test_vacuity_and_inclusion_postulates():
    p = RepairProblem(F0, "a4", "in", "gr", EvidenceUpdate(), CostModel("uniform"))
    r = optimal_repair(p)
    assert r.ops == () and r.cost == 0 and r.repaired == F0
    p2 = problem(CostModel("uniform"))
    r2 = optimal_repair(p2)
    deleted = {o.x for o in r2.ops if o.kind == "del_arg"}
    assert r2.repaired.args >= (apply(F0, DELTA, ()).args - deleted)


def test_unrepairable_within_budget_and_budget_limits_optimum():
    r0 = optimal_repair(problem(CostModel("structure", w=3), max_ops=0))
    assert not r0.feasible and r0.repaired is None
    # With one operation the 2-op augmentation is out of budget; a deletion (cost 3) remains.
    r1 = optimal_repair(problem(CostModel("structure", w=3), max_ops=1))
    assert r1.feasible and r1.cost == 3 and len(r1.ops) == 1


def test_k_neighbourhood_restricts_search():
    chain = AF.of(["t", "x1", "x2", "x3"], [("x1", "t"), ("x2", "x1"), ("x3", "x2")])
    # t is out (x3 in, x2 out, x1 in).
    assert status(chain, "t", "gr") == "out"
    far = RepairProblem(chain, "t", "in", "gr", cost=CostModel("uniform"), k=0)
    assert not optimal_repair(far).feasible
    near = RepairProblem(chain, "t", "in", "gr", cost=CostModel("uniform"), k=1)
    assert optimal_repair(near).cost == 1
    full = RepairProblem(chain, "t", "in", "gr", cost=CostModel("uniform"))
    assert optimal_repair(full).cost == 1
