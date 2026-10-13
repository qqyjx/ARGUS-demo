"""Mock LLM, graph construction and the three toy cases end to end."""
import subprocess
import sys

from argus.cases import CASES
from argus.llm import MockLLM
from argus.ops import CostModel
from argus.pipeline import build_graph, evidence_update, repair_graph
from argus.verify import verify

BY_KEY = {c.key: c for c in CASES}


def graphs(key):
    c = BY_KEY[key]
    llm = MockLLM(c.kb)
    g0 = build_graph(llm, c.question, c.answer, c.explanation)
    g1, delta = evidence_update(llm, g0, c.evidence)
    return llm, g0, g1, delta


def test_mock_is_deterministic():
    c = BY_KEY["diagnosis"]
    a = build_graph(MockLLM(c.kb), c.question, c.answer, c.explanation)
    b = build_graph(MockLLM(c.kb), c.question, c.answer, c.explanation)
    assert a == b


def test_extraction_splits_because_and_records_support():
    _, g0, _, _ = graphs("diagnosis")
    assert g0.units["a1"].claim == "The patient most likely has lupus" and g0.target == "a1"
    assert ("a2", "a1") in g0.supports and ("a2", "a1") not in g0.af.attacks


def test_case1_matches_paper_style_running_example():
    _, g0, g1, delta = graphs("diagnosis")
    assert g0.af.attacks == frozenset({("a5", "a1"), ("a2", "a5")})
    assert verify(g0.af, "a1", "gr").passes
    assert delta.atts_add == frozenset({("e1", "a2")})
    v = verify(g1.af, "a1", "gr")
    assert v.verdict == "rejected" and v.attackers == frozenset({"a5"})


def test_case2_semantics_disagree():
    _, g0, _, _ = graphs("museum")
    assert verify(g0.af, g0.target, "gr").verdict == "undecided"
    assert verify(g0.af, g0.target, "pr").passes


def test_case3_consistent_and_vacuous_repair():
    llm, g0, g1, delta = graphs("capital")
    assert verify(g1.af, g1.target, "gr").passes
    out = repair_graph(llm, g1, delta, g0, CostModel("uniform"))
    assert out.result.cost == 0 and out.result.ops == ()


def test_case1_repairs_have_known_costs():
    llm, g0, g1, delta = graphs("diagnosis")
    uni = repair_graph(llm, g1, delta, g0, CostModel("uniform"))
    assert uni.result.cost == 1 and len(uni.result.optimal_sets) == 4
    st = repair_graph(llm, g1, delta, g0, CostModel("structure", w=3))
    assert st.result.cost == 2 and st.added == ("Anti-dsDNA antibodies are positive",)
    assert st.removed == ()
    assert verify(st.result.repaired, "a1", "gr").passes


def test_case2_grounded_repair_makes_target_accepted():
    llm, g0, g1, delta = graphs("museum")
    out = repair_graph(llm, g1, delta, g0, CostModel("structure", w=3), semantics="gr")
    assert out.result.cost == 2 and out.added == ("The website page is outdated",)
    assert verify(out.result.repaired, g0.target, "gr").passes


def test_demo_runs_and_labels_data_synthetic():
    r = subprocess.run([sys.executable, "-m", "argus.demo"], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0
    assert r.stdout.count("[SYNTHETIC toy data]") == 3
    assert "PASS" in r.stdout and "FAIL" in r.stdout
