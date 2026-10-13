"""Semantics on small textbook frameworks (Dung 1995)."""
import pytest

from argus.af import AF, admissible, defense_sets, extensions, grounded, skeptically_accepted, status


def fs(*xs):
    return frozenset(xs)


def test_chain_grounded():
    f = AF.of("abc", [("a", "b"), ("b", "c")])
    assert grounded(f) == fs("a", "c")
    assert [status(f, x, "gr") for x in "abc"] == ["in", "out", "in"]


def test_mutual_attack():
    f = AF.of("ab", [("a", "b"), ("b", "a")])
    assert grounded(f) == fs()
    assert extensions(f, "pr") == [fs("a"), fs("b")]
    assert extensions(f, "st") == [fs("a"), fs("b")]
    assert status(f, "a", "gr") == "undecided"
    assert status(f, "a", "pr") == "in"
    assert not skeptically_accepted(f, "a", "pr")


def test_odd_cycle_has_no_stable_extension():
    f = AF.of("abc", [("a", "b"), ("b", "c"), ("c", "a")])
    assert grounded(f) == fs()
    assert extensions(f, "pr") == [fs()]
    assert extensions(f, "st") == []


def test_dung_example_preferred_vs_grounded():
    # a<->b, b->c: grounded is empty; preferred {a,c} and {b}
    f = AF.of("abc", [("a", "b"), ("b", "a"), ("b", "c")])
    assert grounded(f) == fs()
    assert extensions(f, "pr") == [fs("b"), fs("a", "c")]
    assert extensions(f, "st") == [fs("b"), fs("a", "c")]


def test_self_attacker_blocks_its_victim_from_grounded():
    f = AF.of("ab", [("a", "a"), ("a", "b")])
    assert grounded(f) == fs()
    assert not admissible(f, fs("a"))
    assert status(f, "b", "gr") == "undecided"


def test_unattacked_arguments_are_grounded():
    f = AF.of("abc", [("a", "b")])
    assert grounded(f) == fs("a", "c")


def test_defense_set_of_paper_example():
    # F0 of the running example: a0 -> a4, a3 -> a0; a1, a2 unrelated.
    f = AF.of(["a0", "a1", "a2", "a3", "a4"], [("a0", "a4"), ("a3", "a0")])
    assert defense_sets(f, "a4", "gr") == [fs("a3", "a4")]
    f1 = AF.of(f.args | {"a5"}, f.attacks | {("a5", "a3")})
    assert defense_sets(f1, "a4", "gr") == []
    assert status(f1, "a4", "gr") == "out"


def test_unknown_semantics():
    with pytest.raises(ValueError):
        extensions(AF.of("a"), "xx")
