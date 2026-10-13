"""Verification check: accepted / rejected / undecided and the certificate."""
from argus.af import AF
from argus.verify import verify


def test_accepted_has_defense_set_and_passes():
    f = AF.of("abc", [("a", "b"), ("b", "c")])
    v = verify(f, "c", "gr")
    assert v.passes and v.verdict == "accepted"
    assert v.defense_sets == (frozenset({"a", "c"}),)


def test_rejected_fails_and_reports_attackers():
    f = AF.of("ab", [("a", "b")])
    v = verify(f, "b", "gr")
    assert not v.passes and v.verdict == "rejected"
    assert v.attackers == frozenset({"a"}) and v.defense_sets == ()


def test_undecided_under_grounded_but_accepted_under_preferred():
    f = AF.of("ab", [("a", "b"), ("b", "a")])
    assert verify(f, "a", "gr").verdict == "undecided"
    assert not verify(f, "a", "gr").passes
    assert verify(f, "a", "pr").passes


def test_rejected_under_preferred_when_in_no_extension():
    f = AF.of("ab", [("a", "b")])
    assert verify(f, "b", "pr").verdict == "rejected"
