"""Argumentation-semantic acceptability check (Section 3.3 of the paper)."""
from __future__ import annotations

from dataclasses import dataclass

from .af import AF, defense_sets, extensions, status

VERDICTS = {"in": "accepted", "out": "rejected", "undecided": "undecided"}


@dataclass(frozen=True)
class Verification:
    """Result of checking the target argument under one semantics."""

    semantics: str
    target: str
    status: str                                  # 'in' | 'out' | 'undecided'
    verdict: str                                 # 'accepted' | 'rejected' | 'undecided'
    extensions: tuple[frozenset[str], ...]
    defense_sets: tuple[frozenset[str], ...]     # certificate when accepted
    attackers: frozenset[str]                    # target's attackers, shown when not accepted

    @property
    def passes(self) -> bool:
        """The explanation is internally consistent iff the target is accepted."""
        return self.verdict == "accepted"


def verify(af: AF, target: str, sigma: str) -> Verification:
    """Check whether the target belongs to at least one sigma-extension."""
    st = status(af, target, sigma)
    accepted = st == "in"
    return Verification(
        semantics=sigma,
        target=target,
        status=st,
        verdict=VERDICTS[st],
        extensions=tuple(extensions(af, sigma)),
        defense_sets=tuple(defense_sets(af, target, sigma)) if accepted else (),
        attackers=frozenset() if accepted else af.attackers(target),
    )
