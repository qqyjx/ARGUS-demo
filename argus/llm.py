"""LLM roles used by ARGUS, behind one interface, plus a deterministic mock.

In the paper the LLM (a) decomposes a rationale into argument units, (b) is
paired with an NLI model that labels claim pairs, and (c) writes a counterargument when
the repair adds an argument. `MockLLM` fills all three roles with string rules
over a small synthetic knowledge base; it understands only the toy cases.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol, Sequence

CONTRADICTION, ENTAILMENT, NEUTRAL = "contradiction", "entailment", "neutral"


@dataclass(frozen=True)
class RawUnit:
    """Structured extraction record: claim, premises (indices), confidence."""

    claim: str
    premises: tuple[int, ...]
    confidence: float


@dataclass(frozen=True)
class Extraction:
    units: tuple[RawUnit, ...]
    target: int  # index of the unit that supports the answer


class LLMClient(Protocol):
    def extract_units(self, question: str, answer: str, explanation: str) -> Extraction:
        """Decompose a free-form rationale into argument units (Section 3.1)."""

    def classify(self, claim_i: str, claim_j: str) -> str:
        """Relation of claim_i to claim_j: contradiction | entailment | neutral (Section 3.2)."""

    def counter(self, claim: str, evidence: Sequence[str]) -> list[str]:
        """Candidate counterarguments of a claim, used as add_arg candidates (Section 3.4)."""


@dataclass(frozen=True)
class ToyKB:
    """Synthetic knowledge base for one toy case. All content is made up.

    defeats: (attacker_phrase, target_phrase). A claim containing the first
        phrase is labelled `contradiction` against a claim containing the second
        (directed, standing in for NLI plus the attack template library).
    entails: (premise_phrase, conclusion_phrase), recorded as a support link.
    counterarguments: phrase -> candidate counterargument claims.
    """

    defeats: tuple[tuple[str, str], ...] = ()
    entails: tuple[tuple[str, str], ...] = ()
    counterarguments: dict[str, tuple[str, ...]] = field(default_factory=dict)


_HEDGES = (  # (pattern, confidence): first match wins
    (r"\b(verified|definitely|clearly)\b", 0.95),
    (r"\b(might|may|assume|assumes)\b", 0.5),
    (r"\b(probably|likely)\b", 0.7),
)
_DEFAULT_CONF = 0.8
_MARKER = re.compile(r"^(however|therefore|also|moreover),?\s+", re.I)


def confidence_of(claim: str) -> float:
    """Self-assessed confidence proxy from hedge words (rule based)."""
    for pat, g in _HEDGES:
        if re.search(pat, claim, re.I):
            return g
    return _DEFAULT_CONF


class MockLLM:
    """Deterministic, offline stand-in for the extraction / NLI / counterargument roles."""

    def __init__(self, kb: ToyKB) -> None:
        self.kb = kb

    def extract_units(self, question: str, answer: str, explanation: str) -> Extraction:
        units: list[RawUnit] = []
        for sent in re.split(r"(?<=[.!?])\s+", explanation.strip()):
            sent = _MARKER.sub("", sent.strip().rstrip(".!?"))
            if not sent:
                continue
            main, _, because = sent.partition(" because ")
            idx = len(units)
            units.append(RawUnit(main, (idx + 1,) if because else (), confidence_of(main)))
            if because:  # atomic split: "X because Y" -> units X and Y
                units.append(RawUnit(because, (), confidence_of(because)))
        for i, u in enumerate(units):
            if answer.lower() in u.claim.lower():
                return Extraction(tuple(units), i)
        raise ValueError("no extracted unit supports the answer")

    def classify(self, claim_i: str, claim_j: str) -> str:
        a, b = claim_i.lower(), claim_j.lower()
        if any(p in a and q in b for p, q in self.kb.defeats):
            return CONTRADICTION
        if any(p in a and q in b for p, q in self.kb.entails):
            return ENTAILMENT
        return NEUTRAL

    def counter(self, claim: str, evidence: Sequence[str]) -> list[str]:
        low = claim.lower()
        return [r for phrase, rs in sorted(self.kb.counterarguments.items()) if phrase in low for r in rs]
