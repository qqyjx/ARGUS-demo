"""Three SYNTHETIC toy cases. All questions, answers, claims and evidence are made up."""
from __future__ import annotations

from dataclasses import dataclass

from .llm import ToyKB


@dataclass(frozen=True)
class Case:
    key: str
    title: str
    question: str
    answer: str          # phrase that identifies the target claim
    explanation: str
    evidence: tuple[str, ...]
    kb: ToyKB
    semantics: str = "gr"


CASES: tuple[Case, ...] = (
    Case(
        key="diagnosis",
        title="Case 1: late evidence defeats a supporting claim",
        question="A patient has fatigue and joint pain. What is the most likely diagnosis?",
        answer="lupus",
        explanation=(
            "The patient most likely has lupus because lupus commonly presents with fatigue "
            "and joint pain. The patient reports chronic fatigue. Polyarthralgia is present. "
            "However, the symptoms are non-specific."
        ),
        evidence=("A verified lab report states the ANA test is negative.",),
        kb=ToyKB(
            defeats=(
                ("symptoms are non-specific", "has lupus"),
                ("lupus commonly presents", "symptoms are non-specific"),
                ("ana test is negative", "lupus commonly presents"),
                ("anti-dsdna antibodies are positive", "ana test is negative"),
            ),
            entails=(("fatigue and joint pain", "has lupus"),),
            counterarguments={"ana test is negative": ("Anti-dsDNA antibodies are positive",)},
        ),
    ),
    Case(
        key="museum",
        title="Case 2: two conflicting sources (grounded vs preferred semantics)",
        question="Is the Harbor Museum open on Mondays?",
        answer="open on mondays",
        explanation=(
            "The museum is open on Mondays because the brochure lists Monday hours. "
            "However, the website says the museum is closed on Mondays."
        ),
        evidence=(),
        kb=ToyKB(
            defeats=(
                ("closed on mondays", "open on mondays"),
                ("open on mondays", "closed on mondays"),
                ("website page is outdated", "closed on mondays"),
            ),
            counterarguments={"closed on mondays": ("The website page is outdated",)},
        ),
    ),
    Case(
        key="capital",
        title="Case 3: consistent explanation, irrelevant evidence (vacuity)",
        question="What is the capital of Australia?",
        answer="canberra",
        explanation=(
            "The capital of Australia is Canberra because Canberra hosts the federal parliament. "
            "Some people assume the capital is Sydney. Sydney is the largest city."
        ),
        evidence=("Melbourne was the temporary seat of parliament until 1927.",),
        kb=ToyKB(
            defeats=(
                ("capital is sydney", "capital of australia is canberra"),
                ("canberra hosts the federal parliament", "capital is sydney"),
            ),
        ),
    ),
)
