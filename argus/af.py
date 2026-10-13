"""Dung-style abstract argumentation frameworks and their semantics.

Implements Definition 1 (AF) and the semantics used by ARGUS: grounded (gr),
preferred (pr) and stable (st). Extensions are enumerated by brute force over
subsets, which is exact and adequate for toy-sized frameworks (<= ~16 arguments).
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable

SEMANTICS = ("gr", "pr", "st")
MAX_ARGS = 16  # brute-force guard


@dataclass(frozen=True)
class AF:
    """F = (A, R): a finite set of arguments and a binary attack relation."""

    args: frozenset[str]
    attacks: frozenset[tuple[str, str]]

    def __post_init__(self) -> None:
        for a, b in self.attacks:
            if a not in self.args or b not in self.args:
                raise ValueError(f"attack ({a},{b}) has an endpoint outside A")

    @staticmethod
    def of(args: Iterable[str], attacks: Iterable[tuple[str, str]] = ()) -> "AF":
        return AF(frozenset(args), frozenset(attacks))

    def attackers(self, a: str) -> frozenset[str]:
        return frozenset(x for x, y in self.attacks if y == a)

    def targets(self, a: str) -> frozenset[str]:
        return frozenset(y for x, y in self.attacks if x == a)


def conflict_free(af: AF, s: frozenset[str]) -> bool:
    """No two members of S attack each other (a self-attacker is never conflict-free)."""
    return not any(x in s and y in s for x, y in af.attacks)


def attacks_set(af: AF, s: frozenset[str], a: str) -> bool:
    """Some member of S attacks a."""
    return any(x in s and y == a for x, y in af.attacks)


def defends(af: AF, s: frozenset[str], a: str) -> bool:
    """S defends a: every attacker of a is attacked by some member of S."""
    return all(attacks_set(af, s, b) for b in af.attackers(a))


def admissible(af: AF, s: frozenset[str]) -> bool:
    return conflict_free(af, s) and all(defends(af, s, a) for a in s)


def characteristic(af: AF, s: frozenset[str]) -> frozenset[str]:
    """Characteristic function F(S) = {a | S defends a}."""
    return frozenset(a for a in af.args if defends(af, s, a))


def grounded(af: AF) -> frozenset[str]:
    """Least fixed point of the characteristic function."""
    s: frozenset[str] = frozenset()
    while True:
        nxt = characteristic(af, s)
        if nxt == s:
            return s
        s = nxt


def _subsets(af: AF) -> Iterable[frozenset[str]]:
    if len(af.args) > MAX_ARGS:
        raise ValueError(f"brute-force semantics limited to {MAX_ARGS} arguments")
    names = sorted(af.args)
    for r in range(len(names) + 1):
        for c in combinations(names, r):
            yield frozenset(c)


def admissible_sets(af: AF) -> list[frozenset[str]]:
    return [s for s in _subsets(af) if admissible(af, s)]


def preferred(af: AF) -> list[frozenset[str]]:
    """Maximal (w.r.t. set inclusion) admissible sets."""
    adm = admissible_sets(af)
    return sorted((s for s in adm if not any(s < t for t in adm)), key=_key)


def stable(af: AF) -> list[frozenset[str]]:
    """Conflict-free sets that attack every argument outside the set."""
    return sorted(
        (
            s
            for s in _subsets(af)
            if conflict_free(af, s) and all(attacks_set(af, s, a) for a in af.args - s)
        ),
        key=_key,
    )


def _key(s: frozenset[str]) -> tuple[int, list[str]]:
    return (len(s), sorted(s))


def extensions(af: AF, sigma: str) -> list[frozenset[str]]:
    """sigma(F) for sigma in {gr, pr, st}."""
    if sigma == "gr":
        return [grounded(af)]
    if sigma == "pr":
        return preferred(af)
    if sigma == "st":
        return stable(af)
    raise ValueError(f"unknown semantics {sigma!r}; expected one of {SEMANTICS}")


def credulously_accepted(af: AF, a: str, sigma: str) -> bool:
    return any(a in e for e in extensions(af, sigma))


def skeptically_accepted(af: AF, a: str, sigma: str) -> bool:
    exts = extensions(af, sigma)
    return bool(exts) and all(a in e for e in exts)


def status(af: AF, a: str, sigma: str) -> str:
    """Status of an argument: 'in', 'out' or 'undecided'.

    Grounded: 'in' if in the grounded extension, 'out' if attacked by it,
    otherwise 'undecided'. Preferred/stable: 'in' if credulously accepted
    (the acceptance criterion of the paper's verification step), else 'out'.
    """
    if a not in af.args:
        raise KeyError(a)
    if sigma == "gr":
        g = grounded(af)
        if a in g:
            return "in"
        return "out" if attacks_set(af, g, a) else "undecided"
    return "in" if credulously_accepted(af, a, sigma) else "out"


def defense_sets(af: AF, t: str, sigma: str) -> list[frozenset[str]]:
    """Def_sigma(t): minimal admissible D with t in D and D inside some sigma-extension."""
    exts = extensions(af, sigma)
    cands = [s for s in admissible_sets(af) if t in s and any(s <= e for e in exts)]
    return sorted((s for s in cands if not any(c < s for c in cands)), key=_key)
