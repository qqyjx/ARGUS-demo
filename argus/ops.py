"""Evidence updates, edit operations and cost models (Definition 4 of the paper)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping

from .af import AF

ADD_ARG, DEL_ARG, ADD_ATT, DEL_ATT = "add_arg", "del_arg", "add_att", "del_att"


@dataclass(frozen=True)
class Op:
    """One elementary edit: add_arg(x), del_arg(x), add_att(x,y) or del_att(x,y)."""

    kind: str
    x: str
    y: str = ""

    def __str__(self) -> str:
        return f"{self.kind}({self.x})" if self.kind in (ADD_ARG, DEL_ARG) else f"{self.kind}({self.x},{self.y})"

    @property
    def is_deletion(self) -> bool:
        return self.kind in (DEL_ARG, DEL_ATT)


@dataclass(frozen=True)
class EvidenceUpdate:
    """Delta = (A+, R+, A-, R-): arguments/attacks to add and to remove."""

    args_add: frozenset[str] = frozenset()
    atts_add: frozenset[tuple[str, str]] = frozenset()
    args_del: frozenset[str] = frozenset()
    atts_del: frozenset[tuple[str, str]] = frozenset()


def apply_update(af: AF, delta: EvidenceUpdate) -> AF:
    """Incorporate the evidence update: (A + A+ - A-, (R + R+ - R-) restricted to A')."""
    args = (af.args | delta.args_add) - delta.args_del
    atts = ((af.attacks | delta.atts_add) - delta.atts_del)
    return AF(args, frozenset(e for e in atts if e[0] in args and e[1] in args))


def apply_ops(af: AF, ops: frozenset[Op] | set[Op] | tuple[Op, ...]) -> AF:
    """Execute edit operations; del_arg also removes incident attacks.

    Raises ValueError for an operation that does not apply (adding an existing
    argument or attack, deleting a missing one, attack with a missing endpoint).
    """
    args, atts = set(af.args), set(af.attacks)
    for o in ops:
        if o.kind == ADD_ARG:
            if o.x in args:
                raise ValueError(f"{o}: argument already present")
            args.add(o.x)
        elif o.kind == DEL_ARG:
            if o.x not in args:
                raise ValueError(f"{o}: argument missing")
            args.discard(o.x)
    for o in ops:
        if o.kind == ADD_ATT:
            if o.x not in args or o.y not in args or (o.x, o.y) in atts:
                raise ValueError(f"{o}: invalid attack addition")
            atts.add((o.x, o.y))
        elif o.kind == DEL_ATT:
            if (o.x, o.y) not in atts:
                raise ValueError(f"{o}: attack missing")
            atts.discard((o.x, o.y))
    atts = {e for e in atts if e[0] in args and e[1] in args}
    return AF(frozenset(args), frozenset(atts))


def apply(af: AF, delta: EvidenceUpdate, ops: frozenset[Op] | set[Op] | tuple[Op, ...]) -> AF:
    """AF' = apply(AF, Delta, Ops): incorporate Delta, then execute Ops."""
    return apply_ops(apply_update(af, delta), ops)


@dataclass(frozen=True)
class CostModel:
    """Strictly positive cost function kappa.

    name: 'uniform' (all ops cost 1); 'confidence' (del_arg(a) costs
    confidence[a], other ops cost 1); 'structure' (additions cost 1,
    deletions cost w > 1).
    """

    name: str = "uniform"
    w: float = 2.0
    confidence: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.name not in ("uniform", "confidence", "structure"):
            raise ValueError(f"unknown cost model {self.name!r}")
        if self.name == "structure" and self.w <= 1:
            raise ValueError("structure-preserving cost needs w > 1")
        if any(g <= 0 for g in self.confidence.values()):
            raise ValueError("confidences must be > 0 so that kappa stays positive")

    def cost(self, o: Op) -> float:
        if self.name == "confidence" and o.kind == DEL_ARG:
            return float(self.confidence.get(o.x, 1.0))
        if self.name == "structure" and o.is_deletion:
            return float(self.w)
        return 1.0

    def total(self, ops: Iterable[Op]) -> float:
        return sum(self.cost(o) for o in ops)

    def describe(self) -> str:
        if self.name == "structure":
            return f"structure-preserving (w={self.w:g})"
        return {"uniform": "uniform", "confidence": "confidence-weighted"}[self.name]
