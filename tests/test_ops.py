import pytest

from argus.af import AF
from argus.ops import CostModel, EvidenceUpdate, Op, apply, apply_ops, apply_update


F = AF.of("abc", [("a", "b"), ("b", "c")])


def test_del_arg_cascades_to_attacks():
    g = apply_ops(F, [Op("del_arg", "b")])
    assert g.args == frozenset("ac") and g.attacks == frozenset()


def test_update_then_ops():
    d = EvidenceUpdate(args_add=frozenset({"d"}), atts_add=frozenset({("d", "a")}))
    g = apply(F, d, [Op("del_att", "d", "a")])
    assert "d" in g.args and ("d", "a") not in g.attacks
    assert apply_update(F, d).attacks == F.attacks | {("d", "a")}


@pytest.mark.parametrize("op", [Op("add_arg", "a"), Op("del_arg", "z"), Op("del_att", "a", "c"),
                                Op("add_att", "a", "z"), Op("add_att", "a", "b")])
def test_invalid_ops_raise(op):
    with pytest.raises(ValueError):
        apply_ops(F, [op])


def test_cost_models():
    ops = [Op("add_arg", "x"), Op("add_att", "x", "a"), Op("del_arg", "b"), Op("del_att", "a", "b")]
    assert [CostModel("uniform").cost(o) for o in ops] == [1, 1, 1, 1]
    assert [CostModel("structure", w=3).cost(o) for o in ops] == [1, 1, 3, 3]
    conf = CostModel("confidence", confidence={"b": 0.9})
    assert [conf.cost(o) for o in ops] == [1, 1, 0.9, 1]
    with pytest.raises(ValueError):
        CostModel("structure", w=1)
    with pytest.raises(ValueError):
        CostModel("confidence", confidence={"b": 0.0})
