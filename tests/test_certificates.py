from beliefnet.reasoning.certificates import (
    minimal_correction_set,
    minimal_unsat_core,
)
from beliefnet.reasoning.exact_checker import is_satisfiable


def C(t, *args, values=None):
    d = {"type": t, "args": list(args)}
    if values is not None:
        d["values"] = values
    return d


def test_core_of_three_cycle_is_all_three():
    cyc = [C("before", "A", "B"), C("before", "B", "C"), C("before", "C", "A")]
    core = minimal_unsat_core(cyc)
    assert len(core) == 3
    assert not is_satisfiable(core)
    for c in core:  # minimality: dropping any one restores SAT
        assert is_satisfiable([x for x in core if x is not c])


def test_core_ignores_irrelevant_constraints():
    cons = [C("before", "A", "B"), C("before", "B", "C"), C("before", "C", "A"),
            C("before", "X", "Y")]
    core = minimal_unsat_core(cons)
    assert len(core) == 3
    assert all(c["args"] != ["X", "Y"] for c in core)


def test_satisfiable_set_has_empty_core():
    assert minimal_unsat_core([C("before", "A", "B")]) == []


def test_correction_set_restores_sat():
    cyc = [C("before", "A", "B"), C("before", "B", "C"), C("before", "C", "A")]
    mcs = minimal_correction_set(cyc)
    assert len(mcs) == 1
    kept = [c for c in cyc if c not in mcs]
    assert is_satisfiable(kept)


def test_anchor_conflict_core_is_the_two_anchors():
    cons = [C("anchor", "P", values=[1.0]), C("anchor", "P", values=[-1.0]),
            C("anchor", "Q", values=[5.0])]
    core = minimal_unsat_core(cons)
    assert len(core) == 2
    assert all(c["args"] == ["P"] for c in core)
