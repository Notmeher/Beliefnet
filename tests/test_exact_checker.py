from itertools import combinations

from beliefnet.reasoning.exact_checker import is_satisfiable


def C(t, *args, values=None):
    d = {"type": t, "args": list(args)}
    if values is not None:
        d["values"] = values
    return d


def test_chain_satisfiable():
    cons = [C("before", "A", "B"), C("before", "B", "C"), C("before", "A", "C")]
    assert is_satisfiable(cons)


def test_two_cycle_unsat():
    assert not is_satisfiable([C("before", "A", "B"), C("before", "B", "A")])


def test_three_cycle_unsat_but_all_pairs_ok():
    cyc = [C("before", "A", "B"), C("before", "B", "C"), C("before", "C", "A")]
    assert not is_satisfiable(cyc)
    for sub in combinations(cyc, 2):  # emergence order 3: every pair is fine
        assert is_satisfiable(list(sub))


def test_greater_cycle_unsat():
    assert not is_satisfiable([C("greater", "A", "B"), C("greater", "B", "A")])


def test_disjoint_before_and_greater_consistent():
    assert is_satisfiable([C("before", "A", "B"), C("greater", "C", "D")])


def test_anchor_conflict_unsat():
    assert not is_satisfiable(
        [C("anchor", "P", values=[1.0]), C("anchor", "P", values=[-1.0])])


def test_good_sum_sat():
    cons = [C("anchor", "A", values=[40]), C("anchor", "B", values=[30]),
            C("anchor", "T", values=[70]), C("sum_eq", "A", "B", "T")]
    assert is_satisfiable(cons)


def test_bad_sum_unsat():
    cons = [C("anchor", "A", values=[40]), C("anchor", "B", values=[30]),
            C("anchor", "T", values=[100]), C("sum_eq", "A", "B", "T")]
    assert not is_satisfiable(cons)


def test_five_cycle_emergent():
    ev = ["a", "b", "c", "d", "e"]
    cyc = [C("before", ev[i], ev[(i + 1) % 5]) for i in range(5)]
    assert not is_satisfiable(cyc)
    for sub in combinations(cyc, 4):  # every 4-subset is acyclic
        assert is_satisfiable(list(sub))
