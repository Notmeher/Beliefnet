from itertools import combinations

from beliefnet.data.emergent import generate_emergent
from beliefnet.reasoning.exact_checker import is_satisfiable


def test_labels_match_exact_oracle():
    cases = generate_emergent((2, 3, 5), (6, 12), n_per_cell=10, seed=1)
    for c in cases:
        sat = is_satisfiable(c.metadata["constraints"])
        assert (not sat) == (c.label == "inconsistent"), c.conflict_subtype


def test_balanced_and_matched_length():
    cases = generate_emergent((3,), (9,), n_per_cell=20, seed=2)
    inc = [c for c in cases if c.label == "inconsistent"]
    con = [c for c in cases if c.label == "consistent"]
    assert len(inc) == len(con) == 20
    assert all(c.metadata["n_statements"] == 9 for c in cases)


def test_emergence_order_property_k3():
    cases = generate_emergent((3,), (3,), n_per_cell=5, seed=3)
    for c in cases:
        if c.label != "inconsistent":
            continue
        cons = c.metadata["constraints"]
        assert not is_satisfiable(cons)
        for sub in combinations(cons, 2):
            assert is_satisfiable(list(sub))


def test_metadata_carries_hop_count_and_emergence_order():
    cases = generate_emergent((2, 5), (8,), n_per_cell=3, seed=4)
    for c in cases:
        md = c.metadata
        assert md["hop_count"] in (2, 5)
        if c.label == "inconsistent":
            assert md["emergence_order"] == md["hop_count"]
        else:
            assert md["emergence_order"] is None
