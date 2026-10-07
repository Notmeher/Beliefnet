from beliefnet.data.emergent import generate_emergent
from beliefnet.graph.graph_builder import build_graph_from_supervisions
from beliefnet.perception.local_extract import rule_supervisions
from beliefnet.reasoning.exact_checker import is_satisfiable


def test_rule_extract_parses_before():
    sups = rule_supervisions(["the biopsy occurred before surgery."])
    assert sups[0]["type"] == "before"
    assert sups[0]["entity1"] == "the biopsy"
    assert sups[0]["entity2"] == "surgery"


def test_rule_extract_filler_for_neutral():
    sups = rule_supervisions(["The study was approved by the ethics board."])
    assert sups[0]["type"] == "filler"


def test_rule_pipeline_matches_labels():
    cases = generate_emergent((2, 3, 5), (6, 12), n_per_cell=8, seed=5)
    for c in cases:
        _, cons = build_graph_from_supervisions(rule_supervisions(c.statements))
        sat = is_satisfiable(cons)
        assert (not sat) == (c.label == "inconsistent"), c.conflict_subtype
