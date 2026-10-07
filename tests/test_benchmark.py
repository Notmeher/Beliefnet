from beliefnet.data.benchmark import build_benchmark, datasheet
from beliefnet.graph.graph_builder import build_graph_from_supervisions
from beliefnet.perception.local_extract import rule_supervisions
from beliefnet.reasoning.exact_checker import is_satisfiable


def test_build_benchmark_has_three_variants():
    b = build_benchmark(n_per_cell=3, seed=1)
    assert {"canonical", "paraphrase", "real_seeded"} <= set(b)
    assert all(len(v) > 0 for v in b.values())


def test_paraphrase_variant_still_parseable():
    b = build_benchmark(n_per_cell=5, seed=2)
    for c in b["paraphrase"]:
        _, cons = build_graph_from_supervisions(rule_supervisions(c.statements))
        assert (not is_satisfiable(cons)) == (c.label == "inconsistent")


def test_real_seeded_uses_real_distractors():
    b = build_benchmark(n_per_cell=5, seed=3)
    joined = " ".join(s for c in b["real_seeded"] for s in c.statements)
    assert "participants" in joined or "trial" in joined


def test_datasheet_counts_balanced():
    b = build_benchmark(n_per_cell=4, seed=4)
    for row in datasheet(b):
        assert row["consistent"] == row["inconsistent"]
