from beliefnet.data.synthetic.single_pair import (
    CONTRADICTION_TYPES,
    LABELS,
    SyntheticConfig,
    difficulty_for,
    generate,
    surface_jaccard,
)


def _dataset():
    return generate(SyntheticConfig(n_per_type=120, seed=0))


def test_fields_and_label_space():
    for ex in _dataset():
        assert ex.label in LABELS
        assert ex.contradiction_type in CONTRADICTION_TYPES
        assert ex.difficulty in {"easy", "medium", "hard"}
        assert ex.claim and ex.evidence
        assert "surface_jaccard" in ex.metadata
        assert ex.split in {"train", "val", "test"}


def test_type_label_consistency():
    for ex in _dataset():
        if ex.contradiction_type == "entailment":
            assert ex.label == "entailment"
        elif ex.contradiction_type == "neutral":
            assert ex.label == "neutral"
        else:
            assert ex.label == "contradiction"


def test_no_exact_pair_leakage_across_splits():
    examples = _dataset()
    split_of: dict[tuple, set] = {}
    for ex in examples:
        split_of.setdefault((ex.claim, ex.evidence), set()).add(ex.split)
    assert all(len(s) == 1 for s in split_of.values())


def test_all_types_present_in_each_split():
    examples = _dataset()
    for split in ("train", "val", "test"):
        present = {e.contradiction_type for e in examples if e.split == split}
        assert present == set(CONTRADICTION_TYPES), (split, present)


def test_hard_contradictions_overlap_more_than_generic():
    examples = _dataset()

    def mean_jac(t):
        vals = [e.metadata["surface_jaccard"] for e in examples if e.contradiction_type == t]
        return sum(vals) / len(vals)

    generic = mean_jac("generic")
    for t in ("minimal_edit", "numeric", "temporal", "entity_swap"):
        assert mean_jac(t) >= 0.5, (t, mean_jac(t))
        assert mean_jac(t) > generic, (t, mean_jac(t), generic)


def test_difficulty_rule():
    assert difficulty_for("contradiction", 0.9) == "hard"
    assert difficulty_for("contradiction", 0.55) == "hard"
    assert difficulty_for("contradiction", 0.3) == "medium"
    assert difficulty_for("entailment", 0.9) == "easy"


def test_surface_jaccard_bounds():
    assert surface_jaccard("a b c", "a b c") == 1.0
    assert surface_jaccard("a b", "c d") == 0.0
