import re

from beliefnet.data.synthetic.case_schema import (
    CONFLICT_KINDS,
    CONSISTENCY_LABELS,
    MultiStmtConfig,
)
from beliefnet.data.synthetic.multi_statement import generate


def _data():
    return generate(MultiStmtConfig(n_per_subtype=120, seed=0))


def test_schema_and_label_space():
    for c in _data():
        assert c.label in CONSISTENCY_LABELS
        assert c.conflict_kind in CONFLICT_KINDS
        assert len(c.statements) >= 2
        assert c.split in {"train", "val", "test"}
        assert all(0 <= i < len(c.statements) for i in c.conflict_indices)


def test_label_kind_consistency():
    for c in _data():
        if c.label == "consistent":
            assert c.conflict_kind == "none" and c.conflict_indices == []
        else:
            assert c.conflict_kind in {"pairwise", "non_pairwise"}
            assert len(c.conflict_indices) >= 2


def test_non_pairwise_has_no_directly_contradictory_pair():
    """Temporal cycles: every pair is a single 'X before Y' relation, so no
    two statements directly contradict; the conflict is only global."""
    for c in _data():
        if c.conflict_subtype == "temporal_cycle":
            befores = [s for s in c.statements if "occurred before" in s]
            # extract (a, b) ordered pairs; no pair should be the exact reverse
            pairs = []
            for s in befores:
                m = re.match(r"(.+?) occurred before (.+?)\.", s)
                pairs.append((m.group(1), m.group(2)))
            for a, b in pairs:
                assert (b, a) not in pairs  # a reverse pair would be a pairwise conflict


def test_all_subtypes_present_each_split():
    cases = _data()
    subtypes = {c.conflict_subtype for c in cases}
    for split in ("train", "val", "test"):
        present = {c.conflict_subtype for c in cases if c.split == split}
        assert present == subtypes, (split, subtypes - present)


def test_conflict_indices_point_to_core_statements():
    # numeric_sum non-pairwise: the 3 flagged statements must include the total
    for c in _data():
        if c.conflict_subtype == "numeric_sum" and c.conflict_kind == "non_pairwise":
            flagged = [c.statements[i] for i in c.conflict_indices]
            assert any("in total" in s for s in flagged)
