"""Tests for build_graph_from_supervisions on hand-crafted supervision lists."""
from __future__ import annotations

from beliefnet.graph.graph_builder import build_graph_from_supervisions


def _constraint(t, args, values=None):
    rec = {"type": t, "args": list(args)}
    if values is not None:
        rec["values"] = list(values)
    return rec


def test_empty_supervisions_emit_empty_graph():
    ents, constraints = build_graph_from_supervisions([])
    assert ents == [] and constraints == []


def test_fillers_emit_nothing():
    ents, constraints = build_graph_from_supervisions([{"type": "filler"}] * 3)
    assert ents == [] and constraints == []


def test_before_relation_adds_two_entities_and_one_edge():
    ents, constraints = build_graph_from_supervisions([
        {"type": "before", "entity1": "biopsy", "entity2": "surgery"},
    ])
    assert ents == ["biopsy", "surgery"]
    assert constraints == [_constraint("before", ["biopsy", "surgery"])]


def test_self_loop_relation_dropped():
    """A self-pointing edge has no semantic content; the assembler skips it."""
    ents, constraints = build_graph_from_supervisions([
        {"type": "before", "entity1": "biopsy", "entity2": "biopsy"},
    ])
    assert ents == [] and constraints == []


def test_arm_total_combo_triggers_sum_eq():
    ents, constraints = build_graph_from_supervisions([
        {"type": "arm_count", "arm": 0, "value": 40},
        {"type": "arm_count", "arm": 1, "value": 30},
        {"type": "total_count", "value": 70},
    ])
    assert set(ents) == {"ARM0", "ARM1", "TOTAL"}
    types = [c["type"] for c in constraints]
    assert types.count("anchor") == 3 and types.count("sum_eq") == 1
    sum_eq = next(c for c in constraints if c["type"] == "sum_eq")
    assert sum_eq["args"] == ["ARM0", "ARM1", "TOTAL"]


def test_arm_only_without_total_does_not_emit_sum_eq():
    _, constraints = build_graph_from_supervisions([
        {"type": "arm_count", "arm": 0, "value": 40},
        {"type": "arm_count", "arm": 1, "value": 30},
    ])
    assert not any(c["type"] == "sum_eq" for c in constraints)


def test_presence_pos_and_neg_emit_opposite_anchors():
    ents, constraints = build_graph_from_supervisions([
        {"type": "presence_pos", "symptom": "fever"},
        {"type": "presence_neg", "symptom": "fever"},
    ])
    # both target the same entity slot
    assert ents == ["presence_fever"]
    values = [c["values"][0] for c in constraints]
    assert sorted(values) == [-1.0, 1.0]


def test_diagnosis_uses_shared_slot_and_codes_lookup():
    from beliefnet.data.synthetic.vocabulary import DIAGNOSIS_CODES
    dx = next(iter(DIAGNOSIS_CODES))
    ents, constraints = build_graph_from_supervisions([
        {"type": "diagnosis", "diagnosis": dx},
    ])
    assert ents == ["diagnosis_slot"]
    assert constraints[0]["values"][0] == float(DIAGNOSIS_CODES[dx])


def test_direct_enrolled_creates_trial_enrolled_anchor():
    ents, constraints = build_graph_from_supervisions([
        {"type": "direct_enrolled", "value": 250},
    ])
    assert ents == ["trial_enrolled"]
    assert constraints == [_constraint("anchor", ["trial_enrolled"], values=[250.0])]
