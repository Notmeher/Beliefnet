"""Tests for the constraint-type schema and supervision-to-targets converter."""
from __future__ import annotations

from beliefnet.graph.constraint_types import (
    BIO_TAG_IDX,
    BIO_TAGS,
    N_BIO_TAGS,
    N_TYPES,
    STATEMENT_TYPES,
    TYPE_TO_IDX,
    supervision_to_targets,
)


def test_statement_types_unique_and_indexed():
    assert len(set(STATEMENT_TYPES)) == len(STATEMENT_TYPES) == N_TYPES
    for i, t in enumerate(STATEMENT_TYPES):
        assert TYPE_TO_IDX[t] == i


def test_bio_tags_have_b_and_i_for_each_entity():
    assert "O" in BIO_TAGS
    entity_kinds = {tag.split("-", 1)[1] for tag in BIO_TAGS if tag != "O"}
    for ent in entity_kinds:
        assert f"B-{ent}" in BIO_TAGS and f"I-{ent}" in BIO_TAGS
    assert len(set(BIO_TAGS)) == N_BIO_TAGS == len(BIO_TAGS)
    for i, tag in enumerate(BIO_TAGS):
        assert BIO_TAG_IDX[tag] == i


def test_filler_supervision_emits_all_minus_one():
    out = supervision_to_targets({"type": "filler"})
    assert out["type"] == TYPE_TO_IDX["filler"]
    for k in ("order_event1", "order_event2", "name1", "name2",
              "symptom", "diagnosis", "drug", "arm"):
        assert out[k] == -1
    assert out["value_mask"] == 0.0


def test_before_supervision_indexes_both_events():
    from beliefnet.data.synthetic.vocabulary import ORDER_EVENTS
    a, b = ORDER_EVENTS[0], ORDER_EVENTS[1]
    out = supervision_to_targets({"type": "before", "entity1": a, "entity2": b})
    assert out["type"] == TYPE_TO_IDX["before"]
    assert out["order_event1"] == 0 and out["order_event2"] == 1
    assert out["name1"] == -1 and out["symptom"] == -1
    assert out["value_mask"] == 0.0


def test_arm_count_carries_value_and_arm_index():
    out = supervision_to_targets({"type": "arm_count", "arm": 1, "value": 75})
    assert out["type"] == TYPE_TO_IDX["arm_count"]
    assert out["arm"] == 1
    assert out["value"] == 75.0
    assert out["value_mask"] == 1.0


def test_total_count_value_mask_set():
    out = supervision_to_targets({"type": "total_count", "value": 200})
    assert out["value"] == 200.0
    assert out["value_mask"] == 1.0
    assert out["arm"] == -1


def test_categorical_supervision_indexes_correct_slot():
    from beliefnet.data.synthetic.vocabulary import DIAGNOSES, DRUGS, SYMPTOMS
    sym = supervision_to_targets({"type": "presence_pos", "symptom": SYMPTOMS[2]})
    dx = supervision_to_targets({"type": "diagnosis", "diagnosis": DIAGNOSES[0]})
    drg = supervision_to_targets({"type": "drug_active", "drug": DRUGS[-1]})
    assert sym["symptom"] == 2 and sym["diagnosis"] == -1
    assert dx["diagnosis"] == 0 and dx["symptom"] == -1
    assert drg["drug"] == len(DRUGS) - 1


def test_unknown_entity_maps_to_minus_one():
    out = supervision_to_targets({"type": "before",
                                  "entity1": "an_unknown_event",
                                  "entity2": "another_unknown"})
    assert out["order_event1"] == -1 and out["order_event2"] == -1
