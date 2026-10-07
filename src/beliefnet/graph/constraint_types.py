"""Label tables for typed constraints, statement types, and BIO tags.

Pure data (no parsing logic): statement-type vocab, entity index tables, BIO
tag vocab, and the helper that converts a parsed supervision dict into the
integer/float targets the perception heads are trained against.

Parsing rules live in ``beliefnet.perception.parser_rules``.
"""
from __future__ import annotations

from beliefnet.data.synthetic.vocabulary import (
    DIAGNOSES,
    DRUGS,
    NAMES,
    ORDER_EVENTS,
    SYMPTOMS,
)

STATEMENT_TYPES = (
    "filler", "before", "greater", "arm_count", "total_count",
    "direct_enrolled", "presence_pos", "presence_neg", "diagnosis", "drug_active",
)
TYPE_TO_IDX = {t: i for i, t in enumerate(STATEMENT_TYPES)}

ORDER_EVENT_TO_IDX = {e: i for i, e in enumerate(ORDER_EVENTS)}
NAME_TO_IDX = {n: i for i, n in enumerate(NAMES)}
SYMPTOM_TO_IDX = {s: i for i, s in enumerate(SYMPTOMS)}
DIAGNOSIS_TO_IDX = {d: i for i, d in enumerate(DIAGNOSES)}
DRUG_TO_IDX = {d: i for i, d in enumerate(DRUGS)}

N_ORDER_EVENTS = len(ORDER_EVENTS)
N_NAMES = len(NAMES)
N_SYMPTOMS = len(SYMPTOMS)
N_DIAGNOSES = len(DIAGNOSES)
N_DRUGS = len(DRUGS)
N_ARMS = 2  # only treatment / control are used as ARM0/ARM1
N_TYPES = len(STATEMENT_TYPES)


def supervision_to_targets(sup: dict) -> dict:
    """Convert a supervision dict into integer/float targets indexed by head."""
    t = sup["type"]
    out = {
        "type": TYPE_TO_IDX[t],
        "order_event1": -1, "order_event2": -1,
        "name1": -1, "name2": -1,
        "symptom": -1, "diagnosis": -1, "drug": -1, "arm": -1,
        "value": 0.0, "value_mask": 0.0,
    }
    if t == "before":
        out["order_event1"] = ORDER_EVENT_TO_IDX.get(sup["entity1"], -1)
        out["order_event2"] = ORDER_EVENT_TO_IDX.get(sup["entity2"], -1)
    elif t == "greater":
        out["name1"] = NAME_TO_IDX.get(sup["entity1"], -1)
        out["name2"] = NAME_TO_IDX.get(sup["entity2"], -1)
    elif t in ("arm_count", "total_count", "direct_enrolled"):
        out["value"] = float(sup.get("value", 0.0))
        out["value_mask"] = 1.0
        if t == "arm_count":
            out["arm"] = int(sup.get("arm", 0))
    elif t in ("presence_pos", "presence_neg"):
        out["symptom"] = SYMPTOM_TO_IDX.get(sup["symptom"], -1)
    elif t == "diagnosis":
        out["diagnosis"] = DIAGNOSIS_TO_IDX.get(sup["diagnosis"], -1)
    elif t == "drug_active":
        out["drug"] = DRUG_TO_IDX.get(sup["drug"], -1)
    return out


BIO_TAGS = (
    "O",
    "B-EVENT", "I-EVENT",
    "B-NAME", "I-NAME",
    "B-SYMPTOM", "I-SYMPTOM",
    "B-DIAGNOSIS", "I-DIAGNOSIS",
    "B-DRUG", "I-DRUG",
)
BIO_TAG_IDX = {t: i for i, t in enumerate(BIO_TAGS)}
N_BIO_TAGS = len(BIO_TAGS)
