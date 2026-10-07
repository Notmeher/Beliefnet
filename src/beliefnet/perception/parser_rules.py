"""Deterministic regex parser that converts a statement into a typed
supervision dict (the per-statement extraction the perception heads are
trained against), plus BIO-tag derivation/decoding and predicted-supervision
inversion.

Pure rules; no PyTorch, no learned components. The label tables this consumes
live in ``beliefnet.graph.constraint_types``; the vocab it indexes against
lives in ``beliefnet.data.synthetic.vocabulary``.
"""
from __future__ import annotations

import re

from beliefnet.data.synthetic.vocabulary import (
    DIAGNOSES,
    DRUGS,
    NAMES,
    NEUTRAL_FILLERS,
    ORDER_EVENTS,
    SYMPTOMS,
)
from beliefnet.graph.constraint_types import (
    BIO_TAG_IDX,
    BIO_TAGS,
    STATEMENT_TYPES,
)

_FILLERS = set(NEUTRAL_FILLERS)


_BEFORE_PATTERNS = [
    re.compile(r"^(.+?) occurred before (.+?)\.\s*$"),
    re.compile(r"^(.+?) preceded (.+?)\.\s*$"),
    re.compile(r"^(.+?) happened earlier than (.+?)\.\s*$"),
    re.compile(r"^(.+?) was observed prior to (.+?)\.\s*$"),
    re.compile(r"^(.+?) took place ahead of (.+?)\.\s*$"),
    re.compile(r"^(.+?) came before (.+?) in sequence\.\s*$"),
]
# Reversed-direction temporal patterns: "B happened after A" means
# "A before B"; capture group 1 is the LATER event, group 2 is the EARLIER.
# Parser swaps them so the supervision is always (early, late).
_BEFORE_REVERSED_PATTERNS = [
    re.compile(r"^(.+?) happened after (.+?)\.\s*$"),
    re.compile(r"^(.+?) occurred later than (.+?)\.\s*$"),
    re.compile(r"^(.+?) followed (.+?) in the timeline\.\s*$"),
    re.compile(r"^(.+?) came after (.+?)\.\s*$"),
]
_GREATER_PATTERNS = [
    re.compile(r"^(.+?) is older than (.+?)\.\s*$"),
    re.compile(r"^(.+?) is senior to (.+?)\.\s*$"),
    re.compile(r"^(.+?) has more years than (.+?)\.\s*$"),
    re.compile(r"^(.+?) outranks (.+?) in age\.\s*$"),
    re.compile(r"^(.+?)'s age exceeds that of (.+?)\.\s*$"),
]
# Reversed-direction comparative: "B is younger than A" means "A > B".
_GREATER_REVERSED_PATTERNS = [
    re.compile(r"^(.+?) is younger than (.+?)\.\s*$"),
    re.compile(r"^(.+?) is less old than (.+?)\.\s*$"),
    re.compile(r"^(.+?) is the junior of (.+?)\.\s*$"),
    re.compile(r"^(.+?) ranks below (.+?) in age\.\s*$"),
]
_TOTAL_PATTERNS = [
    re.compile(r"^The trial enrolled (\d+) patients in total\.\s*$"),
    re.compile(r"^A total of (\d+) patients participated in the trial\.\s*$"),
    re.compile(r"^The trial recruited (\d+) patients in all\.\s*$"),
    re.compile(r"^Across both arms, (\d+) patients were studied\.\s*$"),
    re.compile(r"^The combined enrollment was (\d+) patients\.\s*$"),
    re.compile(r"^The total trial enrollment was (\d+)\.\s*$"),
    re.compile(r"^Across the entire study, (\d+) patients enrolled\.\s*$"),
]
_DIRECT_PATTERNS = [
    re.compile(r"^The trial enrolled (\d+) patients\.\s*$"),
    re.compile(r"^The study included (\d+) participants\.\s*$"),
    re.compile(r"^(\d+) subjects were enrolled in the study\.\s*$"),
    re.compile(r"^The trial received (\d+) participants\.\s*$"),
    re.compile(r"^Across the study, (\d+) patients took part\.\s*$"),
]
_ARM_PATTERNS = [
    (re.compile(r"^The (treatment|control|placebo|observation) arm enrolled (\d+) patients\.\s*$"),
     "arm_then_n"),
    (re.compile(r"^The (treatment|control|placebo|observation) arm had (\d+) participants\.\s*$"),
     "arm_then_n"),
    (re.compile(r"^(\d+) patients were assigned to the (treatment|control|placebo|observation) arm\.\s*$"),
     "n_then_arm"),
    (re.compile(r"^In the (treatment|control|placebo|observation) arm, (\d+) patients were studied\.\s*$"),
     "arm_then_n"),
    (re.compile(r"^(\d+) patients made up the (treatment|control|placebo|observation) arm\.\s*$"),
     "n_then_arm"),
    (re.compile(r"^(\d+) patients were placed in the (treatment|control|placebo|observation) arm\.\s*$"),
     "n_then_arm"),
]
_PRES_NEG_PATTERNS = [
    re.compile(r"^The patient does not have (.+?)\.\s*$"),
    re.compile(r"^The patient lacks (.+?)\.\s*$"),
    re.compile(r"^The patient shows no sign of (.+?)\.\s*$"),
    re.compile(r"^The patient is free from (.+?)\.\s*$"),
    re.compile(r"^(.+?) was not observed in the patient\.\s*$"),
    re.compile(r"^The patient is without (.+?)\.\s*$"),
    re.compile(r"^No (.+?) was detected in the patient\.\s*$"),
]
_PRES_POS_PATTERNS = [
    re.compile(r"^The patient has (.+?)\.\s*$"),
    re.compile(r"^The patient (?:reported|presents with) (.+?)\.\s*$"),
    re.compile(r"^The patient exhibits (.+?)\.\s*$"),
    re.compile(r"^The patient shows (.+?)\.\s*$"),
    re.compile(r"^(.+?) was observed in the patient\.\s*$"),
    re.compile(r"^The patient experiences (.+?)\.\s*$"),
    re.compile(r"^(.+?) is present in the patient\.\s*$"),
]
_DIAGNOSIS_PATTERNS = [
    re.compile(r"^The patient was diagnosed with (.+?)\.\s*$"),
    re.compile(r"^The patient has a diagnosis of (.+?)\.\s*$"),
    re.compile(r"^The patient's diagnosis is (.+?)\.\s*$"),
    re.compile(r"^The patient suffers from (.+?)\.\s*$"),
    re.compile(r"^(.+?) was confirmed in the patient\.\s*$"),
    re.compile(r"^(.+?) is identified in the patient\.\s*$"),
    re.compile(r"^Doctors determined (.+?) for the patient\.\s*$"),
]
_DRUG_PATTERNS = [
    re.compile(r"^Drug (.+?) is administered once daily\.\s*$"),
    re.compile(r"^(.+?) is the patient's daily medication\.\s*$"),
    re.compile(r"^(.+?) is administered to the patient each day\.\s*$"),
    re.compile(r"^(.+?) is the patient's medication\.\s*$"),
    re.compile(r"^Each day, the patient takes (.+?)\.\s*$"),
    re.compile(r"^The patient takes (.+?) every day\.\s*$"),
    re.compile(r"^(.+?) is prescribed for daily use\.\s*$"),
]


def derive_supervision(text: str) -> dict:
    """Map a statement to its typed supervision dict, across all canonical and
    paraphrased templates.

    Returns a dict with `type` and whatever args/values are relevant; missing
    fields are absent. Unrecognized texts fall back to ``filler``.
    """
    text = text.strip()
    if text in _FILLERS:
        return {"type": "filler"}

    for pat in _TOTAL_PATTERNS:
        if m := pat.match(text):
            return {"type": "total_count", "value": float(m.group(1))}
    for pat in _DIRECT_PATTERNS:
        if m := pat.match(text):
            return {"type": "direct_enrolled", "value": float(m.group(1))}
    for pat, mode in _ARM_PATTERNS:
        if m := pat.match(text):
            if mode == "arm_then_n":
                arm_kw, n = m.group(1), m.group(2)
            else:
                n, arm_kw = m.group(1), m.group(2)
            arm = 0 if arm_kw == "treatment" else (1 if arm_kw == "control" else 0)
            return {"type": "arm_count", "arm": arm, "value": float(n)}
    for pat in _BEFORE_PATTERNS:
        if m := pat.match(text):
            return {"type": "before", "entity1": m.group(1).strip(), "entity2": m.group(2).strip()}
    for pat in _BEFORE_REVERSED_PATTERNS:
        if m := pat.match(text):
            return {"type": "before", "entity1": m.group(2).strip(), "entity2": m.group(1).strip()}
    for pat in _GREATER_PATTERNS:
        if m := pat.match(text):
            return {"type": "greater", "entity1": m.group(1).strip(), "entity2": m.group(2).strip()}
    for pat in _GREATER_REVERSED_PATTERNS:
        if m := pat.match(text):
            return {"type": "greater", "entity1": m.group(2).strip(), "entity2": m.group(1).strip()}
    for pat in _PRES_NEG_PATTERNS:
        if m := pat.match(text):
            return {"type": "presence_neg", "symptom": m.group(1).strip()}
    for pat in _PRES_POS_PATTERNS:
        if m := pat.match(text):
            return {"type": "presence_pos", "symptom": m.group(1).strip()}
    for pat in _DIAGNOSIS_PATTERNS:
        if m := pat.match(text):
            return {"type": "diagnosis", "diagnosis": m.group(1).strip()}
    for pat in _DRUG_PATTERNS:
        if m := pat.match(text):
            return {"type": "drug_active", "drug": m.group(1).strip()}
    return {"type": "filler"}


def _entities_to_tag(supervision: dict) -> list[tuple[str, str]]:
    """Return [(surface_text, entity_type)] for tagging this statement."""
    t = supervision.get("type")
    if t == "before":
        return [(supervision.get("entity1", ""), "EVENT"),
                (supervision.get("entity2", ""), "EVENT")]
    if t == "greater":
        return [(supervision.get("entity1", ""), "NAME"),
                (supervision.get("entity2", ""), "NAME")]
    if t in ("presence_pos", "presence_neg"):
        return [(supervision.get("symptom", ""), "SYMPTOM")]
    if t == "diagnosis":
        return [(supervision.get("diagnosis", ""), "DIAGNOSIS")]
    if t == "drug_active":
        return [(supervision.get("drug", ""), "DRUG")]
    return []


def derive_bio_labels(text: str, supervision: dict, offsets: list[tuple[int, int]]) -> list[int]:
    """Per-token BIO label list. `offsets` is the tokenizer's offset_mapping
    for the given `text` (token character spans). Returns one tag index per
    token, defaulting to ``O`` for tokens not inside any entity span."""
    n = len(offsets)
    labels = [BIO_TAG_IDX["O"]] * n
    text_lower = text.lower()
    used_ranges: list[tuple[int, int]] = []

    for entity_text, entity_type in _entities_to_tag(supervision):
        if not entity_text:
            continue
        ent_lower = entity_text.lower()
        start = 0
        char_start = char_end = None
        while True:
            idx = text_lower.find(ent_lower, start)
            if idx == -1:
                break
            ce = idx + len(entity_text)
            if not any(rs < ce and idx < re_ for rs, re_ in used_ranges):
                char_start, char_end = idx, ce
                break
            start = idx + 1
        if char_start is None:
            continue
        used_ranges.append((char_start, char_end))

        first_tok = last_tok = None
        for i, (ts, te) in enumerate(offsets):
            if ts is None or te is None:
                continue
            if te <= char_start:
                continue
            if ts >= char_end:
                break
            if first_tok is None:
                first_tok = i
            last_tok = i
        if first_tok is not None:
            labels[first_tok] = BIO_TAG_IDX[f"B-{entity_type}"]
            for j in range(first_tok + 1, (last_tok if last_tok is not None else first_tok) + 1):
                labels[j] = BIO_TAG_IDX[f"I-{entity_type}"]
    return labels


def decode_bio_spans(text: str, tag_indices: list[int],
                     offsets: list[tuple[int, int]]) -> list[dict]:
    """Greedy BIO decoder: return a list of {type, start, end, text} spans."""
    spans: list[dict] = []
    current: dict | None = None
    for i, ti in enumerate(tag_indices):
        if i >= len(offsets):
            break
        tag = BIO_TAGS[ti]
        ts, te = offsets[i]
        if ts is None or te is None or tag == "O":
            if current is not None:
                spans.append(current)
                current = None
            continue
        prefix, ent = tag.split("-")
        if prefix == "B" or current is None or current["type"] != ent:
            if current is not None:
                spans.append(current)
            current = {"type": ent, "start": int(ts), "end": int(te)}
        else:
            current["end"] = int(te)
    if current is not None:
        spans.append(current)
    for s in spans:
        s["text"] = text[s["start"]:s["end"]].strip()
    return spans


def predicted_supervision(pred: dict) -> dict:
    """Convert PerceptionModule logits + value to a supervision-dict (per stmt)."""
    t = STATEMENT_TYPES[pred["type"]]
    if t == "filler":
        return {"type": "filler"}
    if t == "before":
        return {"type": "before",
                "entity1": ORDER_EVENTS[pred["order_event1"]],
                "entity2": ORDER_EVENTS[pred["order_event2"]]}
    if t == "greater":
        return {"type": "greater",
                "entity1": NAMES[pred["name1"]],
                "entity2": NAMES[pred["name2"]]}
    if t == "arm_count":
        return {"type": "arm_count", "arm": int(pred["arm"]), "value": float(pred["value"])}
    if t == "total_count":
        return {"type": "total_count", "value": float(pred["value"])}
    if t == "direct_enrolled":
        return {"type": "direct_enrolled", "value": float(pred["value"])}
    if t in ("presence_pos", "presence_neg"):
        return {"type": t, "symptom": SYMPTOMS[pred["symptom"]]}
    if t == "diagnosis":
        return {"type": "diagnosis", "diagnosis": DIAGNOSES[pred["diagnosis"]]}
    if t == "drug_active":
        return {"type": "drug_active", "drug": DRUGS[pred["drug"]]}
    return {"type": "filler"}
