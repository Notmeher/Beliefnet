"""Multi-paraphrase relation prototypes used by the prototype-similarity head.

For each fine-grained statement type (matches ``STATEMENT_TYPES``), define a
small set of *concrete* paraphrase examples. Encoded with the frozen Qwen
backbone they become FIXED reference points. The ``PrototypeRelationParser``
classifies a new statement by max-similarity to these reference points in a
learned shared projection space.

The prototype templates here are drawn ONLY from CANONICAL + PARAPHRASED
template pools (never from HELD_OUT_TEMPLATES), so the held-out paraphrase
test evaluates genuine surface-form extrapolation rather than peeking at the
test distribution through the prototypes.
"""
from __future__ import annotations


RELATION_PROTOTYPES: dict[str, list[str]] = {
    "filler": [
        "The patient was referred by their primary care physician.",
        "The study was approved by the ethics board.",
        "The patient has held private insurance for years.",
        "The trial was funded by a national research agency.",
        "The data were recorded in the hospital system.",
    ],
    "before": [
        "the biopsy occurred before surgery.",
        "the MRI scan preceded discharge.",
        "admission happened earlier than the second dose.",
        "the consultation was observed prior to the lab test.",
        "the CT scan occurred before rehabilitation.",
    ],
    "greater": [
        "Alice is older than Brian.",
        "Carla is senior to David.",
        "Elena has more years than Frank.",
        "Grace is older than Henry.",
        "Karen has more years than Liam.",
    ],
    "arm_count": [
        "The treatment arm enrolled 40 patients.",
        "The control arm had 30 participants.",
        "45 patients were assigned to the treatment arm.",
        "The control arm enrolled 50 patients.",
        "The treatment arm had 80 participants.",
    ],
    "total_count": [
        "The trial enrolled 75 patients in total.",
        "A total of 100 patients participated in the trial.",
        "The trial recruited 80 patients in all.",
        "The trial enrolled 200 patients in total.",
        "A total of 150 patients participated in the trial.",
    ],
    "direct_enrolled": [
        "The trial enrolled 120 patients.",
        "The study included 200 participants.",
        "150 subjects were enrolled in the study.",
        "The trial enrolled 350 patients.",
        "The study included 80 participants.",
    ],
    "presence_pos": [
        "The patient has fever.",
        "The patient reported chest pain.",
        "The patient exhibits nausea.",
        "The patient presents with a persistent cough.",
        "The patient has fatigue.",
    ],
    "presence_neg": [
        "The patient does not have fever.",
        "The patient lacks a persistent cough.",
        "The patient shows no sign of nausea.",
        "The patient does not have chest pain.",
        "The patient lacks joint pain.",
    ],
    "diagnosis": [
        "The patient was diagnosed with influenza.",
        "The patient has a diagnosis of pneumonia.",
        "The patient's diagnosis is bronchitis.",
        "The patient was diagnosed with COVID-19.",
        "The patient has a diagnosis of asthma.",
    ],
    "drug_active": [
        "Drug aspirin is administered once daily.",
        "ibuprofen is the patient's daily medication.",
        "metformin is administered to the patient each day.",
        "Drug atorvastatin is administered once daily.",
        "amoxicillin is the patient's daily medication.",
    ],
}


def relation_names() -> list[str]:
    return list(RELATION_PROTOTYPES.keys())
