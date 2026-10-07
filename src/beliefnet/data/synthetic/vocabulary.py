"""Vocabulary constants shared by the synthetic single-pair and multi-statement
generators.

These lists define the closed entity space the perception heads classify into:
events for temporal relations, names for comparatives, arms for numeric sums,
fillers for distractor padding, and the medical-domain categorical vocabularies
(diagnoses, drugs, symptoms) with a stable code per diagnosis.
"""
from __future__ import annotations

# Medical categorical vocabularies
DIAGNOSES = [
    "influenza", "pneumonia", "bronchitis", "tuberculosis", "asthma",
    "COVID-19", "sinusitis",
]
DRUGS = [
    "aspirin", "ibuprofen", "metformin", "atorvastatin", "amoxicillin",
    "lisinopril", "warfarin", "prednisone",
]
SYMPTOMS = [
    "fever", "a persistent cough", "chest pain", "nausea", "fatigue",
    "shortness of breath", "a severe headache", "a skin rash", "joint pain",
    "dizziness",
]

# Events that can be totally ordered relative to each other (for cycles/chains).
ORDER_EVENTS = [
    "admission", "the biopsy", "surgery", "chemotherapy", "discharge",
    "the follow-up visit", "diagnosis", "rehabilitation", "the MRI scan",
    "the transfusion", "the second dose", "the consultation", "the CT scan",
    "physiotherapy", "the lab test", "the screening",
]
# Order-neutral name pool (no alphabetical/positional prior to exploit).
NAMES = [
    "Alice", "Brian", "Carla", "David", "Elena", "Frank", "Grace", "Henry",
    "Irene", "Jack", "Karen", "Liam", "Mona", "Nate", "Olivia", "Peter",
    "Quinn", "Rosa", "Sam", "Tara", "Uma", "Victor", "Wendy", "Xavier",
]
ARMS = ["the treatment arm", "the control arm",
        "the placebo arm", "the observation arm"]

NEUTRAL_FILLERS = [
    "The patient was referred by their primary care physician.",
    "The study was approved by the ethics board.",
    "The patient has held private insurance for years.",
    "The trial was funded by a national research agency.",
    "The patient was seen in the outpatient clinic.",
    "The data were recorded in the hospital system.",
]

# Stable code for each diagnosis: makes entity_swap an anchor-disagreement on
# a single "diagnosis_slot" entity (codes are scalar latents).
DIAGNOSIS_CODES = {dx: float(i + 1) for i, dx in enumerate(DIAGNOSES)}
