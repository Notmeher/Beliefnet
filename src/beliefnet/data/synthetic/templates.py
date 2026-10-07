"""Template sets for the synthetic generators.

The ``TemplateSet`` dataclass packages a paraphrase list per relation. Three
named sets are predefined:

* ``CANONICAL_TEMPLATES``: the single canonical phrasing per relation used in
  in-distribution training and tests.
* ``PARAPHRASED_TEMPLATES``: paraphrase variants used during augmented training
  (carries the EEN-v9 additions for symptom / diagnosis / drug / arm / total).
* ``HELD_OUT_TEMPLATES``: brand-new paraphrase surface forms the model has
  never seen, used to test paraphrase generalization.
* ``AUGMENTED_TEMPLATES``: ``CANONICAL u PARAPHRASED``.

Direction-reversed paraphrases ("B happened after A" -> means "A before B")
are encoded directly in the template strings by reversing slot order so the
generator emits the reversed surface form with the correct semantics.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TemplateSet:
    """List of paraphrases per relation. Each list is sampled uniformly."""
    temporal_before: list[str]
    comparative_greater: list[str]
    arm0_count: list[str]
    arm1_count: list[str]
    total_count: list[str]
    direct_enrolled: list[str]
    presence_pos: list[str]
    presence_neg: list[str]
    diagnosis: list[str]
    drug_active: list[str]


CANONICAL_TEMPLATES = TemplateSet(
    temporal_before=["{a} occurred before {b}."],
    comparative_greater=["{a} is older than {b}."],
    arm0_count=["The treatment arm enrolled {n} patients."],
    arm1_count=["The control arm enrolled {n} patients."],
    total_count=["The trial enrolled {n} patients in total."],
    direct_enrolled=["The trial enrolled {n} patients."],
    presence_pos=["The patient has {s}.", "The patient reported {s}."],
    presence_neg=["The patient does not have {s}."],
    diagnosis=["The patient was diagnosed with {d}."],
    drug_active=["Drug {d} is administered once daily."],
)

PARAPHRASED_TEMPLATES = TemplateSet(
    temporal_before=[
        "{a} preceded {b}.",
        "{a} happened earlier than {b}.",
        "{a} was observed prior to {b}.",
        # Reversed-direction paraphrases ("B happened after A" means same as
        # "A happened before B"); slot order in the template is reversed so the
        # generator emits the reversed surface form with the correct semantics.
        "{b} happened after {a}.",
        "{b} occurred later than {a}.",
    ],
    comparative_greater=[
        "{a} is senior to {b}.",
        "{a} has more years than {b}.",
        # Reversed-direction comparative paraphrases.
        "{b} is younger than {a}.",
        "{b} is less old than {a}.",
    ],
    arm0_count=[
        "The treatment arm had {n} participants.",
        "{n} patients were assigned to the treatment arm.",
        # EEN-v9: v8-failing held-out templates moved into training so the
        # SpanTagger and the type head learn these surface forms.
        "In the treatment arm, {n} patients were studied.",
    ],
    arm1_count=[
        "The control arm had {n} participants.",
        "{n} patients were assigned to the control arm.",
        "In the control arm, {n} patients were studied.",
    ],
    total_count=[
        "A total of {n} patients participated in the trial.",
        "The trial recruited {n} patients in all.",
        # EEN-v9 additions: the v8 numeric-paraphrase failures.
        "Across both arms, {n} patients were studied.",
        "The combined enrollment was {n} patients.",
    ],
    direct_enrolled=[
        "The study included {n} participants.",
        "{n} subjects were enrolled in the study.",
    ],
    presence_pos=[
        "The patient exhibits {s}.",
        "The patient presents with {s}.",
        # EEN-v9 additions.
        "The patient shows {s}.",
        "{s} was observed in the patient.",
    ],
    presence_neg=[
        "The patient lacks {s}.",
        "The patient shows no sign of {s}.",
        # EEN-v9 additions.
        "The patient is free from {s}.",
        "{s} was not observed in the patient.",
    ],
    diagnosis=[
        "The patient has a diagnosis of {d}.",
        "The patient's diagnosis is {d}.",
        # EEN-v9 additions to fix DIAGNOSIS F1 = 0 on paraphrase.
        "The patient suffers from {d}.",
        "{d} was confirmed in the patient.",
    ],
    drug_active=[
        "{d} is the patient's daily medication.",
        "{d} is administered to the patient each day.",
        # EEN-v9 additions to fix DRUG F1 = 0.255 on paraphrase.
        "{d} is the patient's medication.",
        "Each day, the patient takes {d}.",
    ],
)

# Held-out test paraphrases. EEN-v9 rotates the categorical / numeric held-out
# templates (since the v8-failing ones are now in PARAPHRASED training) so
# this set still tests truly-unseen surface forms. Temporal and comparative
# held-out templates are unchanged (v8 hit 1.000 on them).
HELD_OUT_TEMPLATES = TemplateSet(
    temporal_before=[
        "{a} took place ahead of {b}.",
        "{a} came before {b} in sequence.",
        # Held-out reversed-direction temporal phrasings (entity order reversed
        # in the surface form; still means a-before-b).
        "{b} followed {a} in the timeline.",
        "{b} came after {a}.",
    ],
    comparative_greater=[
        "{a} outranks {b} in age.",
        "{a}'s age exceeds that of {b}.",
        # Held-out reversed comparatives.
        "{b} is the junior of {a}.",
        "{b} ranks below {a} in age.",
    ],
    # EEN-v9 NEW held-out templates for the previously-failing categories.
    arm0_count=[
        "{n} patients made up the treatment arm.",
        "{n} patients were placed in the treatment arm.",
    ],
    arm1_count=[
        "{n} patients made up the control arm.",
        "{n} patients were placed in the control arm.",
    ],
    total_count=[
        "The total trial enrollment was {n}.",
        "Across the entire study, {n} patients enrolled.",
    ],
    direct_enrolled=[
        "The trial received {n} participants.",
        "Across the study, {n} patients took part.",
    ],
    presence_pos=[
        "The patient experiences {s}.",
        "{s} is present in the patient.",
    ],
    presence_neg=[
        "The patient is without {s}.",
        "No {s} was detected in the patient.",
    ],
    diagnosis=[
        "{d} is identified in the patient.",
        "Doctors determined {d} for the patient.",
    ],
    drug_active=[
        "The patient takes {d} every day.",
        "{d} is prescribed for daily use.",
    ],
)


def _union_templates(a: TemplateSet, b: TemplateSet) -> TemplateSet:
    return TemplateSet(
        temporal_before=list(a.temporal_before) + list(b.temporal_before),
        comparative_greater=list(a.comparative_greater) + list(b.comparative_greater),
        arm0_count=list(a.arm0_count) + list(b.arm0_count),
        arm1_count=list(a.arm1_count) + list(b.arm1_count),
        total_count=list(a.total_count) + list(b.total_count),
        direct_enrolled=list(a.direct_enrolled) + list(b.direct_enrolled),
        presence_pos=list(a.presence_pos) + list(b.presence_pos),
        presence_neg=list(a.presence_neg) + list(b.presence_neg),
        diagnosis=list(a.diagnosis) + list(b.diagnosis),
        drug_active=list(a.drug_active) + list(b.drug_active),
    )


AUGMENTED_TEMPLATES = _union_templates(CANONICAL_TEMPLATES, PARAPHRASED_TEMPLATES)
