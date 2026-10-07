"""Single-pair synthetic contradiction dataset.

Built to probe one question: can a detector recognize contradiction when the
claim and evidence are lexically near-identical? The "hard" contradiction types
(minimal-edit negation/antonym, numeric, temporal, entity-swap) hold surface
form almost constant while flipping the logical relation, so any detector that
leans on embedding dissimilarity will fail on them.

Generation is template-based and fully seeded: every example carries, in its
metadata, exactly which transformation produced it, so the dataset is
reproducible and auditable rather than model-generated noise.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from beliefnet.data.synthetic.vocabulary import (
    DIAGNOSES,
    DRUGS,
    SYMPTOMS,
)

LABELS = ("entailment", "neutral", "contradiction")
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}

CONTRADICTION_TYPES = (
    "entailment",
    "neutral",
    "generic",
    "minimal_edit",
    "numeric",
    "temporal",
    "entity_swap",
)


@dataclass
class Example:
    id: str
    claim: str
    evidence: str
    label: str               # one of LABELS
    contradiction_type: str  # one of CONTRADICTION_TYPES
    difficulty: str          # easy | medium | hard
    metadata: dict[str, Any] = field(default_factory=dict)
    split: str | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def write_jsonl(path: str | Path, examples: Iterable[Example | dict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w") as fh:
        for ex in examples:
            record = ex.to_json() if isinstance(ex, Example) else ex
            fh.write(json.dumps(record) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> list[Example]:
    out: list[Example] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(Example(**json.loads(line)))
    return out

_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


def surface_jaccard(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta and not tb:
        return 1.0
    return len(ta & tb) / len(ta | tb)


def difficulty_for(label: str, jaccard: float) -> str:
    """Difficulty reflects how unusable surface overlap is as a shortcut.

    For contradictions, high overlap is the adversarial case (hard): the
    evidence looks like the claim but means the opposite. Entailment/neutral
    are easy because overlap aligns with (or is irrelevant to) the label.
    """
    if label == "contradiction":
        return "hard" if jaccard >= 0.5 else "medium"
    return "easy"


# ---------------------------------------------------------------------------
# Vocabulary (biomedical-leaning, matching the target domain)
# ---------------------------------------------------------------------------
SYMPTOM_SUPPORT = [
    "Examination confirmed {s} in the patient.",
    "The patient presented with {s}.",
    "Clinical notes documented {s} on admission.",
    "The patient reported {s} during the visit.",
]
NEUTRAL_PATIENT = [
    "The patient is {age} years old.",
    "The patient was referred by their primary care physician.",
    "The patient lives in {city}.",
    "The patient has held private insurance for years.",
]
AGES = ["34", "57", "61", "42", "29", "73"]
CITIES = ["Orlando", "Tampa", "Miami", "Boston"]

OUTCOMES = [
    "mortality", "blood pressure", "recovery time", "infection risk",
    "symptom severity", "hospital readmission",
]
DIRECTION = [
    ("reduce", "reduces", "increases"),
    ("lower", "lowers", "raises"),
    ("improve", "improves", "worsens"),
    ("decrease", "decreases", "elevates"),
]

LOCATIONS = [
    ("left lung", "right lung"),
    ("upper lobe", "lower lobe"),
    ("left kidney", "right kidney"),
    ("anterior wall", "posterior wall"),
]
TEMPORAL_EVENTS = [
    "treatment", "symptom onset", "antibiotic therapy", "the biopsy",
    "the second dose", "physiotherapy", "the MRI scan", "fever onset",
    "the transfusion", "rehabilitation",
]
TEMPORAL_REFS = [
    "diagnosis", "hospital admission", "surgery", "discharge",
    "the first dose", "referral",
]


@dataclass
class SyntheticConfig:
    n_per_type: int = 280
    seed: int = 0
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15)


# ---------------------------------------------------------------------------
# Per-type raw generators. Each yields (claim, evidence, label, ctype, meta).
# ---------------------------------------------------------------------------
def _gen_entailment(rng):
    """Includes high-overlap controls (same number, restated temporal/entity)
    that mirror the hard contradictions but preserve meaning. These are the
    foils: surface-similar yet logically consistent."""
    out = []
    for s in SYMPTOMS:
        for tmpl in SYMPTOM_SUPPORT:
            out.append(("The patient has " + s + ".", tmpl.format(s=s),
                        "entailment", "entailment",
                        {"family": "symptom", "edit": "paraphrase_support"}))
    for d in DRUGS:
        for _, third, _ in DIRECTION:
            for o in OUTCOMES:
                out.append((f"Drug {d} {third} {o}.",
                            f"Overall, {d} {third} {o} in this study.",
                            "entailment", "entailment",
                            {"family": "drug", "edit": "restate"}))
    for dx in DIAGNOSES:
        out.append((f"The patient was diagnosed with {dx}.",
                    f"The diagnosis was confirmed as {dx}.",
                    "entailment", "entailment",
                    {"family": "diagnosis", "edit": "confirm"}))
    for _ in range(120):
        n = rng.randint(40, 500)
        out.append((f"The trial enrolled {n} patients.",
                    f"The trial enrolled {n} patients in total.",
                    "entailment", "entailment",
                    {"family": "trial_size", "edit": "same_number"}))
    for _ in range(40):
        p = rng.randint(10, 90)
        out.append((f"{p} percent of patients responded to therapy.",
                    f"In total, {p} percent of patients responded to therapy.",
                    "entailment", "entailment",
                    {"family": "response_rate", "edit": "same_number"}))
    for event in TEMPORAL_EVENTS:
        for ref in TEMPORAL_REFS:
            out.append((f"The {event} occurred before {ref}.",
                        f"The {event} took place before {ref}.",
                        "entailment", "entailment",
                        {"family": "temporal", "edit": "restate_before"}))
    for a, _b in LOCATIONS:
        out.append((f"The tumor is located in the {a}.",
                    f"Imaging showed the tumor in the {a}.",
                    "entailment", "entailment",
                    {"family": "location", "edit": "same_entity"}))
    return out


def _gen_neutral(rng):
    out = []
    for s in SYMPTOMS:
        for tmpl in NEUTRAL_PATIENT:
            ev = tmpl.format(age=rng.choice(AGES), city=rng.choice(CITIES))
            out.append(("The patient has " + s + ".", ev,
                        "neutral", "neutral",
                        {"family": "symptom", "edit": "unrelated_predicate"}))
    for d in DRUGS:
        for _, third, _ in DIRECTION:
            o = rng.choice(OUTCOMES)
            out.append((f"Drug {d} {third} {o}.",
                        f"Drug {d} is administered once daily.",
                        "neutral", "neutral",
                        {"family": "drug", "edit": "unrelated_predicate"}))
    for dx in DIAGNOSES:
        out.append((f"The patient was diagnosed with {dx}.",
                    "The patient was seen in the outpatient clinic.",
                    "neutral", "neutral",
                    {"family": "diagnosis", "edit": "unrelated_predicate"}))
    for _ in range(120):
        n = rng.randint(40, 500)
        out.append((f"The trial enrolled {n} patients.",
                    "The trial was funded by a national research agency.",
                    "neutral", "neutral",
                    {"family": "trial_size", "edit": "unrelated_predicate"}))
    for event in TEMPORAL_EVENTS:
        out.append((f"The {event} occurred before {rng.choice(TEMPORAL_REFS)}.",
                    "The case was later reviewed by the ethics board.",
                    "neutral", "neutral",
                    {"family": "temporal", "edit": "unrelated_predicate"}))
    return out


def _gen_generic(rng):
    """Reworded contradictions (lower surface overlap)."""
    out = []
    for d in DRUGS:
        for base, third, _ in DIRECTION:
            for o in OUTCOMES:
                out.append((f"Drug {d} {third} {o}.",
                            f"The study found no significant effect of {d} on {o}.",
                            "contradiction", "generic",
                            {"family": "drug", "edit": "reworded_no_effect"}))
    for s in SYMPTOMS:
        out.append(("The patient has " + s + ".",
                    "The patient was found to be in good health with no complaints.",
                    "contradiction", "generic",
                    {"family": "symptom", "edit": "reworded_healthy"}))
    return out


def _gen_minimal_edit(rng):
    """Negation insertion and single-word antonym swaps (max overlap)."""
    out = []
    for s in SYMPTOMS:
        out.append(("The patient has " + s + ".",
                    "The patient does not have " + s + ".",
                    "contradiction", "minimal_edit",
                    {"family": "symptom", "edit": "negation"}))
    for d in DRUGS:
        for base, third, antonym in DIRECTION:
            for o in OUTCOMES:
                out.append((f"Drug {d} {third} {o}.",
                            f"Drug {d} does not {base} {o}.",
                            "contradiction", "minimal_edit",
                            {"family": "drug", "edit": "negation"}))
                out.append((f"Drug {d} {third} {o}.",
                            f"Drug {d} {antonym} {o}.",
                            "contradiction", "minimal_edit",
                            {"family": "drug", "edit": "directional_antonym"}))
    for _ in range(6):
        out.append(("The biopsy result was positive.",
                    "The biopsy result was negative.",
                    "contradiction", "minimal_edit",
                    {"family": "test", "edit": "antonym"}))
    return out


def _gen_numeric(rng):
    out = []

    def pair_int(lo, hi, min_gap):
        a = rng.randint(lo, hi)
        while True:
            b = rng.randint(lo, hi)
            if abs(a - b) >= min_gap:
                return a, b

    def pair_float(lo, hi, min_gap):
        a = round(rng.uniform(lo, hi), 1)
        while True:
            b = round(rng.uniform(lo, hi), 1)
            if abs(a - b) >= min_gap:
                return a, b

    specs = [
        ("The trial enrolled {} patients.", lambda: pair_int(40, 500, 40), "trial_size"),
        ("{} percent of patients responded to therapy.", lambda: pair_int(10, 90, 20), "response_rate"),
        ("The patient received {} milligrams of the drug.", lambda: pair_int(5, 500, 20), "dose"),
        ("The patient's temperature was {} degrees Celsius.", lambda: pair_float(36.5, 41.0, 1.5), "temperature"),
        ("The tumor measured {} centimeters.", lambda: pair_float(1.0, 9.0, 1.5), "tumor_size"),
    ]
    per_spec = 120
    for tmpl, sampler, fam in specs:
        for _ in range(per_spec):
            a, b = sampler()
            out.append((tmpl.format(a), tmpl.format(b),
                        "contradiction", "numeric",
                        {"family": fam, "edit": f"number_{a}_vs_{b}"}))
    return out


def _gen_temporal(rng):
    out = []
    for event in TEMPORAL_EVENTS:
        for ref in TEMPORAL_REFS:
            claim = f"The {event} occurred before {ref}."
            evidence = f"The {event} occurred after {ref}."
            out.append((claim, evidence, "contradiction", "temporal",
                        {"family": "temporal", "edit": "before_vs_after"}))
    return out


def _gen_entity_swap(rng):
    """Mutually-exclusive slot swaps: a single diagnosis/location/cause."""
    out = []
    for i in range(len(DIAGNOSES)):
        for j in range(len(DIAGNOSES)):
            if i == j:
                continue
            out.append((f"The patient was diagnosed with {DIAGNOSES[i]}.",
                        f"The patient was diagnosed with {DIAGNOSES[j]}.",
                        "contradiction", "entity_swap",
                        {"family": "diagnosis", "edit": "swap",
                         "mutually_exclusive": True}))
    for a, b in LOCATIONS:
        out.append((f"The tumor is located in the {a}.",
                    f"The tumor is located in the {b}.",
                    "contradiction", "entity_swap",
                    {"family": "location", "edit": "swap",
                     "mutually_exclusive": True}))
        out.append((f"The tumor is located in the {b}.",
                    f"The tumor is located in the {a}.",
                    "contradiction", "entity_swap",
                    {"family": "location", "edit": "swap",
                     "mutually_exclusive": True}))
    for i in range(len(DRUGS)):
        j = (i + 1) % len(DRUGS)
        out.append((f"The adverse reaction was caused by {DRUGS[i]}.",
                    f"The adverse reaction was caused by {DRUGS[j]}.",
                    "contradiction", "entity_swap",
                    {"family": "cause", "edit": "swap",
                     "mutually_exclusive": True}))
    return out


_GENERATORS = {
    "entailment": _gen_entailment,
    "neutral": _gen_neutral,
    "generic": _gen_generic,
    "minimal_edit": _gen_minimal_edit,
    "numeric": _gen_numeric,
    "temporal": _gen_temporal,
    "entity_swap": _gen_entity_swap,
}


def _stratified_split(examples: list[Example], ratios, rng) -> None:
    by_key: dict[tuple, list[Example]] = defaultdict(list)
    for ex in examples:
        by_key[(ex.label, ex.contradiction_type)].append(ex)
    for items in by_key.values():
        rng.shuffle(items)
        n = len(items)
        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])
        for i, ex in enumerate(items):
            if i < n_train:
                ex.split = "train"
            elif i < n_train + n_val:
                ex.split = "val"
            else:
                ex.split = "test"


def generate(config: SyntheticConfig | None = None) -> list[Example]:
    config = config or SyntheticConfig()
    import random

    rng = random.Random(config.seed)
    raw: list[tuple] = []
    for ctype, gen in _GENERATORS.items():
        items = gen(rng)
        rng.shuffle(items)
        raw.extend(items[: config.n_per_type])

    seen: set[tuple[str, str]] = set()
    examples: list[Example] = []
    idx = 0
    for claim, evidence, label, ctype, meta in raw:
        key = (claim, evidence)
        if key in seen:
            continue
        seen.add(key)
        jac = surface_jaccard(claim, evidence)
        meta = {**meta, "surface_jaccard": round(jac, 4)}
        examples.append(Example(
            id=f"syn-{idx:06d}",
            claim=claim,
            evidence=evidence,
            label=label,
            contradiction_type=ctype,
            difficulty=difficulty_for(label, jac),
            metadata=meta,
        ))
        idx += 1

    _stratified_split(examples, config.ratios, rng)
    return examples
