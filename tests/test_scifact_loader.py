"""Tests for the SciFact -> beliefnet.Case loader.

Uses a tiny embedded fixture written to a tmp directory, so the test runs
without the real SciFact corpus and is safe for CI.
"""
from __future__ import annotations

import json

import pytest

from beliefnet.data.scifact.claim_evidence import claim_evidence_to_case
from beliefnet.data.scifact.loader import load_scifact

_CORPUS = [
    {"doc_id": 1, "title": "Aspirin and stroke",
     "abstract": [
         "Aspirin reduces the risk of recurrent stroke.",
         "Trial enrolled 1200 patients aged 50 to 75.",
         "Bleeding events were within expected range.",
     ]},
    {"doc_id": 2, "title": "Vitamin D supplementation",
     "abstract": [
         "Vitamin D supplementation did not reduce all-cause mortality.",
         "The cohort followed 8000 participants over five years.",
     ]},
]

_CLAIMS = [
    {"id": 101, "claim": "Aspirin reduces recurrent stroke risk.",
     "cited_doc_ids": [1],
     "evidence": {"1": [{"sentences": [0], "label": "SUPPORT"}]}},
    {"id": 102, "claim": "Vitamin D reduces all-cause mortality.",
     "cited_doc_ids": [2],
     "evidence": {"2": [{"sentences": [0], "label": "CONTRADICT"}]}},
    {"id": 103, "claim": "Caffeine intake is unrelated to migraines.",
     "cited_doc_ids": [1],
     "evidence": {}},
]


@pytest.fixture
def scifact_dir(tmp_path):
    with open(tmp_path / "corpus.jsonl", "w") as fh:
        for rec in _CORPUS:
            fh.write(json.dumps(rec) + "\n")
    with open(tmp_path / "claims_train.jsonl", "w") as fh:
        for rec in _CLAIMS:
            fh.write(json.dumps(rec) + "\n")
    with open(tmp_path / "claims_dev.jsonl", "w") as fh:
        fh.write(json.dumps(_CLAIMS[0]) + "\n")
    return tmp_path


def test_loader_reads_train_and_val(scifact_dir):
    splits = load_scifact(scifact_dir, splits=("train", "val"))
    assert set(splits.keys()) == {"train", "val"}
    assert len(splits["train"]) == 3
    assert len(splits["val"]) == 1


def test_loader_skips_missing_test_silently(scifact_dir):
    splits = load_scifact(scifact_dir, splits=("train", "test"))
    assert set(splits.keys()) == {"train"}


def test_loader_raises_on_missing_corpus(tmp_path):
    with pytest.raises(FileNotFoundError, match="SciFact corpus not found"):
        load_scifact(tmp_path)


def test_support_claim_maps_to_consistent(scifact_dir):
    splits = load_scifact(scifact_dir, splits=("train",))
    support = next(c for c in splits["train"] if c.id.startswith("scifact-101-"))
    assert support.label == "consistent"
    assert support.conflict_kind == "none"
    assert support.conflict_subtype == "scifact_support"
    assert support.statements[0] == "Aspirin reduces recurrent stroke risk."
    assert support.metadata["source"] == "scifact"


def test_contradict_claim_maps_to_inconsistent(scifact_dir):
    splits = load_scifact(scifact_dir, splits=("train",))
    contradict = next(c for c in splits["train"] if c.id.startswith("scifact-102-"))
    assert contradict.label == "inconsistent"
    assert contradict.conflict_kind == "pairwise"
    assert contradict.conflict_subtype == "scifact_contradict"
    # claim is at index 0; evidence sentence index 0 in the abstract becomes 1.
    assert 0 in contradict.conflict_indices and 1 in contradict.conflict_indices


def test_nei_claim_maps_to_no_evidence(scifact_dir):
    splits = load_scifact(scifact_dir, splits=("train",))
    nei = next(c for c in splits["train"] if c.id.startswith("scifact-103-"))
    assert nei.label == "consistent"
    assert nei.conflict_subtype == "scifact_no_evidence"
    assert nei.conflict_indices == []


def test_claim_evidence_mapper_omits_constraints_until_parser_lands():
    """Real-data BeliefParser is a follow-on; until then metadata.constraints is empty."""
    case = claim_evidence_to_case(
        "scifact-1-1-0",
        claim="X reduces Y.",
        abstract_sentences=["X reduces Y in trials.", "N=1000."],
        evidence_label="SUPPORT",
        evidence_sentence_idx=[0],
    )
    assert case.metadata["constraints"] == []
    assert case.metadata["entities"] == []
    assert case.metadata["n_statements"] == 3
