"""Load the SciFact corpus into BeliefNet ``Case`` objects.

The SciFact release ships as:

* ``claims_{train,dev,test}.jsonl`` - one claim per line, with fields
  ``id``, ``claim``, ``cited_doc_ids`` (list[int]), and ``evidence``
  (dict mapping ``doc_id`` -> list of ``{sentences: [int], label: "SUPPORT"|"CONTRADICT"}``).
* ``corpus.jsonl`` - one document per line with ``doc_id`` (int),
  ``title`` (str), ``abstract`` (list[str]).

This loader does NOT download. It expects the user to have already obtained
the corpus and points to the directory. The split files are tolerated as
missing (test labels are held back upstream) so callers can pre-flight
without the test set.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from beliefnet.data.scifact.claim_evidence import claim_evidence_to_case
from beliefnet.data.synthetic.case_schema import Case

CORPUS_FILENAME = "corpus.jsonl"
SPLIT_FILES = {
    "train": "claims_train.jsonl",
    "val": "claims_dev.jsonl",
    "test": "claims_test.jsonl",
}


@dataclass
class SciFactCorpus:
    """Parsed SciFact corpus indexed by ``doc_id``."""
    documents: dict[int, dict]

    def abstract(self, doc_id: int) -> list[str]:
        doc = self.documents.get(int(doc_id))
        if doc is None:
            return []
        return list(doc.get("abstract", []))


def read_corpus(corpus_path: Path) -> SciFactCorpus:
    docs: dict[int, dict] = {}
    with open(corpus_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            docs[int(rec["doc_id"])] = rec
    return SciFactCorpus(documents=docs)


def _claim_to_cases(rec: dict, corpus: SciFactCorpus, split: str) -> list[Case]:
    claim_id = rec["id"]
    claim = rec["claim"]
    cited = rec.get("cited_doc_ids", [])
    evidence = rec.get("evidence", {}) or {}

    cases: list[Case] = []
    seen_doc_ids = set()
    for doc_id_key, ev_list in evidence.items():
        doc_id = int(doc_id_key)
        seen_doc_ids.add(doc_id)
        abstract = corpus.abstract(doc_id)
        if not abstract:
            continue
        for k, ev in enumerate(ev_list):
            label = ev.get("label")
            sent_idx = ev.get("sentences") or []
            case = claim_evidence_to_case(
                case_id=f"scifact-{claim_id}-{doc_id}-{k}",
                claim=claim,
                abstract_sentences=abstract,
                evidence_label=label,
                evidence_sentence_idx=sent_idx,
            )
            case.split = split
            cases.append(case)
    # Cited abstracts without an evidence annotation become NEI cases.
    for doc_id in cited:
        if int(doc_id) in seen_doc_ids:
            continue
        abstract = corpus.abstract(int(doc_id))
        if not abstract:
            continue
        case = claim_evidence_to_case(
            case_id=f"scifact-{claim_id}-{doc_id}-nei",
            claim=claim,
            abstract_sentences=abstract,
            evidence_label=None,
        )
        case.split = split
        cases.append(case)
    return cases


def load_scifact(corpus_dir: str | Path,
                 splits: tuple[str, ...] = ("train", "val")) -> dict[str, list[Case]]:
    """Load SciFact splits from ``corpus_dir``.

    Args:
        corpus_dir: directory containing ``corpus.jsonl`` and ``claims_*.jsonl``.
        splits: which splits to load. ``"test"`` is silently skipped if the
            test claim file is absent (SciFact withholds it).

    Raises:
        FileNotFoundError: if the corpus file or a requested non-test split
            file is missing.
    """
    corpus_dir = Path(corpus_dir)
    corpus_path = corpus_dir / CORPUS_FILENAME
    if not corpus_path.exists():
        raise FileNotFoundError(
            f"SciFact corpus not found at {corpus_path}. "
            "Download the release from https://github.com/allenai/scifact "
            "and extract it to this directory."
        )
    corpus = read_corpus(corpus_path)

    out: dict[str, list[Case]] = {}
    for split in splits:
        if split not in SPLIT_FILES:
            raise ValueError(f"unknown split: {split!r}; expected one of {list(SPLIT_FILES)}")
        path = corpus_dir / SPLIT_FILES[split]
        if not path.exists():
            if split == "test":
                continue
            raise FileNotFoundError(
                f"SciFact split file not found at {path}. "
                f"Expected {SPLIT_FILES[split]} alongside corpus.jsonl."
            )
        cases: list[Case] = []
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                cases.extend(_claim_to_cases(json.loads(line), corpus, split))
        out[split] = cases
    return out
