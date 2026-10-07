"""Dual benchmark packaging: controlled-synthetic + naturalistic + real-seeded.

Builds three instantiations of the emergent-inconsistency benchmark and exports
them to JSONL with a datasheet:

* ``canonical``   - single surface form (the controlled probe for the theorem);
* ``paraphrase``  - varied surface forms (tests extractor robustness);
* ``real_seeded`` - synthetic contradiction cores embedded among *real* document
                    sentences as distractors (bridge toward real data).

Supply your own document sentences via ``document_pool`` for a domain-specific
real-seeded set; the embedded fixture makes the path runnable/testable offline.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from beliefnet.data.emergent import generate_emergent
from beliefnet.data.synthetic.case_schema import Case

# Small embedded fixture of real-style sentences (distractors).
REAL_FIXTURE_SENTENCES = [
    "The randomized controlled trial was pre-registered on ClinicalTrials.gov.",
    "Informed consent was obtained from all participants prior to enrollment.",
    "Adverse events were monitored throughout the twelve-week study period.",
    "The primary endpoint was assessed by a blinded independent committee.",
    "Baseline demographic characteristics were balanced across study arms.",
    "Statistical analyses were performed using a pre-specified protocol.",
    "The manuscript was reviewed by two independent subject-matter experts.",
    "Funding was provided by a peer-reviewed national research grant.",
]


def build_benchmark(kappa_values=(2, 3, 5), n_total_values=(6, 12),
                    n_per_cell: int = 20, seed: int = 0,
                    document_pool: list[str] | None = None) -> dict[str, list[Case]]:
    return {
        "canonical": generate_emergent(kappa_values, n_total_values,
                                       n_per_cell=n_per_cell, seed=seed),
        "paraphrase": generate_emergent(kappa_values, n_total_values,
                                        n_per_cell=n_per_cell, seed=seed + 1,
                                        surface="paraphrase"),
        "real_seeded": generate_emergent(
            kappa_values, n_total_values, n_per_cell=n_per_cell, seed=seed + 2,
            surface="paraphrase",
            distractor_pool=document_pool or REAL_FIXTURE_SENTENCES),
    }


def export_jsonl(cases: list[Case], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for c in cases:
            fh.write(json.dumps(c.to_json()) + "\n")
    return len(cases)


def datasheet(named: dict[str, list[Case]]) -> list[dict]:
    rows = []
    for name, cases in named.items():
        labels = Counter(c.label for c in cases)
        hops = Counter(c.metadata["hop_count"] for c in cases)
        rows.append({
            "variant": name,
            "n_cases": len(cases),
            "consistent": labels.get("consistent", 0),
            "inconsistent": labels.get("inconsistent", 0),
            "hop_counts": dict(sorted(hops.items())),
        })
    return rows


def scifact_document_pool(corpus_dir: str | Path) -> list[str]:
    """Real SciFact abstract sentences, usable as real-text distractors.

    Pass the result as ``document_pool`` to ``build_benchmark`` to make the
    ``real_seeded`` variant embed synthetic contradiction cores among genuine
    biomedical abstract sentences.
    """
    from beliefnet.data.scifact.loader import read_corpus
    corpus = read_corpus(Path(corpus_dir) / "corpus.jsonl")
    sentences: list[str] = []
    for doc in corpus.documents.values():
        sentences.extend(doc.get("abstract", []))
    return sentences
