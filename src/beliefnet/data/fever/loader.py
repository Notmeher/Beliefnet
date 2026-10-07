"""Load the FEVER corpus into BeliefNet ``Case`` objects (skeleton).

FEVER ships as ``train.jsonl`` / ``shared_task_dev.jsonl`` / ``shared_task_test.jsonl``
with a separate Wikipedia sentence corpus. Each claim record has fields
``id``, ``label`` (``"SUPPORTS"`` / ``"REFUTES"`` / ``"NOT ENOUGH INFO"``),
``claim``, and ``evidence`` (list of annotation tuples
``[annotation_id, evidence_id, wiki_page, sentence_id]``).

This is the skeleton in Stage 7 of the BeliefNet refactor: schema is
locked in, the actual sentence-corpus lookup is left for the SciFact-first
real-data pilot. The function signature is parallel to
``beliefnet.data.scifact.loader.load_scifact`` so callers can switch sources
with a config change.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from beliefnet.data.synthetic.case_schema import Case

CLAIM_FILES = {
    "train": "train.jsonl",
    "val": "shared_task_dev.jsonl",
    "test": "shared_task_test.jsonl",
}


@dataclass
class FeverCorpus:
    """Parsed FEVER wiki corpus (skeleton).

    The full FEVER corpus is large (~5M sentences across ~190k Wikipedia
    pages); the productive choice at adaptation time is to lazily fetch
    only the pages referenced by the loaded claim split, not eagerly hold
    everything in memory. That fetcher lives here when it lands.
    """
    pages_root: Path

    def sentences(self, wiki_page: str) -> list[str]:
        raise NotImplementedError(
            "FEVER wiki-page lookup is a real-data pilot deliverable; "
            "implement once SciFact zero-shot has stabilised."
        )


def load_fever(corpus_dir: str | Path,
               splits: tuple[str, ...] = ("train", "val")) -> dict[str, list[Case]]:
    """Load FEVER splits from ``corpus_dir`` (skeleton).

    Raises:
        NotImplementedError: until the wiki-sentence lookup is wired up.
    """
    raise NotImplementedError(
        "FEVER loader pending: wiki-sentence corpus mapper not yet "
        "implemented. Schema is fixed; see CLAIM_FILES and FeverCorpus."
    )
