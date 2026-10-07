"""SciFact zero-shot runner.

Loads the SciFact corpus into BeliefNet ``Case`` objects via
``beliefnet.data.scifact.loader.load_scifact`` and reports basic counts.
The actual zero-shot pipeline (frozen backbone -> trained synthetic
perception heads -> frozen ``StructuredEquilibriumCore``) is the headline
real-data deliverable; this script lands the load+split scaffolding so
that work has a typed entry point.

If ``--corpus`` does not contain ``corpus.jsonl`` the script prints a clear
download hint and exits 1, so calling it on a fresh clone is harmless.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from beliefnet.data.scifact.loader import load_scifact


def _corpus_present(corpus_dir: Path) -> bool:
    return (corpus_dir / "corpus.jsonl").exists()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/scifact",
                    help="directory containing corpus.jsonl and claims_*.jsonl")
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--out", default="experiments/scifact_zero_shot")
    args = ap.parse_args()

    corpus_dir = Path(args.corpus)
    if not _corpus_present(corpus_dir):
        print(f"[scifact-zero-shot] SciFact corpus not found at {corpus_dir}.")
        print("  Download the release from https://github.com/allenai/scifact")
        print(f"  and extract it so that {corpus_dir}/corpus.jsonl exists.")
        return 1

    splits = load_scifact(corpus_dir, splits=tuple(args.splits))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, cases in splits.items():
        n_support = sum(c.conflict_subtype == "scifact_support" for c in cases)
        n_contra = sum(c.conflict_subtype == "scifact_contradict" for c in cases)
        n_nei = sum(c.conflict_subtype == "scifact_no_evidence" for c in cases)
        print(f"  [{name}] cases={len(cases):5d}  "
              f"SUPPORT={n_support} CONTRADICT={n_contra} NEI={n_nei}")

    print("[scifact-zero-shot] zero-shot scoring pipeline not yet wired up; "
          "Stage 8 deliverable.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
