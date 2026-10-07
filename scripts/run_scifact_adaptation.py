"""SciFact adaptation runner.

Fine-tunes the perception and span heads on SciFact while keeping the
``StructuredEquilibriumCore`` frozen. Stage 7 wires the loader; the
adaptation training loop itself is a follow-on (the synthetic-trained
perception needs SciFact-tailored supervision before adaptation makes
sense, and that is a research item, not a refactor item).

If ``--corpus`` does not contain ``corpus.jsonl`` the script prints a clear
download hint and exits 1.
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
    ap.add_argument("--out", default="experiments/scifact_adaptation")
    args = ap.parse_args()

    corpus_dir = Path(args.corpus)
    if not _corpus_present(corpus_dir):
        print(f"[scifact-adaptation] SciFact corpus not found at {corpus_dir}.")
        print("  Download the release from https://github.com/allenai/scifact")
        print(f"  and extract it so that {corpus_dir}/corpus.jsonl exists.")
        return 1

    splits = load_scifact(corpus_dir, splits=tuple(args.splits))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    total = sum(len(cs) for cs in splits.values())
    print(f"[scifact-adaptation] loaded {total} cases across "
          f"splits={list(splits.keys())}; adaptation loop pending Stage 8.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
