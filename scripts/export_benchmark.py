"""Export the dual benchmark (synthetic + naturalistic + real-seeded) to JSONL.

Writes each variant to ``data/benchmark/<variant>.jsonl`` and records a
datasheet under ``results/`` so the benchmark composition appears in the report.

    PYTHONPATH=src python scripts/export_benchmark.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

from beliefnet.data.benchmark import (
    build_benchmark,
    datasheet,
    export_jsonl,
    scifact_document_pool,
)
from beliefnet.eval.reporting import build_details_md, save_result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-cell", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-dir", default="data/benchmark")
    ap.add_argument("--scifact-corpus", default=None,
                    help="use real SciFact abstract sentences as distractors")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    pool = scifact_document_pool(args.scifact_corpus) if args.scifact_corpus else None
    named = build_benchmark(n_per_cell=args.n_per_cell, seed=args.seed,
                            document_pool=pool)
    data_dir = Path(args.data_dir)
    exported = []
    for name, cases in named.items():
        n = export_jsonl(cases, data_dir / f"{name}.jsonl")
        exported.append({"variant": name, "path": str(data_dir / f"{name}.jsonl"),
                         "n_cases": n})

    sheet = datasheet(named)
    payload = {
        "title": "Benchmark — dual (synthetic + naturalistic + real-seeded)",
        "question": "What does the released emergent-inconsistency benchmark "
                    "contain?",
        "config": {"n_per_cell": args.n_per_cell, "seed": args.seed,
                   "data_dir": args.data_dir},
        "summary": {"variants": len(named),
                    "total_cases": sum(r["n_cases"] for r in exported)},
        "tables": {"Datasheet": sheet, "Exported files": exported},
        "notes": ["real_seeded embeds synthetic contradiction cores among real "
                  "document sentences; swap in your corpus via document_pool."],
    }
    save_result(args.out, "benchmark_datasheet", payload)
    build_details_md(args.out)
    print(f"[benchmark] exported {sum(r['n_cases'] for r in exported)} cases "
          f"to {data_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
