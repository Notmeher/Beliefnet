"""Rebuild results/details_results.md from all results/*/result.json payloads.

    PYTHONPATH=src python scripts/build_report.py
"""
from __future__ import annotations

import argparse

from beliefnet.eval.reporting import build_details_md


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    out = build_details_md(args.results)
    print(f"[build_report] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
