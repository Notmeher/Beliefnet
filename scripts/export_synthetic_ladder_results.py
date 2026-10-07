"""Aggregate per-step JSON summaries from the synthetic ladder into one CSV.

Walks ``experiments/step*/`` (or a user-supplied root), reads each step's
``*_summary.json`` (when present), and writes a long-format CSV with one row
per (step, test_set, metric).
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

_STEP_RE = re.compile(r"step(\d+)")
_NUMERIC_METRICS = ("type_accuracy", "overall_auroc", "accuracy")


def _iter_summary_files(root: Path):
    for d in sorted(root.glob("step*")):
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*summary.json")):
            yield d.name, f


def _flatten(record: dict, step: str, test_set: str):
    rows = []
    for k in _NUMERIC_METRICS:
        if k in record:
            rows.append({"step": step, "test_set": test_set,
                         "metric": k, "value": record[k]})
    for k, v in record.get("matched_auroc", {}).items():
        if v is None:
            continue
        rows.append({"step": step, "test_set": test_set,
                     "metric": f"matched_auroc.{k}", "value": v})
    rel = record.get("relation_metrics") or {}
    for k in ("both_entities_accuracy", "argument_order_accuracy"):
        if k in rel:
            rows.append({"step": step, "test_set": test_set,
                         "metric": f"relation.{k}", "value": rel[k]})
    span_micro = (record.get("span_f1") or {}).get("micro") or {}
    if "f1" in span_micro:
        rows.append({"step": step, "test_set": test_set,
                     "metric": "span_f1.micro", "value": span_micro["f1"]})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="experiments",
                    help="root directory containing stepNN/ folders")
    ap.add_argument("--out", default="reports/synthetic/ladder_metrics.csv")
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    n_files = 0
    for step, path in _iter_summary_files(root):
        n_files += 1
        try:
            with open(path) as fh:
                blob = json.load(fh)
        except json.JSONDecodeError:
            print(f"  [skip] {path}: invalid JSON")
            continue
        for test_set, rec in blob.items():
            if not isinstance(rec, dict):
                continue
            rows.extend(_flatten(rec, step, test_set))

    fieldnames = ["step", "test_set", "metric", "value"]
    with open(out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[export] read {n_files} summary files; wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
