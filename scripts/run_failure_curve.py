"""Experiment 1 (make-or-break): the emergent-inconsistency failure curve.

Scores a controlled (hop-count x n_statements) grid with (a) the exact symbolic
oracle (sanity: perfect) and (b) optional Azure GPT-4o / GPT-5 judges, then
records per-cell and per-hop AUROC so we can see whether frontier-LLM detection
collapses as hop-count and statement count grow.

    PYTHONPATH=src python scripts/run_failure_curve.py                 # oracle only
    PYTHONPATH=src python scripts/run_failure_curve.py --models gpt4o gpt5
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

from beliefnet.data.emergent import generate_emergent
from beliefnet.eval.metrics import accuracy, auroc
from beliefnet.eval.reporting import build_details_md, save_result
from beliefnet.llm.azure_client import load_dotenv
from beliefnet.reasoning.exact_checker import is_satisfiable

_REASONING_MODELS = {"gpt5", "o1", "o3", "o4"}


def oracle_score(case) -> float:
    return 0.0 if is_satisfiable(case.metadata["constraints"]) else 1.0


def _llm_scores(model: str, cases) -> np.ndarray | None:
    from beliefnet.llm.azure_client import client_for, model_available
    from beliefnet.llm.judge import judge_consistency
    if not model_available(model):
        print(f"[skip] {model}: Azure env vars not set")
        return None
    client = client_for(model)
    scores = np.zeros(len(cases))
    for i, c in enumerate(cases):
        scores[i] = judge_consistency(client, c.statements)["p_inconsistent"]
        if (i + 1) % 10 == 0:
            print(f"  [{model}] {i + 1}/{len(cases)}")
    return scores


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kappa", type=int, nargs="+", default=[2, 3, 5])
    ap.add_argument("--n-total", type=int, nargs="+", default=[4, 8, 16])
    ap.add_argument("--n-per-cell", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--models", nargs="*", default=[])
    ap.add_argument("--surface", default="canonical",
                    help="canonical | paraphrase")
    ap.add_argument("--tag", default="",
                    help="suffix for the experiment id (keeps runs separate)")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    load_dotenv()
    cases = generate_emergent(tuple(args.kappa), tuple(args.n_total),
                              n_per_cell=args.n_per_cell, seed=args.seed,
                              surface=args.surface)
    labels = np.array([1 if c.label == "inconsistent" else 0 for c in cases])
    orac = np.array([oracle_score(c) for c in cases])
    oracle_acc = accuracy(labels, (orac >= 0.5).astype(int))
    print(f"[oracle] accuracy={oracle_acc:.4f} on {len(cases)} cases")

    systems: dict[str, np.ndarray] = {"oracle": orac}
    for m in args.models:
        sc = _llm_scores(m, cases)
        if sc is not None:
            systems[m] = sc

    idx_by_cell: dict[tuple[int, int], list[int]] = defaultdict(list)
    idx_by_hop: dict[int, list[int]] = defaultdict(list)
    for i, c in enumerate(cases):
        k, n = c.metadata["hop_count"], c.metadata["n_statements"]
        idx_by_cell[(k, n)].append(i)
        idx_by_hop[k].append(i)

    names = list(systems)
    rows_cell = []
    for (k, n) in sorted(idx_by_cell):
        ix = np.array(idx_by_cell[(k, n)])
        row = {"hop_count": k, "n_statements": n, "n": int(ix.size)}
        for s in names:
            row[s] = round(float(auroc(systems[s][ix], labels[ix])), 4)
        rows_cell.append(row)

    rows_hop = []
    for k in sorted(idx_by_hop):
        ix = np.array(idx_by_hop[k])
        row = {"hop_count": k}
        for s in names:
            row[s] = round(float(auroc(systems[s][ix], labels[ix])), 4)
        rows_hop.append(row)

    tag = f" [{args.tag}]" if args.tag else ""
    payload = {
        "title": f"Experiment 1 — Emergent-inconsistency failure curve{tag}",
        "question": "Does joint-consistency AUROC collapse as hop-count and "
                    "statement count grow?",
        "config": {"kappa": args.kappa, "n_total": args.n_total,
                   "n_per_cell": args.n_per_cell, "seed": args.seed,
                   "surface": args.surface, "systems": ",".join(names)},
        "summary": {"oracle_accuracy": round(float(oracle_acc), 4),
                    "n_cases": len(cases)},
        "tables": {"AUROC by cell": rows_cell,
                   "Mean AUROC by hop-count": rows_hop},
        "notes": [
            "Thesis supported if oracle stays ~1.000 across hop-counts while LLM "
            "AUROC declines as hop_count grows.",
        ] if len(names) > 1 else [
            "LLM columns not yet populated; re-run with --models gpt4o gpt5 "
            "(Azure creds in .env).",
        ],
    }
    name = "exp1_failure_curve" + (f"_{args.tag}" if args.tag else "")
    exp_dir = save_result(args.out, name, payload).parent
    with open(exp_dir / "failure_curve.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_cell[0]))
        w.writeheader()
        w.writerows(rows_cell)
    build_details_md(args.out)
    print(f"[exp1] wrote {exp_dir}/result.json and refreshed details_results.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
