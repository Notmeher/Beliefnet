"""Numeric global-sum stress: does LLM joint-consistency detection collapse as
the number of numeric terms (depth) grows, while the exact checker stays perfect?

    PYTHONPATH=src python scripts/run_numeric_stress.py --models gpt4o gpt5
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np

from beliefnet.data.numeric_emergent import generate_numeric
from beliefnet.eval.metrics import accuracy, auroc
from beliefnet.eval.reporting import build_details_md, save_result
from beliefnet.eval.stats import bootstrap_ci
from beliefnet.llm.azure_client import client_for, load_dotenv, model_available
from beliefnet.reasoning.exact_checker import is_satisfiable


def oracle_score(case) -> float:
    return 0.0 if is_satisfiable(case.metadata["constraints"]) else 1.0


def _llm_scores(model: str, cases) -> np.ndarray | None:
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
    ap.add_argument("--depths", type=int, nargs="+", default=[5, 10, 15, 20])
    ap.add_argument("--n-distract", type=int, default=2)
    ap.add_argument("--n-per-cell", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--models", nargs="*", default=[])
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    load_dotenv()
    cases = generate_numeric(tuple(args.depths), args.n_distract,
                             args.n_per_cell, args.seed)
    labels = np.array([1 if c.label == "inconsistent" else 0 for c in cases])
    orac = np.array([oracle_score(c) for c in cases])
    oracle_acc = accuracy(labels, (orac >= 0.5).astype(int))
    print(f"[oracle] accuracy={oracle_acc:.4f} on {len(cases)} cases")

    systems: dict[str, np.ndarray] = {"oracle": orac}
    for m in args.models:
        sc = _llm_scores(m, cases)
        if sc is not None:
            systems[m] = sc

    idx_by_depth: dict[int, list[int]] = defaultdict(list)
    for i, c in enumerate(cases):
        idx_by_depth[c.metadata["depth"]].append(i)

    names = list(systems)
    rows = []
    for d in sorted(idx_by_depth):
        ix = np.array(idx_by_depth[d])
        row = {"depth": d, "n": int(ix.size)}
        for s in names:
            row[s] = round(float(auroc(systems[s][ix], labels[ix])), 4)
        rows.append(row)

    ci_rows = []
    for s in names:
        mean, lo, hi = bootstrap_ci(systems[s], labels, n_boot=1000, seed=0)
        ci_rows.append({"system": s, "overall_auroc": round(mean, 4),
                        "ci95_low": round(lo, 4), "ci95_high": round(hi, 4)})

    payload = {
        "title": "Numeric global-sum stress -- many-term arithmetic consistency",
        "question": "Does LLM joint-consistency AUROC collapse as the number of "
                    "numeric terms (depth) grows, while the exact checker stays "
                    "perfect?",
        "config": {"depths": args.depths, "n_distract": args.n_distract,
                   "n_per_cell": args.n_per_cell, "seed": args.seed,
                   "systems": ",".join(names)},
        "summary": {"oracle_accuracy": round(float(oracle_acc), 4),
                    "n_cases": len(cases)},
        "tables": {"AUROC by depth": rows, "Overall AUROC (95% CI)": ci_rows},
        "notes": ["Emergent: no pair of statements conflicts; only the global "
                  "sum reveals inconsistency. A* signal if LLM AUROC declines "
                  "with depth while the oracle stays 1.000."],
    }
    save_result(args.out, "exp_numeric_stress" + (f"_{args.tag}" if args.tag else ""), payload)
    build_details_md(args.out)
    for s in names:
        print(f"[numeric] {s} AUROC by depth: {[r[s] for r in rows]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
