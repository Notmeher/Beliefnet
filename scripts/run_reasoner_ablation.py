"""Experiment 5: exact checker vs equilibrium residual (ablation).

Justifies (or retires) the differentiable equilibrium core: does the soft
residual from ``StructuredEquilibriumCore`` separate consistent from emergent-
inconsistent cases as well as the exact checker? Requires torch; skips cleanly
if torch is unavailable so the rest of the suite still runs.

    PYTHONPATH=src python scripts/run_reasoner_ablation.py
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np

from beliefnet.data.emergent import generate_emergent
from beliefnet.eval.metrics import auroc
from beliefnet.eval.reporting import build_details_md, save_result
from beliefnet.reasoning.exact_checker import is_satisfiable


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kappa", type=int, nargs="+", default=[2, 3, 5])
    ap.add_argument("--n-total", type=int, nargs="+", default=[6, 12])
    ap.add_argument("--n-per-cell", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    try:
        import torch
        from beliefnet.graph.graph_utils import pack_graphs
        from beliefnet.models.beliefnet import StructuredEquilibriumCore
    except ImportError:
        print("[exp5] torch not available; skipping equilibrium ablation.")
        return 0

    cases = generate_emergent(tuple(args.kappa), tuple(args.n_total),
                              n_per_cell=args.n_per_cell, seed=args.seed)
    labels = np.array([1 if c.label == "inconsistent" else 0 for c in cases])
    exact = np.array([0.0 if is_satisfiable(c.metadata["constraints"]) else 1.0
                      for c in cases])

    model = StructuredEquilibriumCore(n_steps=80).eval()
    spec = pack_graphs(cases)
    with torch.no_grad():
        energy = model.per_case_residuals(spec)["total"].cpu().numpy()

    idx_by_hop: dict[int, list[int]] = defaultdict(list)
    for i, c in enumerate(cases):
        idx_by_hop[c.metadata["hop_count"]].append(i)

    rows = []
    for k in sorted(idx_by_hop):
        ix = np.array(idx_by_hop[k])
        rows.append({"hop_count": k, "n": int(ix.size),
                     "exact_auroc": round(float(auroc(exact[ix], labels[ix])), 4),
                     "energy_auroc": round(float(auroc(energy[ix], labels[ix])), 4)})

    payload = {
        "title": "Experiment 5 — Exact checker vs equilibrium residual",
        "question": "Does the differentiable equilibrium residual separate cases "
                    "as well as the exact checker (does the energy layer earn "
                    "its place)?",
        "config": {"kappa": args.kappa, "n_total": args.n_total,
                   "n_per_cell": args.n_per_cell, "seed": args.seed,
                   "equilibrium_steps": 80},
        "summary": {
            "exact_mean_auroc": round(float(np.mean([r["exact_auroc"] for r in rows])), 4),
            "energy_mean_auroc": round(float(np.mean([r["energy_auroc"] for r in rows])), 4),
        },
        "tables": {"AUROC by hop-count": rows},
        "notes": ["Exact gives the certificate + soundness; the energy gives a "
                  "differentiable training signal. If energy ties exact here, "
                  "the energy layer is justified only by its trainability."],
    }
    save_result(args.out, "exp5_reasoner_ablation", payload)
    build_details_md(args.out)
    print(f"[exp5] exact vs energy mean AUROC: "
          f"{payload['summary']['exact_mean_auroc']} / "
          f"{payload['summary']['energy_mean_auroc']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
