"""Experiment (P4): online BeliefMonitor — streaming detection + localization.

Streams each case's statements through the incremental ``BeliefMonitor`` and
reports, by hop-count, (a) detection AUROC and (b) localization accuracy: on
detected inconsistent cases, whether the flagged statement is the one that
actually closed the contradiction (a constraint-bearing statement).

    PYTHONPATH=src python scripts/run_online_monitor.py
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np

from beliefnet.data.emergent import generate_emergent
from beliefnet.eval.metrics import auroc
from beliefnet.eval.reporting import build_details_md, save_result
from beliefnet.perception.local_extract import rule_supervisions
from beliefnet.reasoning.monitor import BeliefMonitor


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kappa", type=int, nargs="+", default=[2, 3, 5])
    ap.add_argument("--n-total", type=int, nargs="+", default=[6, 12])
    ap.add_argument("--n-per-cell", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    cases = generate_emergent(tuple(args.kappa), tuple(args.n_total),
                              n_per_cell=args.n_per_cell, seed=args.seed)
    labels = np.array([1 if c.label == "inconsistent" else 0 for c in cases])

    detect = np.zeros(len(cases))
    localized = {}
    for i, c in enumerate(cases):
        sups = rule_supervisions(c.statements)
        mon = BeliefMonitor()
        mon.run(c.statements, sups)
        detect[i] = 1.0 if mon.flagged_index is not None else 0.0
        if c.label == "inconsistent" and mon.flagged_index is not None:
            flagged_sup = sups[mon.flagged_index]
            localized[i] = flagged_sup.get("type", "filler") in ("before", "greater")

    idx_by_hop: dict[int, list[int]] = defaultdict(list)
    for i, c in enumerate(cases):
        idx_by_hop[c.metadata["hop_count"]].append(i)

    rows = []
    for k in sorted(idx_by_hop):
        ix = np.array(idx_by_hop[k])
        loc = [localized[i] for i in ix if i in localized]
        rows.append({
            "hop_count": k, "n": int(ix.size),
            "detection_auroc": round(float(auroc(detect[ix], labels[ix])), 4),
            "localization_acc": round(float(np.mean(loc)), 4) if loc else None,
        })

    all_loc = list(localized.values())
    payload = {
        "title": "Experiment P4 — Online BeliefMonitor (detect + localize)",
        "question": "Streaming: does the monitor detect emergent inconsistency "
                    "and flag the exact statement that closes it?",
        "config": {"kappa": args.kappa, "n_total": args.n_total,
                   "n_per_cell": args.n_per_cell, "seed": args.seed},
        "summary": {"n_cases": len(cases),
                    "overall_localization_acc":
                        round(float(np.mean(all_loc)), 4) if all_loc else None},
        "tables": {"Detection + localization by hop-count": rows},
        "notes": ["The monitor flags the first statement that makes the "
                  "accumulated belief set unsatisfiable -- runtime localization "
                  "batch NLI cannot provide."],
    }
    save_result(args.out, "expP4_online_monitor", payload)
    build_details_md(args.out)
    print(f"[P4] overall localization acc = {payload['summary']['overall_localization_acc']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
