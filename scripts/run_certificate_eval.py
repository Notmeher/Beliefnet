"""Experiment 8: certificate quality (unsat core + repair).

For each inconsistent case, computes the minimal unsatisfiable core (proof of
contradiction) and the minimal correction set (repair), and verifies both are
valid: the core is itself unsatisfiable, and removing the correction set
restores satisfiability. Reports core / correction size distributions and one
worked example, so the auditable-artifact claim is quantified, not asserted.

    PYTHONPATH=src python scripts/run_certificate_eval.py
"""
from __future__ import annotations

import argparse
from collections import defaultdict

from beliefnet.data.emergent import generate_emergent
from beliefnet.eval.reporting import build_details_md, save_result
from beliefnet.reasoning.certificates import (
    minimal_correction_set,
    minimal_unsat_core,
)
from beliefnet.reasoning.exact_checker import is_satisfiable


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
    inconsistent = [c for c in cases if c.label == "inconsistent"]

    by_hop: dict[int, list[tuple[int, int]]] = defaultdict(list)
    valid_core = valid_fix = 0
    example = None
    for c in inconsistent:
        cons = c.metadata["constraints"]
        core = minimal_unsat_core(cons)
        mcs = minimal_correction_set(cons)
        core_ok = not is_satisfiable(core)
        fix_ok = is_satisfiable([x for x in cons if x not in mcs])
        valid_core += int(core_ok)
        valid_fix += int(fix_ok)
        by_hop[c.metadata["hop_count"]].append((len(core), len(mcs)))
        if example is None and c.metadata["hop_count"] >= 3:
            example = {
                "statements": " | ".join(c.statements),
                "unsat_core": "; ".join(f"{x['type']}{x['args']}" for x in core),
                "repair (retract)": "; ".join(
                    f"{x['type']}{x['args']}" for x in mcs),
            }

    rows = []
    for k in sorted(by_hop):
        cs = [c for c, _ in by_hop[k]]
        ms = [m for _, m in by_hop[k]]
        rows.append({"hop_count": k, "n_cases": len(cs),
                     "mean_core_size": round(sum(cs) / len(cs), 3),
                     "mean_repair_size": round(sum(ms) / len(ms), 3)})

    n = len(inconsistent)
    payload = {
        "title": "Experiment 8 — Certificate quality (unsat core + repair)",
        "question": "Are the emitted contradiction proofs and repairs valid and "
                    "minimal?",
        "config": {"kappa": args.kappa, "n_total": args.n_total,
                   "n_per_cell": args.n_per_cell, "seed": args.seed},
        "summary": {"n_inconsistent": n,
                    "valid_core_rate": round(valid_core / n, 4),
                    "valid_repair_rate": round(valid_fix / n, 4)},
        "tables": {"Core / repair sizes by hop-count": rows,
                   "Example certificate": [example] if example else []},
        "notes": ["Every flagged contradiction ships with a checkable proof "
                  "(unsat core) and a minimal repair — artifacts no score-only "
                  "detector provides."],
    }
    save_result(args.out, "exp8_certificates", payload)
    build_details_md(args.out)
    print(f"[exp8] valid core rate={valid_core / n:.4f} "
          f"valid repair rate={valid_fix / n:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
