"""Per-case failure analysis (skeleton).

Will load ``failure_cases.json`` from an experiment directory, classify each
failure (type mis-prediction vs entity arg vs ordering vs numeric value vs
equilibrium-residual gap), and emit a summary + figure. Implemented in
Stage 8 of the BeliefNet refactor.
"""
from __future__ import annotations

import argparse
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", required=True)
    ap.add_argument("--out", default=None)
    ap.parse_args()
    raise NotImplementedError(
        "Failure analysis runner pending: implement in Stage 8 "
        "after eval/failure_analysis.py lands."
    )


if __name__ == "__main__":
    try:
        main()
    except NotImplementedError as exc:
        print(str(exc))
        sys.exit(1)
