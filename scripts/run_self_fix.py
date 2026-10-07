"""Experiment 7: the LLM self-fix.

Compares, per hop-count, the AUROC of:

* ``oracle``               - exact checker on the ground-truth graph (ceiling);
* ``rule->checker``        - rule-extracted graph -> exact checker (offline);
* ``<model>_judge``        - the LLM asked to judge global consistency directly;
* ``<model>_extract->checker`` - the LLM's *local* per-statement extractions
                             routed through the exact checker.

The headline: the same model that fails ``_judge`` as hop-count grows can still
supply the local extractions that ``_extract->checker`` turns into a correct
global decision — decomposition + verifiable reasoning fixes the LLM's own
blind spot.

    PYTHONPATH=src python scripts/run_self_fix.py                    # offline ceiling
    PYTHONPATH=src python scripts/run_self_fix.py --models gpt4o gpt5
"""
from __future__ import annotations

import argparse
import os
from collections import defaultdict

import numpy as np

from beliefnet.data.emergent import generate_emergent
from beliefnet.eval.metrics import auroc
from beliefnet.eval.reporting import build_details_md, save_result
from beliefnet.graph.graph_builder import build_graph_from_supervisions
from beliefnet.llm.azure_client import load_dotenv
from beliefnet.perception.local_extract import rule_supervisions
from beliefnet.reasoning.exact_checker import is_satisfiable

_REASONING_MODELS = {"gpt5", "o1", "o3", "o4"}


def _checker_scores(cases, sups_fn) -> np.ndarray:
    scores = np.zeros(len(cases))
    for i, c in enumerate(cases):
        try:
            _, cons = build_graph_from_supervisions(sups_fn(c.statements))
            scores[i] = 0.0 if is_satisfiable(cons) else 1.0
        except Exception:
            scores[i] = 0.0
    return scores


def _oracle_scores(cases) -> np.ndarray:
    return np.array([0.0 if is_satisfiable(c.metadata["constraints"]) else 1.0
                     for c in cases])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kappa", type=int, nargs="+", default=[2, 3, 5])
    ap.add_argument("--n-total", type=int, nargs="+", default=[6, 12])
    ap.add_argument("--n-per-cell", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--models", nargs="*", default=[])
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    load_dotenv()
    cases = generate_emergent(tuple(args.kappa), tuple(args.n_total),
                              n_per_cell=args.n_per_cell, seed=args.seed)
    labels = np.array([1 if c.label == "inconsistent" else 0 for c in cases])

    systems: dict[str, np.ndarray] = {
        "oracle": _oracle_scores(cases),
        "rule->checker": _checker_scores(cases, rule_supervisions),
    }

    for m in args.models:
        from beliefnet.llm.azure_client import client_for, model_available
        from beliefnet.llm.extractor import llm_supervisions
        from beliefnet.llm.judge import judge_consistency
        if not model_available(m):
            print(f"[skip] {m}: Azure env vars not set")
            continue
        client = client_for(m)
        print(f"  [{m}] judging {len(cases)} cases + local extraction ...")
        systems[f"{m}_judge"] = np.array(
            [judge_consistency(client, c.statements)["p_inconsistent"] for c in cases])
        systems[f"{m}_extract->checker"] = _checker_scores(
            cases, lambda st: llm_supervisions(client, st))

    idx_by_hop: dict[int, list[int]] = defaultdict(list)
    for i, c in enumerate(cases):
        idx_by_hop[c.metadata["hop_count"]].append(i)

    names = list(systems)
    rows = []
    for k in sorted(idx_by_hop):
        ix = np.array(idx_by_hop[k])
        row = {"hop_count": k, "n": int(ix.size)}
        for s in names:
            row[s] = round(float(auroc(systems[s][ix], labels[ix])), 4)
        rows.append(row)

    payload = {
        "title": "Experiment 7 — LLM self-fix (decompose + verify)",
        "question": "Can local LLM extraction + a sound checker recover the "
                    "emergent inconsistency the same LLM misjudges end-to-end?",
        "config": {"kappa": args.kappa, "n_total": args.n_total,
                   "n_per_cell": args.n_per_cell, "seed": args.seed,
                   "systems": ",".join(names)},
        "summary": {"n_cases": len(cases)},
        "tables": {"AUROC by hop-count": rows},
        "notes": [
            "Self-fix holds if <model>_extract->checker stays high while "
            "<model>_judge falls as hop_count grows.",
        ] if len(names) > 2 else [
            "Offline ceiling only (oracle + rule->checker). Add --models gpt4o "
            "gpt5 for the LLM judge-vs-extract self-fix contrast.",
        ],
    }
    save_result(args.out, "exp7_self_fix", payload)
    build_details_md(args.out)
    print(f"[exp7] systems: {', '.join(names)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
