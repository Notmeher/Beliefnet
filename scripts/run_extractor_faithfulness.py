"""Experiment 6: local-extractor faithfulness (text -> typed constraints).

Measures how well each extractor recovers the oracle constraint graph, at the
constraint level (micro precision / recall / F1), broken down by hop-count.
The rule extractor runs offline; the LLM extractor (GPT-4o / GPT-5) is added
when --models is given and Azure credentials are present.

    PYTHONPATH=src python scripts/run_extractor_faithfulness.py
    PYTHONPATH=src python scripts/run_extractor_faithfulness.py --models gpt4o gpt5
"""
from __future__ import annotations

import argparse
import os
from collections import defaultdict

from beliefnet.data.emergent import generate_emergent
from beliefnet.eval.reporting import build_details_md, save_result
from beliefnet.graph.graph_builder import build_graph_from_supervisions
from beliefnet.llm.azure_client import load_dotenv
from beliefnet.perception.local_extract import rule_supervisions

_REASONING_MODELS = {"gpt5", "o1", "o3", "o4"}


def _cset(constraints: list[dict]) -> set[tuple]:
    return {(c["type"], tuple(c["args"])) for c in constraints}


def _score_backend(cases, sups_fn) -> list[dict]:
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    counts = defaultdict(int)
    for c in cases:
        gold = _cset(c.metadata["constraints"])
        try:
            _, pred_cons = build_graph_from_supervisions(sups_fn(c.statements))
            pred = _cset(pred_cons)
        except Exception:
            pred = set()
        k = c.metadata["hop_count"]
        counts[k] += 1
        tp[k] += len(gold & pred)
        fp[k] += len(pred - gold)
        fn[k] += len(gold - pred)

    rows = []
    for k in sorted(counts):
        p = tp[k] / max(tp[k] + fp[k], 1)
        r = tp[k] / max(tp[k] + fn[k], 1)
        f = 2 * p * r / max(p + r, 1e-9)
        rows.append({"hop_count": k, "n_cases": counts[k],
                     "precision": round(p, 4), "recall": round(r, 4),
                     "f1": round(f, 4)})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kappa", type=int, nargs="+", default=[2, 3, 5])
    ap.add_argument("--n-total", type=int, nargs="+", default=[6, 12])
    ap.add_argument("--n-per-cell", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--models", nargs="*", default=[])
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    load_dotenv()
    cases = generate_emergent(tuple(args.kappa), tuple(args.n_total),
                              n_per_cell=args.n_per_cell, seed=args.seed)

    tables = {"rule extractor — P/R/F1 by hop-count": _score_backend(cases, rule_supervisions)}
    summary = {"n_cases": len(cases)}
    rule_f1 = [r["f1"] for r in tables["rule extractor — P/R/F1 by hop-count"]]
    summary["rule_mean_f1"] = round(sum(rule_f1) / max(len(rule_f1), 1), 4)

    for m in args.models:
        from beliefnet.llm.azure_client import client_for, model_available
        from beliefnet.llm.extractor import llm_supervisions
        if not model_available(m):
            print(f"[skip] {m}: Azure env vars not set")
            continue
        client = client_for(m)
        rows = _score_backend(cases, lambda st: llm_supervisions(client, st))
        tables[f"{m} extractor — P/R/F1 by hop-count"] = rows
        f1s = [r["f1"] for r in rows]
        summary[f"{m}_mean_f1"] = round(sum(f1s) / max(len(f1s), 1), 4)

    payload = {
        "title": "Experiment 6 — Local-extractor faithfulness",
        "question": "How faithfully does each extractor recover the oracle "
                    "typed constraint graph (constraint-level micro F1)?",
        "config": {"kappa": args.kappa, "n_total": args.n_total,
                   "n_per_cell": args.n_per_cell, "seed": args.seed},
        "summary": summary,
        "tables": tables,
        "notes": ["Perception is the bottleneck: end-to-end detection can only "
                  "be as good as constraint-level extraction faithfulness."],
    }
    save_result(args.out, "exp6_extractor_faithfulness", payload)
    build_details_md(args.out)
    print(f"[exp6] rule mean F1 = {summary['rule_mean_f1']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
