"""Experiment 4: pairwise baselines vs emergent inconsistency.

The all-pairs symbolic judge checks every 2-subset of the extracted constraints
and flags inconsistency iff any pair conflicts -- the exact analogue of an
all-pairs NLI aggregator. It is the empirical counterpart to the separation
theorem: it detects kappa = 2 but must miss kappa >= 3. Bootstrap CIs are
reported. An optional neural all-pairs NLI backend (transformers) can be added
with --nli when torch is available.

    PYTHONPATH=src python scripts/run_baselines.py
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from itertools import combinations

import numpy as np

from beliefnet.data.emergent import generate_emergent
from beliefnet.eval.metrics import auroc
from beliefnet.eval.reporting import build_details_md, save_result
from beliefnet.eval.stats import bootstrap_ci
from beliefnet.graph.graph_builder import build_graph_from_supervisions
from beliefnet.perception.local_extract import rule_supervisions
from beliefnet.reasoning.exact_checker import is_satisfiable


def all_pairs_symbolic(cases) -> np.ndarray:
    scores = np.zeros(len(cases))
    for i, c in enumerate(cases):
        _, cons = build_graph_from_supervisions(rule_supervisions(c.statements))
        flag = 0.0
        for a, b in combinations(cons, 2):
            try:
                if not is_satisfiable([a, b]):
                    flag = 1.0
                    break
            except Exception:
                pass
        scores[i] = flag
    return scores


def oracle_scores(cases) -> np.ndarray:
    return np.array([0.0 if is_satisfiable(c.metadata["constraints"]) else 1.0
                     for c in cases])


def all_pairs_nli(cases, model_name: str) -> np.ndarray:
    """Neural all-pairs NLI baseline (needs torch + transformers).

    Runs an NLI model on every ordered statement pair; the case score is the max
    contradiction probability over pairs. Detects kappa=2 pairwise conflicts
    but -- like any pairwise judge -- must miss emergent kappa>=3 cycles.
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).eval()
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    contra_idx = next(i for i, l in id2label.items() if "contra" in l.lower())

    scores = np.zeros(len(cases))
    for ci, c in enumerate(cases):
        s = c.statements
        pairs = [(s[i], s[j]) for i in range(len(s)) for j in range(len(s)) if i != j]
        maxp = 0.0
        for k in range(0, len(pairs), 32):
            batch = pairs[k:k + 32]
            enc = tok([p for p, _ in batch], [h for _, h in batch],
                      return_tensors="pt", padding=True, truncation=True,
                      max_length=128)
            with torch.no_grad():
                probs = model(**enc).logits.softmax(-1)[:, contra_idx]
            maxp = max(maxp, float(probs.max()))
        scores[ci] = maxp
        if (ci + 1) % 25 == 0:
            print(f"  [nli] {ci + 1}/{len(cases)}")
    return scores


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kappa", type=int, nargs="+", default=[2, 3, 5])
    ap.add_argument("--n-total", type=int, nargs="+", default=[6, 12])
    ap.add_argument("--n-per-cell", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nli", action="store_true",
                    help="add neural DeBERTa-NLI all-pairs baseline (needs torch)")
    ap.add_argument("--nli-model", default="microsoft/deberta-large-mnli")
    ap.add_argument("--tag", default="",
                    help="suffix for the experiment id (keeps runs separate)")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    cases = generate_emergent(tuple(args.kappa), tuple(args.n_total),
                              n_per_cell=args.n_per_cell, seed=args.seed)
    labels = np.array([1 if c.label == "inconsistent" else 0 for c in cases])
    systems = {"oracle": oracle_scores(cases),
               "all_pairs_symbolic": all_pairs_symbolic(cases)}
    if args.nli:
        try:
            systems["all_pairs_nli"] = all_pairs_nli(cases, args.nli_model)
        except Exception as e:  # torch/transformers absent or model download fails
            print(f"[nli] skipped ({type(e).__name__}: {e})")

    idx_by_hop: dict[int, list[int]] = defaultdict(list)
    for i, c in enumerate(cases):
        idx_by_hop[c.metadata["hop_count"]].append(i)

    rows = []
    for k in sorted(idx_by_hop):
        ix = np.array(idx_by_hop[k])
        row = {"hop_count": k, "n": int(ix.size)}
        for s, sc in systems.items():
            row[f"{s}_auroc"] = round(float(auroc(sc[ix], labels[ix])), 4)
        rows.append(row)

    ci_rows = []
    for s, sc in systems.items():
        mean, lo, hi = bootstrap_ci(sc, labels, n_boot=1000, seed=0)
        ci_rows.append({"system": s, "overall_auroc": round(mean, 4),
                        "ci95_low": round(lo, 4), "ci95_high": round(hi, 4)})

    payload = {
        "title": "Experiment 4 — Pairwise baseline vs emergent inconsistency",
        "question": "Does an all-pairs (pairwise-factored) judge miss emergent "
                    "(kappa >= 3) contradictions, as the theorem predicts?",
        "config": {"kappa": args.kappa, "n_total": args.n_total,
                   "n_per_cell": args.n_per_cell, "seed": args.seed},
        "summary": {"n_cases": len(cases)},
        "tables": {"AUROC by hop-count": rows,
                   "Overall AUROC with 95% bootstrap CI": ci_rows},
        "notes": ["all_pairs_symbolic should be ~1.0 at kappa=2 and ~0.5 "
                  "(chance) at kappa>=3 -- the empirical separation result."],
    }
    save_result(args.out, "exp4_pairwise_baseline" + (f"_{args.tag}" if args.tag else ""), payload)
    build_details_md(args.out)
    print(f"[exp4] all-pairs AUROC by hop: "
          f"{[r['all_pairs_symbolic_auroc'] for r in rows]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
