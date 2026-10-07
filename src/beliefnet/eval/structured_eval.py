"""Matched-AUROC evaluation harness for structured cases.

Same evaluation protocol as ``consistency_eval`` but driven by precomputed
case-level scores, so the symbolic oracle and ``StructuredEquilibriumCore``
use the identical reporting code path.
"""
from __future__ import annotations

import numpy as np

from beliefnet.data.synthetic.case_schema import Case
from beliefnet.eval.metrics import accuracy, auroc, binary_f1


def evaluate_from_scores(scores: np.ndarray, cases: list[Case]) -> dict:
    """Matched-AUROC report driven by case-level scores."""
    y = np.array([1 if c.label == "inconsistent" else 0 for c in cases])
    pred = (scores >= 0.5).astype(int)

    def matched(pos_pred, neg_pred):
        pos = np.array([scores[i] for i, c in enumerate(cases) if pos_pred(c)])
        neg = np.array([scores[i] for i, c in enumerate(cases) if neg_pred(c)])
        if len(pos) == 0 or len(neg) == 0:
            return None
        s = np.concatenate([pos, neg])
        lab = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
        return round(auroc(s, lab), 4)

    flag_rate = {}
    for sub in sorted({c.conflict_subtype for c in cases}):
        idx = [i for i, c in enumerate(cases) if c.conflict_subtype == sub]
        flag_rate[sub] = {"label": cases[idx[0]].label, "n": len(idx),
                          "flagged_inconsistent": round(float(pred[idx].mean()), 4)}

    return {
        "overall_auroc": round(auroc(scores, y), 4),
        "accuracy": round(accuracy(y, pred), 4),
        "f1_inconsistent": round(binary_f1(y, pred), 4),
        "matched_auroc": {
            "temporal_cycle_vs_chain": matched(
                lambda c: c.conflict_subtype == "temporal_cycle",
                lambda c: c.conflict_subtype == "temporal_chain"),
            "comparative_cycle_vs_chain": matched(
                lambda c: c.conflict_subtype == "comparative_cycle",
                lambda c: c.conflict_subtype == "comparative_chain"),
            "numeric_bad_vs_good_sum": matched(
                lambda c: c.conflict_subtype == "numeric_sum" and c.label == "inconsistent",
                lambda c: c.conflict_subtype == "numeric_sum" and c.label == "consistent"),
            "pairwise_vs_independent": matched(
                lambda c: c.conflict_kind == "pairwise",
                lambda c: c.conflict_subtype == "independent"),
        },
        "flag_rate_by_subtype": flag_rate,
    }
