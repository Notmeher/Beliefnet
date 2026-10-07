"""Certificate auditability: reviewer agreement + review-item export.

Supports the human-study protocol: render each flagged contradiction with its
certificate (unsat core + repair) and, optionally, an LLM free-text rationale,
so reviewers can judge which is more auditable. ``cohen_kappa`` scores
inter-rater agreement.
"""
from __future__ import annotations

import numpy as np


def cohen_kappa(rater_a, rater_b) -> float:
    a = np.asarray(rater_a)
    b = np.asarray(rater_b)
    po = float((a == b).mean())
    labels = set(a.tolist()) | set(b.tolist())
    pe = sum(float((a == l).mean()) * float((b == l).mean()) for l in labels)
    return 1.0 if pe >= 1.0 else float((po - pe) / (1.0 - pe))


def build_review_items(cases, decisions) -> list[dict]:
    """Pair each case with its BeliefNet decision for blind review."""
    items = []
    for c, d in zip(cases, decisions):
        items.append({
            "id": c.id,
            "statements": c.statements,
            "flagged_inconsistent": bool(d.get("inconsistent", False)),
            "certificate": [f"{x['type']}{x['args']}" for x in d.get("unsat_core", [])],
            "repair": [f"{x['type']}{x['args']}" for x in d.get("repair", [])],
            "llm_rationale": d.get("rationale", ""),
            "gold_label": c.label,
        })
    return items
