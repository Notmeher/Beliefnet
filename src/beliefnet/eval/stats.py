"""Statistical utilities for the evaluation program (numpy only).

* ``bootstrap_ci``            - bootstrap mean + CI of any metric;
* ``auroc_difference_test``  - paired bootstrap test on an AUROC gap;
* ``mcnemar_test``           - McNemar's test on paired correctness;
* ``benjamini_hochberg``     - BH false-discovery-rate control.

A paired-bootstrap AUROC test is used in place of DeLong: it is assumption-
light, easy to verify, and standard for AUROC deltas.
"""
from __future__ import annotations

import math

import numpy as np

from beliefnet.eval.metrics import auroc


def bootstrap_ci(scores, labels, metric=auroc, n_boot: int = 1000,
                 alpha: float = 0.05, seed: int = 0):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    n = len(scores)
    stats = []
    for _ in range(n_boot):
        b = rng.integers(0, n, n)
        try:
            v = metric(scores[b], labels[b])
        except Exception:
            v = np.nan
        stats.append(v)
    stats = np.array(stats, dtype=float)
    mean = float(np.nanmean(stats))
    lo, hi = np.nanpercentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return mean, float(lo), float(hi)


def auroc_difference_test(scores_a, scores_b, labels, n_boot: int = 1000,
                          seed: int = 0):
    """Paired bootstrap: returns (mean_diff = AUROC_a - AUROC_b, two-sided p)."""
    scores_a = np.asarray(scores_a, dtype=float)
    scores_b = np.asarray(scores_b, dtype=float)
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    n = len(labels)
    diffs = []
    for _ in range(n_boot):
        b = rng.integers(0, n, n)
        try:
            diffs.append(auroc(scores_a[b], labels[b]) - auroc(scores_b[b], labels[b]))
        except Exception:
            diffs.append(np.nan)
    diffs = np.array(diffs, dtype=float)
    mean_diff = float(np.nanmean(diffs))
    frac_pos = float(np.nanmean(diffs > 0))
    p = 2.0 * min(frac_pos, 1.0 - frac_pos)
    return mean_diff, max(0.0, min(1.0, p))


def mcnemar_test(correct_a, correct_b):
    """McNemar chi-square (continuity-corrected) on paired 0/1 correctness."""
    a = np.asarray(correct_a).astype(bool)
    b = np.asarray(correct_b).astype(bool)
    n01 = int((~a & b).sum())
    n10 = int((a & ~b).sum())
    if n01 + n10 == 0:
        return 0.0, 1.0
    stat = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    p = math.erfc(math.sqrt(stat / 2.0))  # chi-square df=1 survival
    return float(stat), float(p)


def benjamini_hochberg(pvals, alpha: float = 0.05):
    """Return a boolean array of rejections under BH FDR control."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    thresh = alpha * (np.arange(1, m + 1) / m)
    passed = p[order] <= thresh
    reject = np.zeros(m, dtype=bool)
    if passed.any():
        kmax = np.max(np.where(passed))
        reject[order[:kmax + 1]] = True
    return reject
