"""Numpy metrics for BeliefNet evaluation.

Covers the synthetic-ladder metric list: classification F1 (macro + per
class), AUROC for binary detection (with tie-aware Mann-Whitney U), Expected
Calibration Error.
"""
from __future__ import annotations

import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((np.asarray(y_true) == np.asarray(y_pred)).mean())


def _avg_ranks(x: np.ndarray) -> np.ndarray:
    """Average ranks (1-based), ties share the mean rank."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    sx = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def auroc(scores: np.ndarray, binary_labels: np.ndarray) -> float:
    """AUROC via the Mann-Whitney U statistic (tie-aware)."""
    y = np.asarray(binary_labels).astype(int)
    s = np.asarray(scores, dtype=float)
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _avg_ranks(s)
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def per_class_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> list[float]:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    out = []
    for c in range(n_classes):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        out.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return out


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    return float(np.mean(per_class_f1(y_true, y_pred, n_classes)))


def binary_f1(y_true_bin: np.ndarray, y_pred_bin: np.ndarray) -> float:
    y_true_bin, y_pred_bin = np.asarray(y_true_bin), np.asarray(y_pred_bin)
    tp = int(((y_pred_bin == 1) & (y_true_bin == 1)).sum())
    fp = int(((y_pred_bin == 1) & (y_true_bin == 0)).sum())
    fn = int(((y_pred_bin == 0) & (y_true_bin == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0


def expected_calibration_error(confidences: np.ndarray, correct: np.ndarray,
                               n_bins: int = 15) -> float:
    conf = np.asarray(confidences, dtype=float)
    corr = np.asarray(correct, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece, n = 0.0, len(conf)
    for b in range(n_bins):
        lo, hi = bins[b], bins[b + 1]
        mask = (conf > lo) & (conf <= hi) if b > 0 else (conf >= lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        ece += abs(corr[mask].mean() - conf[mask].mean()) * mask.sum() / n
    return float(ece)
