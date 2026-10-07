import numpy as np

from beliefnet.eval.stats import (
    auroc_difference_test,
    benjamini_hochberg,
    bootstrap_ci,
    mcnemar_test,
)


def test_bootstrap_ci_perfect_predictor():
    y = np.array([0, 0, 1, 1] * 8)
    s = y.astype(float)
    mean, lo, hi = bootstrap_ci(s, y, n_boot=300, seed=0)
    assert mean > 0.95 and lo <= mean <= hi


def test_auroc_difference_perfect_vs_chance():
    y = np.array([0, 0, 1, 1] * 20)
    perfect = y.astype(float)
    chance = np.zeros_like(perfect)
    diff, p = auroc_difference_test(perfect, chance, y, n_boot=300, seed=0)
    assert diff > 0.3 and 0.0 <= p <= 1.0


def test_mcnemar_returns_valid_p():
    stat, p = mcnemar_test(np.array([1, 1, 0, 0]), np.array([1, 0, 1, 0]))
    assert 0.0 <= p <= 1.0


def test_benjamini_hochberg_rejects_small():
    reject = benjamini_hochberg([0.001, 0.02, 0.9], alpha=0.05)
    assert bool(reject[0]) and not bool(reject[2])
