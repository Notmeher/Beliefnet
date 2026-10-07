import numpy as np

from beliefnet.eval.metrics import (
    accuracy,
    auroc,
    binary_f1,
    expected_calibration_error,
    macro_f1,
)


def test_auroc_perfect():
    assert auroc(np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1])) == 1.0


def test_auroc_inverse():
    assert auroc(np.array([0.9, 0.8, 0.2, 0.1]), np.array([0, 0, 1, 1])) == 0.0


def test_auroc_all_ties_is_half():
    assert abs(auroc(np.array([0.5, 0.5, 0.5, 0.5]), np.array([0, 1, 0, 1])) - 0.5) < 1e-9


def test_auroc_single_class_is_nan():
    assert np.isnan(auroc(np.array([0.1, 0.2]), np.array([1, 1])))


def test_accuracy():
    assert abs(accuracy(np.array([0, 1, 2]), np.array([0, 1, 1])) - 2 / 3) < 1e-9


def test_macro_f1_perfect():
    y = np.array([0, 1, 2, 0])
    assert macro_f1(y, y, 3) == 1.0


def test_binary_f1():
    # tp=1, fp=0, fn=1 -> precision=1, recall=0.5, f1=2/3
    assert abs(binary_f1(np.array([1, 1, 0, 0]), np.array([1, 0, 0, 0])) - 2 / 3) < 1e-9


def test_ece_bounds_and_zero():
    # perfectly calibrated: confidence equals accuracy in each bin
    conf = np.array([1.0, 1.0, 0.0, 0.0])
    corr = np.array([1, 1, 0, 0])
    assert expected_calibration_error(conf, corr, n_bins=10) == 0.0
    e = expected_calibration_error(np.array([0.9, 0.8, 0.7]), np.array([1, 0, 0]), n_bins=5)
    assert 0.0 <= e <= 1.0
