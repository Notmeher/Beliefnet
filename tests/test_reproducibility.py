"""Reproducibility tests: the synthetic data generators must be deterministic
under a fixed seed across re-runs. Catches regressions where module-level
state (e.g. dict ordering) silently changes the RNG trajectory.
"""
from __future__ import annotations

from beliefnet.data.synthetic.case_schema import MultiStmtConfig
from beliefnet.data.synthetic.multi_statement import generate as gen_multi
from beliefnet.data.synthetic.ood import OodConfig, generate_ood
from beliefnet.data.synthetic.single_pair import SyntheticConfig
from beliefnet.data.synthetic.single_pair import generate as gen_single
from beliefnet.data.synthetic.templates import AUGMENTED_TEMPLATES


def _multi(seed: int = 0):
    return gen_multi(MultiStmtConfig(n_per_subtype=40, seed=seed))


def _single(seed: int = 0):
    return gen_single(SyntheticConfig(n_per_type=40, seed=seed))


def _ood(seed: int = 0):
    return generate_ood(OodConfig(n_per_subtype=40, seed=seed,
                                  templates=AUGMENTED_TEMPLATES,
                                  id_prefix="repro"))


def test_multi_statement_identical_first_10_under_same_seed():
    a = _multi()
    b = _multi()
    for i in range(10):
        assert a[i].statements == b[i].statements
        assert a[i].label == b[i].label
        assert a[i].conflict_subtype == b[i].conflict_subtype


def test_single_pair_identical_first_10_under_same_seed():
    a = _single()
    b = _single()
    for i in range(10):
        assert a[i].claim == b[i].claim
        assert a[i].evidence == b[i].evidence
        assert a[i].label == b[i].label


def test_ood_identical_first_10_under_same_seed():
    a = _ood()
    b = _ood()
    for i in range(10):
        assert a[i].statements == b[i].statements
        assert a[i].label == b[i].label


def test_different_seeds_yield_different_data():
    a = _multi(seed=0)
    b = _multi(seed=1)
    differing = sum(1 for x, y in zip(a, b) if x.statements != y.statements)
    assert differing > len(a) // 4  # most cases should differ


def test_multi_statement_split_distribution_is_deterministic():
    a = _multi()
    b = _multi()
    a_splits = [c.split for c in a]
    b_splits = [c.split for c in b]
    assert a_splits == b_splits
