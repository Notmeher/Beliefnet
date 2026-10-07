"""Tests for the residual readout (Linear(4,1) on log1p of residual buckets)."""
from __future__ import annotations

import torch

from beliefnet.equilibrium.residual_scoring import ResidualReadout, _inv_softplus


def _residuals(anchor, hinge, sm, total=None, batch=1):
    if total is None:
        total = anchor + hinge + sm
    def _t(x):
        return torch.tensor([float(x)] * batch)
    return {"anchor": _t(anchor), "hinge": _t(hinge),
            "sum": _t(sm), "total": _t(total)}


def test_readout_output_shape_matches_batch():
    head = ResidualReadout()
    r = _residuals(1.0, 2.0, 3.0, batch=5)
    out = head(r)
    assert out.shape == (5,)


def test_zero_residuals_give_bias_only():
    head = ResidualReadout()
    out = head(_residuals(0.0, 0.0, 0.0))
    # log1p(0) = 0, so the readout collapses to its bias.
    assert torch.allclose(out, head.readout.bias.detach())


def test_initial_sign_is_inconsistent_for_positive_residual():
    """Initial bias is -1, weight is 0.5; for residual e.g. 5, the logit is
    positive (i.e. 'inconsistent before training')."""
    head = ResidualReadout()
    out = head(_residuals(5.0, 0.0, 0.0))
    assert float(out[0]) > 0.0


def test_gradients_flow_through_readout():
    head = ResidualReadout()
    r = _residuals(1.0, 1.0, 1.0)
    loss = head(r).sum()
    loss.backward()
    assert head.readout.weight.grad is not None
    assert head.readout.bias.grad is not None
    assert head.readout.weight.grad.abs().sum() > 0


def test_inv_softplus_roundtrip():
    """`softplus(inv_softplus(x)) ≈ x` for the equilibrium margin initialization."""
    x = 0.7
    y = torch.nn.functional.softplus(_inv_softplus(x))
    assert abs(float(y) - x) < 1e-5
