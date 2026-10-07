"""Residual readout: per-case energy buckets -> single inconsistency logit.

The readout is intentionally tiny (Linear(4, 1)) so the headline result is
attributable to the equilibrium mechanism and the typed constraints, not to
classifier capacity. The bias is initialised so positive residual maps to
"inconsistent" before any training, matching the structural intent of the
energy.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


def _inv_softplus(x: float) -> Tensor:
    return torch.log(torch.expm1(torch.tensor(float(x))))


class ResidualReadout(nn.Module):
    """``log1p`` of the four residual buckets -> scalar inconsistency logit."""

    def __init__(self):
        super().__init__()
        self.readout = nn.Linear(4, 1)
        with torch.no_grad():
            self.readout.weight.fill_(0.5)
            self.readout.bias.fill_(-1.0)

    def forward(self, residuals: dict[str, Tensor]) -> Tensor:
        feats = torch.stack([
            torch.log1p(residuals["anchor"]),
            torch.log1p(residuals["hinge"]),
            torch.log1p(residuals["sum"]),
            torch.log1p(residuals["total"]),
        ], dim=-1)
        return self.readout(feats).squeeze(-1)
