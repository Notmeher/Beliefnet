"""Baseline probe over frozen features.

This same MLP doubles, in later steps, as the *matched-capacity* control that
BeliefNet must beat: if a plain MLP on the frozen features matches BeliefNet,
the energy/equilibrium machinery is not justified.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch.nn as nn
from torch import Tensor


class MLPProbe(nn.Module):
    def __init__(self, in_dim: int, hidden: Sequence[int] = (256,),
                 n_classes: int = 3, dropout: float = 0.1):
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
