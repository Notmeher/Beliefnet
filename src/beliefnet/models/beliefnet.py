"""``StructuredEquilibriumCore``: BeliefNet's typed-constraint reasoner.

Composes the three equilibrium modules into one trainable nn.Module:

* per-entity scalar latents ``u``;
* typed constraints (``anchor`` / ``before`` / ``greater`` / ``sum_eq``)
  contribute fixed energy terms via ``beliefnet.equilibrium.energy_functions``;
* equilibrium is found by unrolled GD
  (``beliefnet.equilibrium.solver.unrolled_gradient_descent``);
* the four per-case residual buckets feed a tiny linear readout
  (``beliefnet.equilibrium.residual_scoring.ResidualReadout``) to produce a
  scalar inconsistency logit.

The class is parameter-compatible with the legacy ``EENv2Structured`` (same
``log_margin`` parameter, same ``readout.weight/bias``), so checkpoints from
the synthetic ladder load directly here.
"""
from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from beliefnet.equilibrium.energy_functions import per_case_residuals, total_energy
from beliefnet.equilibrium.residual_scoring import ResidualReadout, _inv_softplus
from beliefnet.equilibrium.solver import unrolled_gradient_descent


class StructuredEquilibriumCore(nn.Module):
    def __init__(self, n_steps: int = 50, lr_inner: float = 0.2,
                 margin_init: float = 1.0):
        super().__init__()
        self.n_steps = n_steps
        self.lr_inner = lr_inner
        self.log_margin = nn.Parameter(_inv_softplus(margin_init))
        self._scoring = ResidualReadout()
        # Expose `.readout` so legacy code that pokes at model.readout still works.
        self.readout = self._scoring.readout

    def equilibrium(self, spec):
        margin = F.softplus(self.log_margin)
        return unrolled_gradient_descent(
            spec, self.n_steps, self.lr_inner, margin, self.training,
        )

    def forward(self, spec) -> Tensor:
        margin = F.softplus(self.log_margin)
        u, _ = self.equilibrium(spec)
        residuals = per_case_residuals(u, spec, margin)
        return self._scoring(residuals)

    def per_case_residuals(self, spec) -> dict[str, Tensor]:
        margin = F.softplus(self.log_margin)
        u, _ = self.equilibrium(spec)
        return per_case_residuals(u, spec, margin)

    def total_energy_at(self, u: Tensor, spec) -> Tensor:
        margin = F.softplus(self.log_margin)
        return total_energy(u, spec, margin)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
