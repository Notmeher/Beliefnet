"""Unrolled gradient-descent solver for the typed-constraint energy.

Initialises per-entity latents to zero and runs ``n_steps`` of GD on the
joint energy ``E(u)``. ``create_graph=training`` lets the outer loss
back-propagate through the inner trajectory at training time while staying
no-grad-safe at evaluation.

The solver is the *equilibrium* mechanism: it does NOT depend on a learned
parameter beyond the ``margin`` passed in by the caller. Anything trainable
lives in the model that owns the margin and the readout.
"""
from __future__ import annotations

import torch
from torch import Tensor

from beliefnet.equilibrium.energy_functions import total_energy


def unrolled_gradient_descent(spec, n_steps: int, lr_inner: float,
                              margin: Tensor, training: bool
                              ) -> tuple[Tensor, list[float]]:
    """Return ``(u_star, energy_trajectory)`` after ``n_steps`` of inner GD."""
    device = spec.anchor_idx.device if spec.n_anchors else spec.before_a.device
    with torch.enable_grad():
        u = torch.zeros(spec.n_entities, device=device, requires_grad=True)
        traj: list[float] = []
        for _ in range(n_steps):
            E = total_energy(u, spec, margin)
            (grad,) = torch.autograd.grad(E, u, create_graph=training)
            u = u - lr_inner * grad
            traj.append(float(E.detach()))
    return u, traj
