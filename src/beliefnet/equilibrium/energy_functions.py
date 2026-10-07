"""Typed-constraint energy terms over per-entity scalar latents ``u``.

Each constraint kind contributes a fixed, structurally-forced energy:

* ``anchor``  : squared deviation ``(u_i - v)^2``;
* ``before``  : hinge ``relu(margin - (u_b - u_a))^2`` enforcing ``u_a < u_b``;
* ``greater`` : hinge ``relu(margin - (u_a - u_b))^2`` enforcing ``u_a > u_b``;
* ``sum_eq``  : squared residual ``(u_a + u_b - u_t)^2``.

The functions here are *stateless* (no learned parameters) so the equilibrium
solver and the residual readout can call them without owning the structure.
Callers pass a ``BatchSpec`` describing entity indices and per-case index
buckets; see ``beliefnet.graph.graph_utils`` for the schema.
"""
from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor


def total_energy(u: Tensor, spec, margin: Tensor) -> Tensor:
    """Sum of all constraint energies for the batch (scalar)."""
    E = u.new_zeros(())
    if spec.n_anchors > 0:
        E = E + ((u[spec.anchor_idx] - spec.anchor_values) ** 2).sum()
    if spec.n_before > 0:
        d = u[spec.before_b] - u[spec.before_a]
        E = E + (F.relu(margin - d) ** 2).sum()
    if spec.n_greater > 0:
        d = u[spec.greater_a] - u[spec.greater_b]
        E = E + (F.relu(margin - d) ** 2).sum()
    if spec.n_sum > 0:
        r = u[spec.sum_a] + u[spec.sum_b] - u[spec.sum_t]
        E = E + (r ** 2).sum()
    return E


def per_case_residuals(u: Tensor, spec, margin: Tensor) -> dict[str, Tensor]:
    """Scatter the per-constraint energies back to per-case totals.

    Returns a dict with the four buckets (``anchor``, ``hinge``, ``sum``,
    ``total``); each entry is shape ``[n_cases]``.
    """
    def Z():
        return u.new_zeros((spec.n_cases,))

    E_anc, E_hin, E_sum = Z(), Z(), Z()
    if spec.n_anchors > 0:
        v = (u[spec.anchor_idx] - spec.anchor_values) ** 2
        E_anc = E_anc.index_add(0, spec.anchor_case, v)
    if spec.n_before > 0:
        d = u[spec.before_b] - u[spec.before_a]
        E_hin = E_hin.index_add(0, spec.before_case, F.relu(margin - d) ** 2)
    if spec.n_greater > 0:
        d = u[spec.greater_a] - u[spec.greater_b]
        E_hin = E_hin.index_add(0, spec.greater_case, F.relu(margin - d) ** 2)
    if spec.n_sum > 0:
        r = u[spec.sum_a] + u[spec.sum_b] - u[spec.sum_t]
        E_sum = E_sum.index_add(0, spec.sum_case, r ** 2)
    return {"anchor": E_anc, "hinge": E_hin, "sum": E_sum,
            "total": E_anc + E_hin + E_sum}
