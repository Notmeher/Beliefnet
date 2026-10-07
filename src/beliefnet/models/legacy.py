"""``StatementEnergyBaseline``: pairwise-coupling energy baseline.

A learned, gated, directed coupling proposes a target offset r_ij for each
pair of statements: an active edge wants ``z_j ~= z_i + r_ij``. The energy

    E(z) = lambda*||z - anchor||^2 + sum_ij g_ij * ||z_j - z_i - r_ij||^2

is driven to equilibrium by preconditioned gradient steps. The RESIDUAL
energy at equilibrium is the inconsistency score.

This is the v1 model (formerly ``EENConsistency``): it reasons over statement
embeddings rather than over typed entity constraints, so by construction it
cannot recover from a global / multi-statement inconsistency that has no
pairwise contradiction. It is retained as the baseline that the structured
``StructuredEquilibriumCore`` is meant to beat.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class StatementEnergyBaseline(nn.Module):
    def __init__(self, hidden_size: int, latent_dim: int = 64, mlp_hidden: int = 128,
                 n_steps: int = 25, step_size: float = 0.5):
        super().__init__()
        d = latent_dim
        self.n_steps = n_steps
        self.step_size = step_size

        self.proj = nn.Linear(hidden_size, d)
        self.gate_mlp = nn.Sequential(
            nn.Linear(4 * d, mlp_hidden), nn.ReLU(), nn.Linear(mlp_hidden, 1))
        self.rel_mlp = nn.Sequential(
            nn.Linear(2 * d, mlp_hidden), nn.ReLU(), nn.Linear(mlp_hidden, d))
        self.log_lambda = nn.Parameter(torch.zeros(()))
        self.readout = nn.Linear(4, 1)

    def _couplings(self, p: Tensor, mask: Tensor):
        b, k, d = p.shape
        pi = p.unsqueeze(2).expand(b, k, k, d)
        pj = p.unsqueeze(1).expand(b, k, k, d)
        gate_in = torch.cat([pi, pj, (pi - pj).abs(), pi * pj], dim=-1)
        g = torch.sigmoid(self.gate_mlp(gate_in)).squeeze(-1)
        eye = torch.eye(k, dtype=torch.bool, device=p.device).unsqueeze(0)
        pair_mask = (mask.unsqueeze(2) & mask.unsqueeze(1) & ~eye).float()
        g = g * pair_mask
        r = self.rel_mlp(torch.cat([pi, pj], dim=-1))
        return g, r, pair_mask

    def _equilibrium(self, emb: Tensor, mask: Tensor):
        node_mask = mask.float()
        p = self.proj(emb)
        g, r, pair_mask = self._couplings(p, mask)
        lam = F.softplus(self.log_lambda)

        deg = lam + g.sum(dim=1) + g.sum(dim=2)
        precond = (self.step_size / (deg + 1e-6)).unsqueeze(-1)

        z = p.clone()
        traj = []
        for _ in range(self.n_steps):
            c = z.unsqueeze(1) - z.unsqueeze(2) - r
            gc = g.unsqueeze(-1) * c
            grad_rel = 2.0 * (gc.sum(dim=1) - gc.sum(dim=2))
            grad_data = 2.0 * lam * (z - p)
            grad = (grad_data + grad_rel) * node_mask.unsqueeze(-1)
            z = z - precond * grad
            traj.append(((g * (c * c).sum(-1)).sum((1, 2))
                         / pair_mask.sum((1, 2)).clamp_min(1)).mean().item())

        c = z.unsqueeze(1) - z.unsqueeze(2) - r
        edge_energy = g * (c * c).sum(-1)
        n_edges = pair_mask.sum((1, 2)).clamp_min(1.0)
        mean_res = edge_energy.sum((1, 2)) / n_edges
        max_res = edge_energy.flatten(1).max(dim=1).values
        data_res = ((z - p).pow(2).sum(-1) * node_mask).sum(1) / node_mask.sum(1).clamp_min(1.0)
        gate_density = g.sum((1, 2)) / n_edges
        feats = torch.stack([mean_res, max_res, data_res, gate_density], dim=-1)
        return feats, traj

    def forward(self, emb: Tensor, mask: Tensor) -> Tensor:
        feats, _ = self._equilibrium(emb, mask)
        return self.readout(feats).squeeze(-1)

    def energy_trajectory(self, emb: Tensor, mask: Tensor) -> list[float]:
        with torch.no_grad():
            _, traj = self._equilibrium(emb, mask)
        return traj

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
