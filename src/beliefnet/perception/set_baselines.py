"""Strong baselines for multi-statement consistency.

Both consume frozen statement embeddings for a case, padded to a fixed number
of statements with a boolean mask, and emit a single inconsistency logit.

* MeanPoolClassifier - pools the set (mean + masked max) then classifies. A
  pure set-summary model.
* AllPairsClassifier  - scores every ordered pair with a shared MLP and
  aggregates (max or logsumexp). This one HAS pairwise reasoning, so it should
  catch pairwise conflicts. By construction it cannot see conflicts that only
  emerge from 3+ statements (cycles, global sums); that gap is the test.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

_NEG = -1e9


class MeanPoolClassifier(nn.Module):
    def __init__(self, hidden_size: int, mlp_hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * hidden_size, mlp_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(self, emb: Tensor, mask: Tensor) -> Tensor:
        m = mask.unsqueeze(-1).float()
        mean = (emb * m).sum(1) / m.sum(1).clamp_min(1.0)
        masked = torch.where(mask.unsqueeze(-1), emb, torch.full_like(emb, _NEG))
        mx = masked.max(dim=1).values
        return self.net(torch.cat([mean, mx], dim=-1)).squeeze(-1)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class AllPairsClassifier(nn.Module):
    def __init__(self, hidden_size: int, mlp_hidden: int = 256, dropout: float = 0.1,
                 agg: str = "max"):
        super().__init__()
        self.agg = agg
        self.pair = nn.Sequential(
            nn.Linear(4 * hidden_size, mlp_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(self, emb: Tensor, mask: Tensor) -> Tensor:
        b, k, h = emb.shape
        ei = emb.unsqueeze(2).expand(b, k, k, h)
        ej = emb.unsqueeze(1).expand(b, k, k, h)
        feat = torch.cat([ei, ej, (ei - ej).abs(), ei * ej], dim=-1)
        logit = self.pair(feat).squeeze(-1)

        pair_mask = mask.unsqueeze(2) & mask.unsqueeze(1)
        eye = torch.eye(k, dtype=torch.bool, device=emb.device).unsqueeze(0)
        pair_mask = pair_mask & ~eye
        masked = torch.where(pair_mask, logit, torch.full_like(logit, _NEG))
        flat = masked.flatten(1)
        if self.agg == "logsumexp":
            return torch.logsumexp(flat, dim=1)
        return flat.max(dim=1).values

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
