"""Numeric value extractors for the perception stack.

Four variants compared on numeric statements ("The trial enrolled N patients",
"The treatment arm had N participants", etc.):

(a) PooledRegressor      - linear head on the pooled embedding.
                           Cheap, but frozen Qwen last-token pooling barely
                           separates "40" from "41".
(b) TokenAttentionExtractor - learned attention over per-token hidden states.
                           Pools by attending to numeric-bearing tokens.
(c) regex_extract_number - rule-based digit reader. Always works for the
                           templated dataset; expected MAE = 0.
(d) hybrid                - regex when it finds a digit, neural fallback (b)
                           or (a) when it doesn't.

All four are interchangeable at the value-replacement stage of the structured
equilibrium pipeline, so we can hold the perception type/categorical heads
fixed and isolate the effect of the numeric value path on downstream `sum_eq`
and `direct_numeric` residuals.
"""
from __future__ import annotations

import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

_NUM_RE = re.compile(r"\d+")


def regex_extract_number(text: str) -> float | None:
    m = _NUM_RE.search(text)
    return float(m.group(0)) if m else None


class TokenAttentionExtractor(nn.Module):
    """Attention pool over per-token Qwen hidden states -> scalar value."""

    def __init__(self, hidden_size: int, mlp_hidden: int = 128):
        super().__init__()
        self.query = nn.Parameter(torch.randn(hidden_size) / (hidden_size ** 0.5))
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(self, token_embs: Tensor, mask: Tensor) -> Tensor:
        scores = (token_embs @ self.query)
        scores = scores.masked_fill(~mask.bool(), -1e9)
        weights = F.softmax(scores, dim=-1).unsqueeze(-1)
        pooled = (token_embs * weights).sum(dim=1)
        return self.proj(pooled).squeeze(-1)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def hybrid_value(text: str, neural_value: float | None = None) -> float:
    """Prefer regex; fall back to neural value if regex finds no digit."""
    v = regex_extract_number(text)
    if v is not None:
        return v
    return float(neural_value) if neural_value is not None else 0.0
