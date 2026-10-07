"""Turn (claim, evidence) pairs into frozen-backbone features.

Encodes claim and evidence separately so we can measure their cosine similarity
(the embedding-similarity floor) and build the standard NLI pair feature
[u, v, |u-v|, u*v] for a baseline classifier. Embeddings are cached to disk so
the similarity analysis and the baseline do not re-encode the same texts.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from beliefnet.data.synthetic.single_pair import Example
from beliefnet.encoders.frozen_backbone import FrozenBackbone


@torch.no_grad()
def encode_texts(backbone: FrozenBackbone, texts: list[str], batch_size: int = 32) -> Tensor:
    chunks = []
    for i in range(0, len(texts), batch_size):
        chunks.append(backbone.encode(list(texts[i:i + batch_size])).cpu())
    return torch.cat(chunks, dim=0)


def encode_pairs(backbone: FrozenBackbone, examples: list[Example],
                 batch_size: int = 32) -> tuple[Tensor, Tensor]:
    u = encode_texts(backbone, [e.claim for e in examples], batch_size)
    v = encode_texts(backbone, [e.evidence for e in examples], batch_size)
    return u, v


def cosine(u: Tensor, v: Tensor) -> Tensor:
    return F.cosine_similarity(u, v, dim=-1)


def build_pair_features(u: Tensor, v: Tensor) -> Tensor:
    """Classic NLI feature: concat of both vectors, abs difference, product."""
    return torch.cat([u, v, (u - v).abs(), u * v], dim=-1)


def get_or_compute_embeddings(backbone: FrozenBackbone, examples: list[Example],
                              cache_path: str | Path,
                              batch_size: int = 32) -> tuple[Tensor, Tensor]:
    cache_path = Path(cache_path)
    expected_ids = [e.id for e in examples]
    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=True)
        if list(data["ids"]) == expected_ids:
            return torch.from_numpy(data["u"]), torch.from_numpy(data["v"])
    u, v = encode_pairs(backbone, examples, batch_size)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, ids=np.array(expected_ids), u=u.numpy(), v=v.numpy())
    return u, v
