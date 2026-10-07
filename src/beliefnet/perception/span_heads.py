"""Token-level entity span tagger (BIO heads on per-token Qwen states).

Reads per-token Qwen hidden states (from ``FrozenBackbone.encode_tokens``) and
emits BIO tags per token. At inference time the predictions are decoded into
entity spans which are then matched against a closed vocabulary to recover
entity IDs, so the entity-grounding step is span-based rather than relying on
the pooled-embedding entity heads.

Combined with direction-aware ordering (forward vs reversed-direction
relations like "X happened after Y"), this delivers paraphrase-robust entity
extraction that the closed-vocab heads could not.

The module class is named ``SpanGroundedParser`` for the BeliefNet API;
``SpanTagger`` is kept as an alias for backward compatibility with the EEN-v8
scripts during the migration.
"""
from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from beliefnet.graph.constraint_types import N_BIO_TAGS
from beliefnet.perception.parser_rules import decode_bio_spans


class SpanGroundedParser(nn.Module):
    def __init__(self, hidden_size: int, mlp_hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden), nn.GELU(),
        )
        self.tag_head = nn.Linear(mlp_hidden, N_BIO_TAGS)

    def forward(self, token_embs: Tensor) -> Tensor:
        """token_embs: [B, T, H] -> [B, T, N_BIO_TAGS] logits."""
        return self.tag_head(self.encoder(token_embs))

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


SpanTagger = SpanGroundedParser


def _match_vocab(span_text: str, candidates: list[str]) -> str | None:
    s = span_text.lower().strip()
    if not s:
        return None
    for v in candidates:
        if v.lower() == s:
            return v
    for v in candidates:
        if v.lower() in s or s in v.lower():
            return v
    return None


# Direction keywords. If any reverse keyword appears in the statement, swap
# the order of the two matched entities so the constraint args are always
# canonical (early -> late, greater -> lesser).
_TEMPORAL_REVERSE_KW = (
    "after", "later than", "followed", "came after", "the timeline",
)
_COMPARATIVE_REVERSE_KW = (
    "younger than", "less old", "junior of", "below", "less than", "smaller than",
)


def is_reversed_direction(text: str, relation_type: str) -> bool:
    lower = text.lower()
    kws = _TEMPORAL_REVERSE_KW if relation_type == "before" else _COMPARATIVE_REVERSE_KW
    return any(kw in lower for kw in kws)


def extract_relation_args(text: str, tag_indices: list[int],
                          offsets: list[tuple[int, int]],
                          relation_type: str,
                          event_vocab: list[str],
                          name_vocab: list[str]) -> tuple[str | None, str | None]:
    """For a relation-typed statement, decode spans and return (entity1, entity2)
    in canonical order (entity1 < entity2 by the relation). Returns (None, None)
    if span extraction can't find two matching entities."""
    spans = decode_bio_spans(text, tag_indices, offsets)
    if relation_type == "before":
        target_type, vocab = "EVENT", event_vocab
    elif relation_type == "greater":
        target_type, vocab = "NAME", name_vocab
    else:
        return None, None

    matched: list[str] = []
    for s in spans:
        if s["type"] != target_type:
            continue
        v = _match_vocab(s["text"], vocab)
        if v is not None and v not in matched:
            matched.append(v)
        if len(matched) >= 2:
            break
    if len(matched) < 2:
        return None, None

    if is_reversed_direction(text, relation_type):
        return matched[1], matched[0]
    return matched[0], matched[1]


def extract_categorical(text: str, tag_indices: list[int],
                        offsets: list[tuple[int, int]],
                        target_type: str, vocab: list[str]) -> str | None:
    """For single-entity statements (symptom/diagnosis/drug), find one matching
    span and return its vocab ID."""
    spans = decode_bio_spans(text, tag_indices, offsets)
    for s in spans:
        if s["type"] != target_type:
            continue
        v = _match_vocab(s["text"], vocab)
        if v is not None:
            return v
    return None
