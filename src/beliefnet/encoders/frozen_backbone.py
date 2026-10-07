"""Frozen LLM backbone for BeliefNet.

The backbone is a pretrained causal LM held frozen. BeliefNet's perception and
equilibrium modules consume its hidden-state representations; the backbone
itself is never trained here. This keeps the contribution of the perception
and equilibrium layers isolable from any change in the underlying LM.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import yaml
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

Pooling = Literal["last", "mean"]

_DTYPES: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


@dataclass
class FrozenBackboneConfig:
    """Configuration for the frozen backbone.

    Defaults target Turing GPUs (TITAN RTX, sm_75): fp16 compute (no bf16) and
    SDPA attention (no FlashAttention-2).
    """

    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    load_in_4bit: bool = True
    compute_dtype: str = "float16"
    attn_implementation: str = "sdpa"
    layer: int = -1            # index into output.hidden_states (-1 = last layer)
    pooling: Pooling = "last"  # how to collapse the token axis to one vector
    max_length: int = 512
    device_map: str = "auto"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "FrozenBackboneConfig":
        with open(path) as fh:
            data = yaml.safe_load(fh) or {}
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown backbone config keys: {sorted(unknown)}")
        return cls(**data)

    @property
    def torch_dtype(self) -> torch.dtype:
        if self.compute_dtype not in _DTYPES:
            raise ValueError(f"Unsupported compute_dtype: {self.compute_dtype}")
        return _DTYPES[self.compute_dtype]


def pool_hidden(hidden: Tensor, attention_mask: Tensor, mode: Pooling) -> Tensor:
    """Collapse [B, T, H] token states to [B, H] using the attention mask.

    Pure function (no model state) so it can be unit-tested directly. Assumes
    right-padding: the last real token of row i is at index sum(mask_i) - 1.
    """
    if mode == "mean":
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)  # [B, T, 1]
        summed = (hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp_min(1.0)
        return summed / counts
    if mode == "last":
        lengths = attention_mask.sum(dim=1)                   # [B]
        last_idx = (lengths - 1).clamp_min(0).long()          # [B]
        batch_idx = torch.arange(hidden.size(0), device=hidden.device)
        return hidden[batch_idx, last_idx]
    raise ValueError(f"Unknown pooling mode: {mode}")


class FrozenBackbone:
    """Loads a pretrained causal LM (frozen) and extracts pooled hidden states."""

    def __init__(self, config: FrozenBackboneConfig):
        self.config = config

        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Right-padding keeps last-token pooling well-defined.
        self.tokenizer.padding_side = "right"

        quant_config = None
        if config.load_in_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=config.torch_dtype,
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            quantization_config=quant_config,
            dtype=config.torch_dtype,
            attn_implementation=config.attn_implementation,
            device_map=config.device_map,
        )
        self.model.eval()
        self.model.requires_grad_(False)

    @property
    def hidden_size(self) -> int:
        return int(self.model.config.hidden_size)

    @property
    def device(self) -> torch.device:
        return self.model.get_input_embeddings().weight.device

    @torch.no_grad()
    def encode(self, texts: list[str]) -> Tensor:
        """Return float32 pooled representations of shape [len(texts), H]."""
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
        ).to(self.device)

        outputs = self.model(**batch, output_hidden_states=True)
        hidden = outputs.hidden_states[self.config.layer].float()  # [B, T, H]
        return pool_hidden(hidden, batch["attention_mask"], self.config.pooling)

    @torch.no_grad()
    def encode_tokens(self, texts: list[str]) -> tuple[Tensor, Tensor]:
        """Return per-token hidden states and attention mask (no pooling).

        Used by token-level extractors that need access to specific positions
        (e.g. digit tokens) rather than the pooled last-token vector.
        """
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
        ).to(self.device)
        outputs = self.model(**batch, output_hidden_states=True)
        hidden = outputs.hidden_states[self.config.layer].float()
        return hidden, batch["attention_mask"]
