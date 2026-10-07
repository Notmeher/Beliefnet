"""Type-prediction heads for the perception stack.

Three interchangeable parsers map a frozen statement embedding to typed
structured predictions (statement type + type-conditional entity/category
indices + scalar value):

* ``ClosedVocabBeliefParser``    - shared MLP encoder + Linear type head over
                                   the closed STATEMENT_TYPES vocab.
* ``PrototypeRelationParser``    - prototype-similarity type head with
                                   FROZEN Qwen reference embeddings.
* ``HybridRelationParser``       - both branches with a confidence-aware gate.

All three expose the same ``forward / decode / encode`` interface, so the
constraint assembler downstream is invariant to the choice. The legacy class
names (``PerceptionModule``, ``PrototypePerception``, ``HybridPerception``)
remain available as aliases until Stage 6.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from beliefnet.graph.constraint_types import (
    N_ARMS,
    N_DIAGNOSES,
    N_DRUGS,
    N_NAMES,
    N_ORDER_EVENTS,
    N_SYMPTOMS,
    N_TYPES,
    STATEMENT_TYPES,
)


# ---------------------------------------------------------------------------
# Closed-vocab parser (shared MLP + Linear type head)
# ---------------------------------------------------------------------------
class ClosedVocabBeliefParser(nn.Module):
    def __init__(self, hidden_size: int = 896, mlp_hidden: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden), nn.GELU(),
        )
        self.type_head = nn.Linear(mlp_hidden, N_TYPES)
        self.order_event1_head = nn.Linear(mlp_hidden, N_ORDER_EVENTS)
        self.order_event2_head = nn.Linear(mlp_hidden, N_ORDER_EVENTS)
        self.name1_head = nn.Linear(mlp_hidden, N_NAMES)
        self.name2_head = nn.Linear(mlp_hidden, N_NAMES)
        self.symptom_head = nn.Linear(mlp_hidden, N_SYMPTOMS)
        self.diagnosis_head = nn.Linear(mlp_hidden, N_DIAGNOSES)
        self.drug_head = nn.Linear(mlp_hidden, N_DRUGS)
        self.arm_head = nn.Linear(mlp_hidden, N_ARMS)
        self.value_head = nn.Linear(mlp_hidden, 1)

    def encode(self, embeddings: Tensor) -> Tensor:
        return self.encoder(embeddings)

    def heads_from_hidden(self, h: Tensor) -> dict[str, Tensor]:
        return {
            "type": self.type_head(h),
            "order_event1": self.order_event1_head(h),
            "order_event2": self.order_event2_head(h),
            "name1": self.name1_head(h),
            "name2": self.name2_head(h),
            "symptom": self.symptom_head(h),
            "diagnosis": self.diagnosis_head(h),
            "drug": self.drug_head(h),
            "arm": self.arm_head(h),
            "value": self.value_head(h).squeeze(-1),
        }

    def forward(self, embeddings: Tensor, return_hidden: bool = False):
        h = self.encoder(embeddings)
        out = self.heads_from_hidden(h)
        if return_hidden:
            out["_hidden"] = h
        return out

    def decode(self, embeddings: Tensor) -> list[dict]:
        with torch.no_grad():
            out = self.forward(embeddings)
            types = out["type"].argmax(-1).tolist()
            order1 = out["order_event1"].argmax(-1).tolist()
            order2 = out["order_event2"].argmax(-1).tolist()
            name1 = out["name1"].argmax(-1).tolist()
            name2 = out["name2"].argmax(-1).tolist()
            symptom = out["symptom"].argmax(-1).tolist()
            diagnosis = out["diagnosis"].argmax(-1).tolist()
            drug = out["drug"].argmax(-1).tolist()
            arm = out["arm"].argmax(-1).tolist()
            value = out["value"].tolist()
        return [
            {"type": types[i], "order_event1": order1[i], "order_event2": order2[i],
             "name1": name1[i], "name2": name2[i], "symptom": symptom[i],
             "diagnosis": diagnosis[i], "drug": drug[i], "arm": arm[i],
             "value": value[i]}
            for i in range(len(types))
        ]

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Prototype-similarity parser
# ---------------------------------------------------------------------------
class PrototypeRelationParser(nn.Module):
    def __init__(self, hidden_size: int, prototype_embeddings: dict[str, Tensor],
                 mlp_hidden: int = 256, dropout: float = 0.1,
                 temperature_init: float = 10.0):
        super().__init__()
        assert tuple(prototype_embeddings.keys()) == STATEMENT_TYPES, (
            "prototype_embeddings must be keyed by STATEMENT_TYPES in order")
        self.relations = list(prototype_embeddings.keys())

        max_protos = max(p.shape[0] for p in prototype_embeddings.values())
        R = len(self.relations)
        D = hidden_size
        proto_t = torch.zeros(R, max_protos, D)
        proto_m = torch.zeros(R, max_protos, dtype=torch.bool)
        for i, r in enumerate(self.relations):
            p = prototype_embeddings[r]
            proto_t[i, :p.shape[0]] = p
            proto_m[i, :p.shape[0]] = True
        self.register_buffer("prototypes", proto_t.float())
        self.register_buffer("proto_mask", proto_m)

        self.encoder = nn.Sequential(
            nn.Linear(D, mlp_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden), nn.GELU(),
        )

        self.order_event1_head = nn.Linear(mlp_hidden, N_ORDER_EVENTS)
        self.order_event2_head = nn.Linear(mlp_hidden, N_ORDER_EVENTS)
        self.name1_head = nn.Linear(mlp_hidden, N_NAMES)
        self.name2_head = nn.Linear(mlp_hidden, N_NAMES)
        self.symptom_head = nn.Linear(mlp_hidden, N_SYMPTOMS)
        self.diagnosis_head = nn.Linear(mlp_hidden, N_DIAGNOSES)
        self.drug_head = nn.Linear(mlp_hidden, N_DRUGS)
        self.arm_head = nn.Linear(mlp_hidden, N_ARMS)
        self.value_head = nn.Linear(mlp_hidden, 1)

        self.log_temperature = nn.Parameter(torch.tensor(float(temperature_init)).log())

    def encode(self, embeddings: Tensor) -> Tensor:
        return self.encoder(embeddings)

    def heads_from_hidden(self, h: Tensor) -> dict[str, Tensor]:
        return {
            "order_event1": self.order_event1_head(h),
            "order_event2": self.order_event2_head(h),
            "name1": self.name1_head(h),
            "name2": self.name2_head(h),
            "symptom": self.symptom_head(h),
            "diagnosis": self.diagnosis_head(h),
            "drug": self.drug_head(h),
            "arm": self.arm_head(h),
            "value": self.value_head(h).squeeze(-1),
        }

    def relation_logits(self, h: Tensor) -> Tensor:
        R, K, D = self.prototypes.shape
        protos_flat = self.prototypes.view(R * K, D)
        protos_proj = self.encoder(protos_flat).view(R, K, -1)

        h_n = F.normalize(h, dim=-1)
        protos_n = F.normalize(protos_proj, dim=-1)
        sims = torch.einsum("bd,rkd->brk", h_n, protos_n)
        sims = sims.masked_fill(~self.proto_mask.unsqueeze(0), -1e9)
        max_sim = sims.max(dim=-1).values
        return max_sim * self.log_temperature.exp()

    def forward(self, embeddings: Tensor, return_hidden: bool = False):
        h = self.encode(embeddings)
        out = self.heads_from_hidden(h)
        out["type"] = self.relation_logits(h)
        if return_hidden:
            out["_hidden"] = h
        return out

    def decode(self, embeddings: Tensor) -> list[dict]:
        with torch.no_grad():
            out = self.forward(embeddings)
            types = out["type"].argmax(-1).tolist()
            order1 = out["order_event1"].argmax(-1).tolist()
            order2 = out["order_event2"].argmax(-1).tolist()
            name1 = out["name1"].argmax(-1).tolist()
            name2 = out["name2"].argmax(-1).tolist()
            symptom = out["symptom"].argmax(-1).tolist()
            diagnosis = out["diagnosis"].argmax(-1).tolist()
            drug = out["drug"].argmax(-1).tolist()
            arm = out["arm"].argmax(-1).tolist()
            value = out["value"].tolist()
        return [
            {"type": types[i], "order_event1": order1[i], "order_event2": order2[i],
             "name1": name1[i], "name2": name2[i], "symptom": symptom[i],
             "diagnosis": diagnosis[i], "drug": drug[i], "arm": arm[i],
             "value": value[i]}
            for i in range(len(types))
        ]

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Hybrid (closed-vocab + prototype) with a confidence-aware gate
# ---------------------------------------------------------------------------
class HybridRelationParser(nn.Module):
    def __init__(self, hidden_size: int, prototype_embeddings: dict[str, Tensor],
                 mlp_hidden: int = 256, dropout: float = 0.1,
                 temperature_init: float = 10.0,
                 trainable_prototypes: bool = False):
        super().__init__()
        assert tuple(prototype_embeddings.keys()) == STATEMENT_TYPES
        D = hidden_size

        self.encoder_a = nn.Sequential(
            nn.Linear(D, mlp_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden), nn.GELU(),
        )
        self.type_head_a = nn.Linear(mlp_hidden, N_TYPES)

        self.encoder_b = nn.Sequential(
            nn.Linear(D, mlp_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden), nn.GELU(),
        )
        R = N_TYPES
        max_protos = max(p.shape[0] for p in prototype_embeddings.values())
        proto_t = torch.zeros(R, max_protos, D)
        proto_m = torch.zeros(R, max_protos, dtype=torch.bool)
        for i, r in enumerate(STATEMENT_TYPES):
            p = prototype_embeddings[r]
            proto_t[i, :p.shape[0]] = p
            proto_m[i, :p.shape[0]] = True
        self.register_buffer("prototypes", proto_t.float())
        self.register_buffer("proto_mask", proto_m)
        if trainable_prototypes:
            self.proto_offsets = nn.Parameter(torch.zeros_like(proto_t))
        else:
            self.register_parameter("proto_offsets", None)
        self.log_temperature = nn.Parameter(torch.tensor(float(temperature_init)).log())

        self.gate_head = nn.Sequential(
            nn.Linear(mlp_hidden + 1, mlp_hidden // 4), nn.GELU(),
            nn.Linear(mlp_hidden // 4, 1),
        )
        with torch.no_grad():
            self.gate_head[-1].bias.fill_(0.85)

        self.order_event1_head = nn.Linear(mlp_hidden, N_ORDER_EVENTS)
        self.order_event2_head = nn.Linear(mlp_hidden, N_ORDER_EVENTS)
        self.name1_head = nn.Linear(mlp_hidden, N_NAMES)
        self.name2_head = nn.Linear(mlp_hidden, N_NAMES)
        self.symptom_head = nn.Linear(mlp_hidden, N_SYMPTOMS)
        self.diagnosis_head = nn.Linear(mlp_hidden, N_DIAGNOSES)
        self.drug_head = nn.Linear(mlp_hidden, N_DRUGS)
        self.arm_head = nn.Linear(mlp_hidden, N_ARMS)
        self.value_head = nn.Linear(mlp_hidden, 1)

    def branch_a(self, embeddings: Tensor) -> tuple[Tensor, Tensor]:
        h = self.encoder_a(embeddings)
        return h, self.type_head_a(h)

    def branch_b(self, embeddings: Tensor) -> tuple[Tensor, Tensor]:
        h = self.encoder_b(embeddings)
        protos = self.prototypes
        if self.proto_offsets is not None:
            protos = protos + self.proto_offsets
        R, K, D = protos.shape
        protos_flat = protos.view(R * K, D)
        protos_proj = self.encoder_b(protos_flat).view(R, K, -1)
        h_n = F.normalize(h, dim=-1)
        p_n = F.normalize(protos_proj, dim=-1)
        sims = torch.einsum("bd,rkd->brk", h_n, p_n)
        sims = sims.masked_fill(~self.proto_mask.unsqueeze(0), -1e9)
        max_sim = sims.max(dim=-1).values
        return h, max_sim * self.log_temperature.exp()

    def gate_value(self, h_a: Tensor, a_logits: Tensor) -> Tensor:
        max_conf = F.softmax(a_logits, dim=-1).max(dim=-1, keepdim=True).values
        gate_in = torch.cat([h_a, max_conf], dim=-1)
        return torch.sigmoid(self.gate_head(gate_in).squeeze(-1))

    def heads_from_hidden(self, h_a: Tensor) -> dict[str, Tensor]:
        return {
            "order_event1": self.order_event1_head(h_a),
            "order_event2": self.order_event2_head(h_a),
            "name1": self.name1_head(h_a),
            "name2": self.name2_head(h_a),
            "symptom": self.symptom_head(h_a),
            "diagnosis": self.diagnosis_head(h_a),
            "drug": self.drug_head(h_a),
            "arm": self.arm_head(h_a),
            "value": self.value_head(h_a).squeeze(-1),
        }

    def encode(self, embeddings: Tensor) -> Tensor:
        return self.encoder_a(embeddings)

    def forward(self, embeddings: Tensor, return_hidden: bool = False,
                return_gate: bool = False):
        h_a, a_logits = self.branch_a(embeddings)
        h_b, b_logits = self.branch_b(embeddings)
        gate = self.gate_value(h_a, a_logits)
        g = gate.unsqueeze(-1)
        final_logits = g * a_logits + (1.0 - g) * b_logits
        out = self.heads_from_hidden(h_a)
        out["type"] = final_logits
        if return_hidden:
            out["_hidden"] = h_a
        if return_gate:
            out["_gate"] = gate
            out["_a_logits"] = a_logits
            out["_b_logits"] = b_logits
        return out

    def decode(self, embeddings: Tensor) -> list[dict]:
        with torch.no_grad():
            out = self.forward(embeddings)
            types = out["type"].argmax(-1).tolist()
            order1 = out["order_event1"].argmax(-1).tolist()
            order2 = out["order_event2"].argmax(-1).tolist()
            name1 = out["name1"].argmax(-1).tolist()
            name2 = out["name2"].argmax(-1).tolist()
            symptom = out["symptom"].argmax(-1).tolist()
            diagnosis = out["diagnosis"].argmax(-1).tolist()
            drug = out["drug"].argmax(-1).tolist()
            arm = out["arm"].argmax(-1).tolist()
            value = out["value"].tolist()
        return [
            {"type": types[i], "order_event1": order1[i], "order_event2": order2[i],
             "name1": name1[i], "name2": name2[i], "symptom": symptom[i],
             "diagnosis": diagnosis[i], "drug": drug[i], "arm": arm[i],
             "value": value[i]}
            for i in range(len(types))
        ]

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Multi-task / contrastive losses (shared across parsers)
# ---------------------------------------------------------------------------
def supervised_contrastive_loss(features: Tensor, labels: Tensor,
                                temperature: float = 0.1) -> Tensor:
    """SupCon: pull together features that share a label, push apart others.

    For each anchor, log-likelihood is averaged over its positives (same-class,
    excluding self). Anchors with no positives in the batch are ignored.
    """
    features = F.normalize(features, dim=-1)
    sim = features @ features.T / temperature
    n = features.shape[0]
    self_mask = torch.eye(n, dtype=torch.bool, device=features.device)
    sim = sim.masked_fill(self_mask, -1e9)

    label_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~self_mask
    log_prob = sim - torch.logsumexp(sim, dim=-1, keepdim=True)
    pos_count = label_mask.sum(dim=-1).clamp_min(1)
    per_anchor = -(log_prob * label_mask).sum(dim=-1) / pos_count
    has_pos = label_mask.sum(dim=-1) > 0
    if has_pos.sum() == 0:
        return features.new_zeros(())
    return per_anchor[has_pos].mean()


def perception_loss(logits: dict[str, Tensor], targets: dict[str, Tensor]) -> tuple[Tensor, dict]:
    """Multi-task loss with type-conditional masking.

    `type` is always supervised; the entity / value heads only contribute when
    their corresponding target index is >= 0 (which the labeler sets to -1 for
    statements where that field is irrelevant).
    """
    parts = {}
    parts["type"] = F.cross_entropy(logits["type"], targets["type"])

    def masked_ce(name):
        t = targets[name]
        mask = t >= 0
        if mask.sum() == 0:
            return logits["type"].new_zeros(())
        return F.cross_entropy(logits[name][mask], t[mask])

    for name in ("order_event1", "order_event2", "name1", "name2",
                 "symptom", "diagnosis", "drug", "arm"):
        parts[name] = masked_ce(name)

    v_mask = targets["value_mask"].bool()
    if v_mask.sum() > 0:
        parts["value"] = F.mse_loss(logits["value"][v_mask], targets["value"][v_mask])
    else:
        parts["value"] = logits["type"].new_zeros(())

    total = sum(parts.values())
    return total, {k: float(v.detach()) for k, v in parts.items()}


# Legacy aliases for backward compatibility (Stage 6 removes these).
PerceptionModule = ClosedVocabBeliefParser
PrototypePerception = PrototypeRelationParser
HybridPerception = HybridRelationParser
