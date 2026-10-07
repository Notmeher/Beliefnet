"""``GraphBatch``: vectorized batch container for typed-constraint cases.

Packs a list of ``Case`` objects into flat tensors so the entire batch's
equilibrium can be solved in one vectorized GD loop. Each entity gets a
unique global index; per-constraint tensors reference entities by that
index, and the ``*_case`` tensors let us scatter residuals back per case.

The legacy name ``BatchSpec`` is aliased here so checkpoints / scripts that
imported the old class keep working until Stage 6.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from beliefnet.data.synthetic.case_schema import Case


@dataclass
class GraphBatch:
    n_entities: int
    n_cases: int
    anchor_idx: torch.Tensor
    anchor_values: torch.Tensor
    anchor_case: torch.Tensor
    before_a: torch.Tensor
    before_b: torch.Tensor
    before_case: torch.Tensor
    greater_a: torch.Tensor
    greater_b: torch.Tensor
    greater_case: torch.Tensor
    sum_a: torch.Tensor
    sum_b: torch.Tensor
    sum_t: torch.Tensor
    sum_case: torch.Tensor

    @property
    def n_anchors(self) -> int:
        return int(self.anchor_idx.numel())

    @property
    def n_before(self) -> int:
        return int(self.before_a.numel())

    @property
    def n_greater(self) -> int:
        return int(self.greater_a.numel())

    @property
    def n_sum(self) -> int:
        return int(self.sum_a.numel())

    def to(self, device) -> "GraphBatch":
        return GraphBatch(
            n_entities=self.n_entities,
            n_cases=self.n_cases,
            anchor_idx=self.anchor_idx.to(device),
            anchor_values=self.anchor_values.to(device),
            anchor_case=self.anchor_case.to(device),
            before_a=self.before_a.to(device),
            before_b=self.before_b.to(device),
            before_case=self.before_case.to(device),
            greater_a=self.greater_a.to(device),
            greater_b=self.greater_b.to(device),
            greater_case=self.greater_case.to(device),
            sum_a=self.sum_a.to(device),
            sum_b=self.sum_b.to(device),
            sum_t=self.sum_t.to(device),
            sum_case=self.sum_case.to(device),
        )


BatchSpec = GraphBatch


def _long(x: list[int]) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.long) if x else torch.zeros(0, dtype=torch.long)


def _float(x: list[float]) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.float) if x else torch.zeros(0, dtype=torch.float)


def pack_graphs(cases: list[Case]) -> GraphBatch:
    """Pack per-case structured constraints into flat tensors.

    Each entity gets a unique global index; constraints reference entities by
    that index. The case_idx tensors let us scatter residuals back per case.
    """
    n_cases = len(cases)
    n_entities = 0
    anchor_idx, anchor_vals, anchor_case = [], [], []
    before_a, before_b, before_case = [], [], []
    greater_a, greater_b, greater_case = [], [], []
    sum_a, sum_b, sum_t, sum_case = [], [], [], []

    for ci, case in enumerate(cases):
        ents = case.metadata.get("entities", []) or []
        local_to_global = {e: n_entities + i for i, e in enumerate(ents)}
        n_entities += len(ents)
        for c in case.metadata.get("constraints", []) or []:
            ctype = c["type"]
            if ctype == "anchor":
                anchor_idx.append(local_to_global[c["args"][0]])
                anchor_vals.append(float(c["values"][0]))
                anchor_case.append(ci)
            elif ctype == "before":
                a, b = c["args"]
                before_a.append(local_to_global[a])
                before_b.append(local_to_global[b])
                before_case.append(ci)
            elif ctype == "greater":
                a, b = c["args"]
                greater_a.append(local_to_global[a])
                greater_b.append(local_to_global[b])
                greater_case.append(ci)
            elif ctype == "sum_eq":
                a, b, t = c["args"]
                sum_a.append(local_to_global[a])
                sum_b.append(local_to_global[b])
                sum_t.append(local_to_global[t])
                sum_case.append(ci)

    return GraphBatch(
        n_entities=n_entities,
        n_cases=n_cases,
        anchor_idx=_long(anchor_idx),
        anchor_values=_float(anchor_vals),
        anchor_case=_long(anchor_case),
        before_a=_long(before_a),
        before_b=_long(before_b),
        before_case=_long(before_case),
        greater_a=_long(greater_a),
        greater_b=_long(greater_b),
        greater_case=_long(greater_case),
        sum_a=_long(sum_a),
        sum_b=_long(sum_b),
        sum_t=_long(sum_t),
        sum_case=_long(sum_case),
    )


# Legacy alias.
build_batch_spec = pack_graphs
