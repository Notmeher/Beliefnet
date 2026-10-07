"""Tests for the unrolled-GD equilibrium solver and the typed energies."""
from __future__ import annotations

import torch

from beliefnet.equilibrium.energy_functions import per_case_residuals, total_energy
from beliefnet.equilibrium.solver import unrolled_gradient_descent
from beliefnet.graph.graph_utils import pack_graphs

from beliefnet.data.synthetic.case_schema import Case


def _case(cid, entities, constraints, kind="none", label="consistent", subtype="x"):
    return Case(id=cid, statements=[], label=label, conflict_kind=kind,
                conflict_subtype=subtype, conflict_indices=[],
                metadata={"n_statements": 0,
                          "entities": list(entities),
                          "constraints": list(constraints)})


def _margin(value: float = 1.0) -> torch.Tensor:
    return torch.tensor(value, dtype=torch.float)


def test_energy_zero_on_satisfied_anchor():
    spec = pack_graphs([_case("c", ["e"],
                              [{"type": "anchor", "args": ["e"], "values": [5.0]}])])
    u = torch.tensor([5.0])
    assert float(total_energy(u, spec, _margin())) == 0.0


def test_energy_positive_on_anchor_disagreement():
    spec = pack_graphs([_case("c", ["e"],
                              [{"type": "anchor", "args": ["e"], "values": [1.0]},
                               {"type": "anchor", "args": ["e"], "values": [-1.0]}])])
    u = torch.tensor([0.0])
    assert float(total_energy(u, spec, _margin())) > 1.5


def test_solver_descends_monotonically():
    spec = pack_graphs([_case("c", ["a", "b", "c"],
                              [{"type": "before", "args": ["a", "b"]},
                               {"type": "before", "args": ["b", "c"]},
                               {"type": "before", "args": ["c", "a"]}],
                              kind="non_pairwise", label="inconsistent",
                              subtype="temporal_cycle")])
    _, traj = unrolled_gradient_descent(spec, n_steps=30, lr_inner=0.1,
                                        margin=_margin(), training=False)
    # GD on a convex-relaxed energy from zero init should be monotone non-increasing.
    for prev, nxt in zip(traj[:-1], traj[1:]):
        assert nxt <= prev + 1e-6


def test_cycle_has_higher_residual_than_chain():
    chain = _case("ch", ["A", "B", "C"],
                  [{"type": "before", "args": ["A", "B"]},
                   {"type": "before", "args": ["B", "C"]},
                   {"type": "before", "args": ["A", "C"]}])
    cycle = _case("cy", ["A", "B", "C"],
                  [{"type": "before", "args": ["A", "B"]},
                   {"type": "before", "args": ["B", "C"]},
                   {"type": "before", "args": ["C", "A"]}],
                  kind="non_pairwise", label="inconsistent")
    spec = pack_graphs([chain, cycle])
    u, _ = unrolled_gradient_descent(spec, n_steps=80, lr_inner=0.2,
                                     margin=_margin(), training=False)
    r = per_case_residuals(u, spec, _margin())
    assert float(r["total"][0]) < 0.05
    assert float(r["total"][1]) > 0.5


def test_per_case_residuals_partition_by_bucket():
    """Anchor disagreement contributes only to the anchor bucket, not hinge."""
    spec = pack_graphs([_case("c", ["e"],
                              [{"type": "anchor", "args": ["e"], "values": [5.0]},
                               {"type": "anchor", "args": ["e"], "values": [-5.0]}])])
    u = torch.tensor([0.0])
    r = per_case_residuals(u, spec, _margin())
    assert float(r["anchor"]) > 1.0
    assert float(r["hinge"]) == 0.0
    assert float(r["sum"]) == 0.0
    assert float(r["total"]) == float(r["anchor"])


def test_per_case_residuals_scatter_across_cases():
    a = _case("a", ["e"], [{"type": "anchor", "args": ["e"], "values": [3.0]}])
    b = _case("b", ["f"], [{"type": "anchor", "args": ["f"], "values": [3.0]}])
    spec = pack_graphs([a, b])
    u = torch.tensor([3.0, 0.0])  # case a satisfied, case b not
    r = per_case_residuals(u, spec, _margin())
    assert float(r["total"][0]) == 0.0
    assert float(r["total"][1]) > 1.0
