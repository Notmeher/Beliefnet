
from beliefnet.data.synthetic.case_schema import Case
from beliefnet.graph.graph_utils import pack_graphs
from beliefnet.models.beliefnet import StructuredEquilibriumCore


def _case(cid, label, subtype, entities, constraints, kind="none"):
    return Case(id=cid, statements=[], label=label, conflict_kind=kind,
                conflict_subtype=subtype, conflict_indices=[],
                metadata={"n_statements": 0,
                          "entities": list(entities),
                          "constraints": list(constraints)})


def test_temporal_cycle_has_higher_residual_than_chain():
    chain = _case("ch", "consistent", "temporal_chain", ["A", "B", "C"],
                  [{"type": "before", "args": ["A", "B"]},
                   {"type": "before", "args": ["B", "C"]},
                   {"type": "before", "args": ["A", "C"]}])
    cycle = _case("cy", "inconsistent", "temporal_cycle", ["A", "B", "C"],
                  [{"type": "before", "args": ["A", "B"]},
                   {"type": "before", "args": ["B", "C"]},
                   {"type": "before", "args": ["C", "A"]}], kind="non_pairwise")
    model = StructuredEquilibriumCore(n_steps=100, lr_inner=0.2).eval()
    res = model.per_case_residuals(pack_graphs([chain, cycle]))
    assert float(res["total"][0]) < 0.05    # chain ~ 0
    assert float(res["total"][1]) > 0.5     # cycle has positive residual
    assert float(res["total"][1]) > float(res["total"][0])


def test_numeric_bad_sum_has_higher_residual_than_good_sum():
    good = _case("g", "consistent", "numeric_sum", ["A", "B", "T"],
                 [{"type": "anchor", "args": ["A"], "values": [40.0]},
                  {"type": "anchor", "args": ["B"], "values": [30.0]},
                  {"type": "anchor", "args": ["T"], "values": [70.0]},
                  {"type": "sum_eq", "args": ["A", "B", "T"]}])
    bad = _case("b", "inconsistent", "numeric_sum", ["A", "B", "T"],
                [{"type": "anchor", "args": ["A"], "values": [40.0]},
                 {"type": "anchor", "args": ["B"], "values": [30.0]},
                 {"type": "anchor", "args": ["T"], "values": [100.0]},
                 {"type": "sum_eq", "args": ["A", "B", "T"]}], kind="non_pairwise")
    model = StructuredEquilibriumCore(n_steps=200, lr_inner=0.05).eval()
    res = model.per_case_residuals(pack_graphs([good, bad]))
    assert float(res["total"][0]) < 5.0
    assert float(res["total"][1]) > float(res["total"][0]) + 50.0


def test_direct_negation_residual_above_zero():
    neg = _case("n", "inconsistent", "direct_negation", ["P"],
                [{"type": "anchor", "args": ["P"], "values": [1.0]},
                 {"type": "anchor", "args": ["P"], "values": [-1.0]}], kind="pairwise")
    model = StructuredEquilibriumCore(n_steps=80).eval()
    res = model.per_case_residuals(pack_graphs([neg]))
    assert float(res["total"][0]) > 0.5


def test_comparative_cycle_higher_than_chain():
    chain = _case("ch", "consistent", "comparative_chain", ["X", "Y", "Z"],
                  [{"type": "greater", "args": ["X", "Y"]},
                   {"type": "greater", "args": ["Y", "Z"]},
                   {"type": "greater", "args": ["X", "Z"]}])
    cycle = _case("cy", "inconsistent", "comparative_cycle", ["X", "Y", "Z"],
                  [{"type": "greater", "args": ["X", "Y"]},
                   {"type": "greater", "args": ["Y", "Z"]},
                   {"type": "greater", "args": ["Z", "X"]}], kind="non_pairwise")
    model = StructuredEquilibriumCore(n_steps=100, lr_inner=0.2).eval()
    res = model.per_case_residuals(pack_graphs([chain, cycle]))
    assert float(res["total"][0]) < 0.05
    assert float(res["total"][1]) > float(res["total"][0])


def test_independent_consistent_has_zero_residual():
    ind = _case("i", "consistent", "independent",
                ["diagnosis_slot", "drug_active_x", "presence_y"],
                [{"type": "anchor", "args": ["diagnosis_slot"], "values": [3.0]},
                 {"type": "anchor", "args": ["drug_active_x"], "values": [1.0]},
                 {"type": "anchor", "args": ["presence_y"], "values": [1.0]}])
    model = StructuredEquilibriumCore(n_steps=80).eval()
    res = model.per_case_residuals(pack_graphs([ind]))
    assert float(res["total"][0]) < 1e-3


def test_param_count_is_tiny():
    model = StructuredEquilibriumCore()
    # 1 margin + (4 weights + 1 bias) for readout = 6
    assert model.num_params() == 6
