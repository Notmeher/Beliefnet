import torch

from beliefnet.data.synthetic.case_schema import MultiStmtConfig
from beliefnet.data.synthetic.multi_statement import generate
from beliefnet.graph.graph_builder import build_graph_from_supervisions
from beliefnet.perception.parser_rules import derive_supervision
from beliefnet.perception.type_heads import ClosedVocabBeliefParser


# ---------- parser tests --------------------------------------------------
def test_parses_temporal_before():
    s = derive_supervision("the biopsy occurred before surgery.")
    assert s == {"type": "before", "entity1": "the biopsy", "entity2": "surgery"}


def test_parses_comparative_greater():
    s = derive_supervision("Alice is older than Brian.")
    assert s == {"type": "greater", "entity1": "Alice", "entity2": "Brian"}


def test_parses_arm_count_and_total():
    s = derive_supervision("The treatment arm enrolled 40 patients.")
    assert s == {"type": "arm_count", "arm": 0, "value": 40.0}
    s = derive_supervision("The control arm enrolled 30 patients.")
    assert s == {"type": "arm_count", "arm": 1, "value": 30.0}
    s = derive_supervision("The trial enrolled 70 patients in total.")
    assert s == {"type": "total_count", "value": 70.0}
    s = derive_supervision("The trial enrolled 120 patients.")
    assert s == {"type": "direct_enrolled", "value": 120.0}


def test_parses_presence_and_diagnosis_and_drug():
    s = derive_supervision("The patient has fever.")
    assert s == {"type": "presence_pos", "symptom": "fever"}
    s = derive_supervision("The patient does not have fever.")
    assert s == {"type": "presence_neg", "symptom": "fever"}
    s = derive_supervision("The patient was diagnosed with pneumonia.")
    assert s == {"type": "diagnosis", "diagnosis": "pneumonia"}
    s = derive_supervision("Drug aspirin is administered once daily.")
    assert s == {"type": "drug_active", "drug": "aspirin"}


def test_parses_fillers_as_filler():
    s = derive_supervision("The patient was referred by their primary care physician.")
    assert s == {"type": "filler"}


# ---------- assembler tests ----------------------------------------------
def test_assembler_reproduces_oracle_constraints_on_full_dataset():
    """For every case, deterministic parser -> assembler -> constraints should
    match what the generator emitted in metadata (modulo entity ORDER)."""
    cases = generate(MultiStmtConfig(n_per_subtype=80, seed=0))
    mismatches = 0
    for c in cases:
        sups = [derive_supervision(s) for s in c.statements]
        ents_pred, constr_pred = build_graph_from_supervisions(sups)
        ents_oracle = sorted(c.metadata["entities"])
        constr_oracle = c.metadata["constraints"]
        # Compare as sets of (type, args, values)
        def key(con):
            return (con["type"], tuple(con["args"]),
                    tuple(round(v, 4) for v in con.get("values", [])))
        if (set(ents_pred) != set(ents_oracle) or
                {key(c) for c in constr_pred} != {key(c) for c in constr_oracle}):
            mismatches += 1
    assert mismatches == 0, f"{mismatches}/{len(cases)} cases mismatch the oracle"


# ---------- perception module shape/sanity -------------------------------
def test_perception_module_shapes():
    m = ClosedVocabBeliefParser(hidden_size=128, mlp_hidden=64)
    e = torch.randn(7, 128)
    out = m(e)
    assert out["type"].shape == (7, 10)
    assert out["order_event1"].shape == (7, 16)
    assert out["arm"].shape == (7, 2)
    assert out["value"].shape == (7,)


def test_perception_module_decode_length():
    m = ClosedVocabBeliefParser(hidden_size=64, mlp_hidden=32)
    out = m.decode(torch.randn(5, 64))
    assert len(out) == 5
    assert set(out[0].keys()) >= {"type", "order_event1", "name1", "symptom",
                                  "diagnosis", "drug", "arm", "value"}
