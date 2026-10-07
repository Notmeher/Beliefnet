from beliefnet.models.pipeline import BeliefNet
from beliefnet.perception.belief_parser import RuleBeliefParser


def test_beliefnet_detects_and_certifies():
    net = BeliefNet(RuleBeliefParser())
    out = net.predict(["A occurred before B.", "B occurred before C.",
                       "C occurred before A."])
    assert out["inconsistent"] is True
    assert len(out["unsat_core"]) == 3
    assert len(out["repair"]) == 1


def test_beliefnet_consistent_has_no_certificate():
    net = BeliefNet(RuleBeliefParser())
    out = net.predict(["A occurred before B.", "B occurred before C."])
    assert out["inconsistent"] is False
    assert "unsat_core" not in out
