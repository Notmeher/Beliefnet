from beliefnet.perception.local_extract import rule_supervisions
from beliefnet.reasoning.monitor import BeliefMonitor


def _run(statements, confidences=None, min_conf=0.0):
    sups = rule_supervisions(statements)
    m = BeliefMonitor(min_confidence=min_conf)
    events = m.run(statements, sups, confidences)
    return m, events


def test_monitor_flags_and_localizes_cycle():
    stmts = ["A occurred before B.", "B occurred before C.", "C occurred before A."]
    m, _ = _run(stmts)
    assert m.flagged_index == 2  # the third statement closes the cycle


def test_monitor_consistent_chain_no_flag():
    stmts = ["A occurred before B.", "B occurred before C.", "A occurred before C."]
    m, _ = _run(stmts)
    assert m.flagged_index is None


def test_monitor_emits_certificate():
    stmts = ["A occurred before B.", "B occurred before A."]
    _, events = _run(stmts)
    assert events[-1].status == "inconsistent"
    assert len(events[-1].unsat_core) == 2
    assert len(events[-1].repair) == 1


def test_monitor_abstains_below_confidence():
    stmts = ["A occurred before B.", "B occurred before A."]
    _, events = _run(stmts, confidences=[0.4, 0.4], min_conf=0.9)
    assert events[-1].status == "abstain"
