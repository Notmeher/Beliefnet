from beliefnet.perception.belief_parser import (
    BeliefParser,
    ParaphraseRobustBeliefParser,
    RuleBeliefParser,
)
from beliefnet.reasoning.exact_checker import is_satisfiable


def test_rule_parser_is_beliefparser():
    assert issubclass(RuleBeliefParser, BeliefParser)


def test_rule_parser_detects_cycle():
    _, cons = RuleBeliefParser().parse(
        ["A occurred before B.", "B occurred before C.", "C occurred before A."])
    assert not is_satisfiable(cons)


def test_paraphrase_robust_uses_supervision_fn():
    def fake_fn(statements):
        return [{"type": "before", "entity1": "A", "entity2": "B"},
                {"type": "before", "entity1": "B", "entity2": "A"}]
    _, cons = ParaphraseRobustBeliefParser(fake_fn).parse(["x", "y"])
    assert not is_satisfiable(cons)
