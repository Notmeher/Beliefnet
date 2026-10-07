"""BeliefParser: the perception interface (text -> typed constraint graph).

Closes the doc/code gap: the public ``BeliefParser`` ABC plus concrete parsers.

* ``RuleBeliefParser``               - deterministic, offline (rule parser);
* ``LLMBeliefParser``                - Azure LLM per-statement extraction;
* ``ParaphraseRobustBeliefParser``   - driven by a trained supervision function
                                       (wire from the trained Hybrid + Span +
                                       numeric pipeline in train_synthetic.py).

All return ``(entities, constraints)`` via ``build_graph_from_supervisions``,
so any downstream reasoner is invariant to the parser choice.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

from beliefnet.graph.graph_builder import build_graph_from_supervisions


class BeliefParser(ABC):
    @abstractmethod
    def parse(self, statements: list[str]) -> tuple[list[str], list[dict]]:
        """Map statements to (entities, typed constraints)."""

    def supervisions(self, statements: list[str]) -> list[dict]:
        raise NotImplementedError


class RuleBeliefParser(BeliefParser):
    def supervisions(self, statements: list[str]) -> list[dict]:
        from beliefnet.perception.local_extract import rule_supervisions
        return rule_supervisions(statements)

    def parse(self, statements: list[str]) -> tuple[list[str], list[dict]]:
        return build_graph_from_supervisions(self.supervisions(statements))


class LLMBeliefParser(BeliefParser):
    """Per-statement (strictly local) extraction via an Azure chat client."""

    def __init__(self, client):
        self.client = client

    def supervisions(self, statements: list[str]) -> list[dict]:
        from beliefnet.llm.extractor import llm_supervisions
        return llm_supervisions(self.client, statements)

    def parse(self, statements: list[str]) -> tuple[list[str], list[dict]]:
        return build_graph_from_supervisions(self.supervisions(statements))


class ParaphraseRobustBeliefParser(BeliefParser):
    """Composite parser driven by a trained supervision function.

    Wire ``supervision_fn`` from the trained Hybrid type head + SpanGrounded
    tagger + regex numeric pipeline (the inference glue in
    ``scripts/train_synthetic.py``). Kept backend-agnostic so this module has
    no hard torch dependency.
    """

    def __init__(self, supervision_fn: Callable[[list[str]], list[dict]]):
        self._fn = supervision_fn

    def supervisions(self, statements: list[str]) -> list[dict]:
        return list(self._fn(statements))

    def parse(self, statements: list[str]) -> tuple[list[str], list[dict]]:
        return build_graph_from_supervisions(self.supervisions(statements))
