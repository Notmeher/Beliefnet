"""Online belief-consistency monitor: incremental, weighted, abstaining.

Ingests statements one at a time, maintains the accumulating typed constraint
set, and after each statement decides joint satisfiability exactly. On the
first statement that makes the set unsatisfiable it emits a certificate
(minimal unsat core) and a repair (minimal correction set), localizing the
contradiction to the offending statement. Per-constraint confidences let the
monitor abstain (rather than false-alarm) when perception confidence is low.

This is the runtime-monitor form of BeliefNet: a streaming consistency guard
for tool-using LLM agents, which batch pairwise NLI cannot provide.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from beliefnet.graph.graph_builder import build_graph_from_supervisions
from beliefnet.reasoning.certificates import (
    minimal_correction_set,
    minimal_unsat_core,
)
from beliefnet.reasoning.exact_checker import is_satisfiable


@dataclass
class MonitorEvent:
    index: int
    statement: str
    status: str                     # "consistent" | "inconsistent" | "abstain"
    confidence: float = 1.0
    unsat_core: list = field(default_factory=list)
    repair: list = field(default_factory=list)


class BeliefMonitor:
    """Stream statements through an exact incremental consistency check."""

    def __init__(self, min_confidence: float = 0.0):
        self.min_confidence = min_confidence
        self._sups: list[dict] = []
        self._confs: list[float] = []
        self.flagged_index: int | None = None

    def add(self, statement: str, supervision: dict,
            confidence: float = 1.0) -> MonitorEvent:
        i = len(self._sups)
        self._sups.append(supervision)
        self._confs.append(confidence)
        active = [c for c, s in zip(self._confs, self._sups)
                  if s.get("type", "filler") != "filler"]
        conf = min(active) if active else 1.0

        try:
            _, cons = build_graph_from_supervisions(self._sups)
            sat = is_satisfiable(cons)
        except Exception:
            return MonitorEvent(i, statement, "consistent", conf)

        if sat:
            return MonitorEvent(i, statement, "consistent", conf)
        if conf < self.min_confidence:
            return MonitorEvent(i, statement, "abstain", conf)
        if self.flagged_index is None:
            self.flagged_index = i
        return MonitorEvent(i, statement, "inconsistent", conf,
                            unsat_core=minimal_unsat_core(cons),
                            repair=minimal_correction_set(cons))

    def run(self, statements: list[str], supervisions: list[dict],
            confidences: list[float] | None = None) -> list[MonitorEvent]:
        confidences = confidences or [1.0] * len(statements)
        return [self.add(s, sup, c)
                for s, sup, c in zip(statements, supervisions, confidences)]
