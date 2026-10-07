"""``BeliefNet``: the end-to-end pipeline as one object.

Composes a ``BeliefParser`` (perception) with a reasoner:

* ``reasoner="exact"``       - sound checker + certificate (pure Python);
* ``reasoner="equilibrium"`` - differentiable ``StructuredEquilibriumCore``
                               residual (lazy torch).

``predict`` returns the decision and, on a contradiction, the auditable
certificate (minimal unsat core + minimal repair).
"""
from __future__ import annotations

from beliefnet.perception.belief_parser import BeliefParser


class BeliefNet:
    def __init__(self, parser: BeliefParser, reasoner: str = "exact"):
        self.parser = parser
        self.reasoner = reasoner

    def predict(self, statements: list[str]) -> dict:
        entities, constraints = self.parser.parse(statements)
        if self.reasoner == "exact":
            from beliefnet.reasoning.certificates import (
                minimal_correction_set,
                minimal_unsat_core,
            )
            from beliefnet.reasoning.exact_checker import is_satisfiable
            sat = is_satisfiable(constraints)
            out = {"inconsistent": not sat, "entities": entities,
                   "constraints": constraints}
            if not sat:
                out["unsat_core"] = minimal_unsat_core(constraints)
                out["repair"] = minimal_correction_set(constraints)
            return out
        if self.reasoner == "equilibrium":
            import torch
            from beliefnet.data.synthetic.case_schema import Case
            from beliefnet.graph.graph_utils import pack_graphs
            from beliefnet.models.beliefnet import StructuredEquilibriumCore
            case = Case(id="rt", statements=statements, label="consistent",
                        conflict_kind="none", conflict_subtype="runtime",
                        metadata={"entities": entities, "constraints": constraints})
            core = StructuredEquilibriumCore().eval()
            with torch.no_grad():
                res = core.per_case_residuals(pack_graphs([case]))["total"]
            return {"residual": float(res[0]), "entities": entities,
                    "constraints": constraints}
        raise ValueError(f"unknown reasoner: {self.reasoner!r}")
