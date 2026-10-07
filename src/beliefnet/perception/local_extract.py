"""Local (per-statement) rule extraction to supervision dicts.

Parses each statement independently with the deterministic rule parser, so no
global reasoning happens during extraction (the sound checker does all global
reasoning). The LLM counterpart is ``beliefnet.llm.extractor.llm_supervisions``;
both return the same supervision-dict format consumed by
``beliefnet.graph.graph_builder.build_graph_from_supervisions``.
"""
from __future__ import annotations

from beliefnet.perception.parser_rules import derive_supervision


def rule_supervisions(statements: list[str]) -> list[dict]:
    return [derive_supervision(s) for s in statements]
