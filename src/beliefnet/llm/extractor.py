"""LLM local (per-statement) extractor -> supervision dicts.

Strictly local: each statement is sent to the model on its own, so global
reasoning is left entirely to the sound checker (this is what makes the
self-fix result clean — the LLM never does the global step). Output dicts match
the supervision schema consumed by ``build_graph_from_supervisions``.

Currently emits the order relations used by the emergent benchmark
(``before`` / ``greater``); returns ``filler`` when no relation is asserted.
"""
from __future__ import annotations

from beliefnet.llm.azure_client import AzureChatClient

_SYSTEM = (
    "Extract the single typed relation asserted by ONE statement, if any. "
    "Allowed types: 'before' (entity1 happens/ranks before entity2), "
    "'greater' (entity1 is greater/older than entity2), or 'none'. "
    "Use the exact surface noun phrases from the sentence; do not infer beyond "
    'it. Respond ONLY as JSON: '
    '{"type": "before|greater|none", "entity1": "...", "entity2": "..."}.'
)


def _extract_one(client: AzureChatClient, statement: str) -> dict:
    out = client.complete_json(_SYSTEM, f"Statement: {statement}")
    t = out.get("type", "none")
    e1, e2 = out.get("entity1"), out.get("entity2")
    if t in ("before", "greater") and e1 and e2:
        return {"type": t, "entity1": str(e1).strip(), "entity2": str(e2).strip()}
    return {"type": "filler"}


def llm_supervisions(client: AzureChatClient, statements: list[str]) -> list[dict]:
    return [_extract_one(client, s) for s in statements]


# --- Schema extension: polarity/direction triples for semantic contradiction ---
_SCIFACT_POLARITY_SYSTEM = (
    "Extract the DIRECTION of the main scientific relation, independently for a "
    "claim and for an abstract, so a checker can compare them. "
    "claim_direction: 'positive' if the CLAIM asserts its relation holds / "
    "increases / is associated; 'negative' if it asserts the opposite / "
    "decreases / no association. evidence_direction: based ONLY on the ABSTRACT, "
    "the direction it establishes for that same relation -- 'positive', "
    "'negative', or 'unaddressed'. Respond ONLY as JSON: "
    '{"claim_direction":"positive|negative",'
    '"evidence_direction":"positive|negative|unaddressed"}.'
)


def scifact_polarity_constraints(client: AzureChatClient, claim: str,
                                 abstract_sentences: list[str]) -> list[dict]:
    """Independently extract claim + evidence direction as anchors on a shared
    key; opposite polarities make the exact checker flag a contradiction."""
    abstract = " ".join(abstract_sentences)
    out = client.complete_json(
        _SCIFACT_POLARITY_SYSTEM, f"CLAIM: {claim}\n\nABSTRACT: {abstract}")
    key = "claim_relation"
    cons: list[dict] = []
    cd = out.get("claim_direction")
    ed = out.get("evidence_direction")
    if cd in ("positive", "negative"):
        cons.append({"type": "anchor", "args": [key],
                     "values": [1.0 if cd == "positive" else -1.0]})
    if ed in ("positive", "negative"):
        cons.append({"type": "anchor", "args": [key],
                     "values": [1.0 if ed == "positive" else -1.0]})
    return cons
