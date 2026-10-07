"""Global joint-consistency judge over a statement set (LLM baseline).

Asks the model to decide whether a whole set of statements can be
simultaneously true, considering transitive and cyclic relations across the
entire set (not just adjacent pairs). Returns ``p_inconsistent`` in [0, 1] so
the failure-curve harness can compute AUROC directly.
"""
from __future__ import annotations

from beliefnet.llm.azure_client import AzureChatClient

_SYSTEM = (
    "You are a careful logical consistency checker. You receive a set of "
    "statements. Decide whether they can ALL be simultaneously true (jointly "
    "consistent) or are jointly contradictory. Consider transitive and cyclic "
    "relationships across the WHOLE set, not only adjacent pairs. Respond ONLY "
    'as JSON: {"inconsistent": true|false, "p_inconsistent": <0..1>, '
    '"reason": "<short>"}.'
)


def judge_consistency(client: AzureChatClient, statements: list[str]) -> dict:
    numbered = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(statements))
    user = f"Statements:\n{numbered}\n\nReturn the JSON verdict."
    out = client.complete_json(_SYSTEM, user)

    inconsistent = bool(out.get("inconsistent", False))
    p = out.get("p_inconsistent")
    if not isinstance(p, (int, float)):
        p = 1.0 if inconsistent else 0.0
    p = float(min(1.0, max(0.0, p)))
    return {"inconsistent": inconsistent, "p_inconsistent": p,
            "reason": str(out.get("reason", "")), "raw": out}


_SCIFACT_SYSTEM = (
    "You are a scientific fact-checker. Given a CLAIM and an ABSTRACT, decide "
    "whether the abstract CONTRADICTS the claim (provides evidence the claim is "
    "false). If it supports or is neutral, contradicts=false. Respond ONLY as "
    'JSON: {"contradicts": true|false, "p_contradict": <0..1>}.'
)


def judge_support_contradict(client: AzureChatClient, claim: str,
                             abstract_sentences: list[str]) -> dict:
    abstract = " ".join(abstract_sentences)
    user = f"CLAIM: {claim}\n\nABSTRACT: {abstract}\n\nReturn the JSON verdict."
    out = client.complete_json(_SCIFACT_SYSTEM, user)
    contradicts = bool(out.get("contradicts", False))
    p = out.get("p_contradict")
    if not isinstance(p, (int, float)):
        p = 1.0 if contradicts else 0.0
    return {"contradicts": contradicts,
            "p_contradict": float(min(1.0, max(0.0, p)))}
