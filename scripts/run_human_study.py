"""Proxy-reviewer study: does a sound certificate help verification more than an
LLM's free-text rationale?

A reviewer (Claude Haiku) judges the joint consistency of hard numeric cases
under three conditions: (none) statements only; (cert) statements + the sound
checker's arithmetic certificate; (rationale) statements + a GPT-4o free-text
rationale. If the certificate yields higher reviewer accuracy than the rationale
(which propagates the assistant's arithmetic errors), auditability is a measured
benefit, not just a claim.

    PYTHONPATH=src python scripts/run_human_study.py --reviewer-model claude-haiku-4-5
"""
from __future__ import annotations

import argparse

import numpy as np

from beliefnet.data.numeric_emergent import generate_numeric
from beliefnet.eval.human_study import cohen_kappa
from beliefnet.eval.reporting import build_details_md, save_result
from beliefnet.llm.azure_client import client_for, load_dotenv, model_available

_SYS = ("You verify whether a set of statements is jointly CONSISTENT (all can "
        "be true at once) or jointly INCONSISTENT. You may be given an AID. "
        'Decide. Respond ONLY as JSON: {"inconsistent": true|false}.')


def _numeric_certificate(case) -> str:
    cons = case.metadata["constraints"]
    anchors = {c["args"][0]: c["values"][0] for c in cons if c["type"] == "anchor"}
    depth = case.metadata["depth"]
    v0 = anchors["acc0"]
    deltas = [anchors[f"d{i}"] for i in range(1, depth + 1)]
    final = anchors[f"acc{depth}"]
    true = v0 + sum(deltas)
    if abs(true - final) < 1e-9:
        return (f"Sound checker: {v0:.0f} + sum(transactions) = {true:.0f} = "
                f"stated final. SATISFIABLE (consistent).")
    return (f"Sound checker: {v0:.0f} + sum(transactions) = {true:.0f}, but "
            f"stated final = {final:.0f} (off by {final - true:.0f}). "
            f"UNSATISFIABLE (inconsistent).")


def _reviewer(client, statements, aid) -> int:
    numbered = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(statements))
    user = f"Statements:\n{numbered}\n"
    if aid:
        user += f"\nAID:\n{aid}\n"
    user += "\nReturn the JSON verdict."
    out = client.complete_json(_SYS, user)
    return 1 if bool(out.get("inconsistent", False)) else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--n-per-cell", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reviewer-model", default="claude-haiku-4-5")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    load_dotenv()
    cases = generate_numeric((args.depth,), n_distract=2,
                             n_per_cell=args.n_per_cell, seed=args.seed)
    labels = np.array([1 if c.label == "inconsistent" else 0 for c in cases])

    from beliefnet.llm.anthropic_client import AnthropicClient
    from beliefnet.llm.judge import judge_consistency
    if not model_available("gpt4o"):
        print("[skip] gpt4o not available for rationale aid")
        return 1
    gpt4o = client_for("gpt4o")
    reviewer = AnthropicClient(model=args.reviewer_model)

    # GPT-4o rationale aid (verdict + reason) and its own accuracy.
    rationales, assistant_pred = [], []
    for c in cases:
        r = judge_consistency(gpt4o, c.statements)
        rationales.append(r)
        assistant_pred.append(1 if r["inconsistent"] else 0)
    assistant_pred = np.array(assistant_pred)

    conditions = {
        "reviewer_alone": lambda c, r: None,
        "reviewer_+_certificate": lambda c, r: _numeric_certificate(c),
        "reviewer_+_LLM_rationale":
            lambda c, r: f"Assistant judged "
                         f"{'INCONSISTENT' if r['inconsistent'] else 'CONSISTENT'}. "
                         f"Reasoning: {r.get('reason', '')}",
    }

    rows = []
    for name, aid_fn in conditions.items():
        preds = np.array([_reviewer(reviewer, c.statements, aid_fn(c, r))
                          for c, r in zip(cases, rationales)])
        rows.append({"condition": name,
                     "reviewer_accuracy": round(float((preds == labels).mean()), 4),
                     "kappa_vs_truth": round(cohen_kappa(preds, labels), 4)})
        print(f"  [{name}] acc={rows[-1]['reviewer_accuracy']}")

    rows.append({"condition": "(ref) GPT-4o assistant alone",
                 "reviewer_accuracy": round(float((assistant_pred == labels).mean()), 4),
                 "kappa_vs_truth": round(cohen_kappa(assistant_pred, labels), 4)})

    payload = {
        "title": "Human study (proxy reviewer = Claude Haiku)",
        "question": "Does a sound certificate help a reviewer verify hard numeric "
                    "consistency more than an LLM free-text rationale?",
        "config": {"depth": args.depth, "n_cases": len(cases), "seed": args.seed,
                   "reviewer_model": args.reviewer_model},
        "summary": {"n_cases": len(cases)},
        "tables": {"Reviewer accuracy by aid condition": rows},
        "notes": ["The certificate is a checkable arithmetic proof; the rationale "
                  "propagates the assistant's arithmetic errors. Higher accuracy "
                  "under certificate = measured auditability benefit."],
    }
    save_result(args.out, "exp_human_study" + (f"_{args.tag}" if args.tag else ""),
                payload)
    build_details_md(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
