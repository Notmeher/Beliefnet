"""Experiment: cost / latency / certificate comparison (real).

On the same cases, measures per-decision wall-clock latency and accuracy for
(a) BeliefNet's exact checker, (b) GPT-4o judge, (c) GPT-5 judge, and records
which systems return an auditable certificate. Given that a reasoning LLM can be
*accurate* on clean emergent inconsistency, this quantifies the actual
deployment advantage: BeliefNet is orders of magnitude cheaper and returns a
proof; the LLMs do not.

    PYTHONPATH=src python scripts/run_cost_comparison.py --models gpt4o gpt5
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from beliefnet.data.emergent import generate_emergent
from beliefnet.eval.reporting import build_details_md, save_result
from beliefnet.llm.azure_client import client_for, load_dotenv, model_available
from beliefnet.models.pipeline import BeliefNet
from beliefnet.perception.belief_parser import RuleBeliefParser


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kappa", type=int, nargs="+", default=[3, 5, 7])
    ap.add_argument("--n-total", type=int, nargs="+", default=[8])
    ap.add_argument("--n-per-cell", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--models", nargs="*", default=["gpt4o", "gpt5"])
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    load_dotenv()
    cases = generate_emergent(tuple(args.kappa), tuple(args.n_total),
                              n_per_cell=args.n_per_cell, seed=args.seed)
    labels = np.array([1 if c.label == "inconsistent" else 0 for c in cases])

    rows = []

    # BeliefNet exact checker (rule perception + sound reasoning).
    net = BeliefNet(RuleBeliefParser())
    lat, pred = [], []
    for c in cases:
        t0 = time.perf_counter()
        out = net.predict(c.statements)
        lat.append((time.perf_counter() - t0) * 1000.0)
        pred.append(1 if out["inconsistent"] else 0)
    rows.append({"system": "BeliefNet (exact)",
                 "mean_latency_ms": round(float(np.mean(lat)), 3),
                 "accuracy": round(float((np.array(pred) == labels).mean()), 4),
                 "certificate": "yes (unsat core + repair)"})

    # LLM judges.
    from beliefnet.llm.judge import judge_consistency
    for m in args.models:
        if not model_available(m):
            print(f"[skip] {m}: Azure env vars not set")
            continue
        client = client_for(m)
        lat, pred = [], []
        for i, c in enumerate(cases):
            t0 = time.perf_counter()
            r = judge_consistency(client, c.statements)
            lat.append((time.perf_counter() - t0) * 1000.0)
            pred.append(1 if r["inconsistent"] else 0)
            if (i + 1) % 10 == 0:
                print(f"  [{m}] {i + 1}/{len(cases)}")
        rows.append({"system": f"{m} judge",
                     "mean_latency_ms": round(float(np.mean(lat)), 1),
                     "accuracy": round(float((np.array(pred) == labels).mean()), 4),
                     "certificate": "no (opaque yes/no)"})

    # Speedup vs the slowest LLM, for the summary.
    exact_ms = rows[0]["mean_latency_ms"]
    llm_ms = [r["mean_latency_ms"] for r in rows[1:]] or [exact_ms]
    speedup = round(max(llm_ms) / max(exact_ms, 1e-6), 1)

    payload = {
        "title": "Experiment — Cost / latency / certificate comparison",
        "question": "Given a reasoning LLM can be accurate, what is BeliefNet's "
                    "deployment advantage (speed + proof)?",
        "config": {"kappa": args.kappa, "n_total": args.n_total,
                   "n_per_cell": args.n_per_cell, "seed": args.seed,
                   "n_cases": len(cases)},
        "summary": {"belieftnet_speedup_vs_slowest_llm": f"{speedup}x",
                    "belieftnet_returns_proof": True,
                    "llm_returns_proof": False},
        "tables": {"Per-decision cost / accuracy / certificate": rows},
        "notes": ["The exact checker decides in ~microseconds and returns a "
                  "human-checkable proof; the LLMs take seconds and return an "
                  "opaque verdict. Accuracy parity (on clean text) makes cost + "
                  "verifiability the differentiator."],
    }
    save_result(args.out, "exp_cost_comparison", payload)
    build_details_md(args.out)
    print(f"[cost] BeliefNet ~{exact_ms:.3f} ms/decision; "
          f"speedup vs slowest LLM ~{speedup}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
