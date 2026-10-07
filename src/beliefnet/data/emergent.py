"""Controlled generator for *emergent* joint inconsistency.

Produces balanced consistency cases parameterized by:

* ``hop_count`` (kappa) - the emergence order: an inconsistent case is an
  order cycle of length kappa, so every subset of size < kappa is satisfiable
  and only the full set is not. kappa = 2 is a pairwise contradiction;
  kappa >= 3 is emergent (invisible to any pairwise judge).
* ``n_total`` - total statements per case (core + neutral distractors), held
  identical across the consistent / inconsistent pair so surface length is not
  a shortcut.

Each ``Case`` carries the oracle typed graph in ``metadata`` (entities +
constraints) plus ``hop_count``, ``emergence_order`` and ``n_distractors``, so
the exact checker can be used as ground truth and the LLM baselines get only
the raw statements.
"""
from __future__ import annotations

import random

from beliefnet.data.synthetic.case_schema import Case
from beliefnet.data.synthetic.vocabulary import NEUTRAL_FILLERS, ORDER_EVENTS

_BEFORE_SURFACES = (
    "occurred before", "preceded", "happened earlier than",
    "was observed prior to", "took place ahead of",
)


def _render(rng: random.Random, a: str, b: str, surface: str) -> str:
    rel = rng.choice(_BEFORE_SURFACES) if surface == "paraphrase" else "occurred before"
    return f"{a} {rel} {b}."


def _cycle(events: list[str]) -> list[dict]:
    k = len(events)
    return [{"type": "before", "args": [events[i], events[(i + 1) % k]]}
            for i in range(k)]


def _chain(events: list[str]) -> list[dict]:
    """Acyclic order with the same edge count as the cycle (path + transitive)."""
    k = len(events)
    cons = [{"type": "before", "args": [events[i], events[i + 1]]}
            for i in range(k - 1)]
    cons.append({"type": "before", "args": [events[0], events[k - 1]]})
    return cons


def _make_case(rng: random.Random, kappa: int, n_total: int, cid: int,
               label: str, surface: str = "canonical",
               distractor_pool: list[str] | None = None) -> Case:
    n_distract = n_total - kappa
    if label == "inconsistent":
        events = rng.sample(ORDER_EVENTS, kappa)
        constraints = _cycle(events)
        entities = list(events)
        emergence_order: int | None = kappa
        kind = "pairwise" if kappa == 2 else "non_pairwise"
        subtype = f"order_cycle_k{kappa}"
    else:
        if kappa == 2:
            ev = rng.sample(ORDER_EVENTS, 4)
            constraints = [{"type": "before", "args": [ev[0], ev[1]]},
                           {"type": "before", "args": [ev[2], ev[3]]}]
            entities = list(ev)
        else:
            events = rng.sample(ORDER_EVENTS, kappa)
            constraints = _chain(events)
            entities = list(events)
        emergence_order = None
        kind = "none"
        subtype = f"order_chain_k{kappa}"

    core = [_render(rng, c["args"][0], c["args"][1], surface) for c in constraints]
    pool = distractor_pool or NEUTRAL_FILLERS
    fillers = [rng.choice(pool) for _ in range(max(0, n_distract))]
    statements = core + fillers
    rng.shuffle(statements)

    return Case(
        id=f"emg-{cid:06d}",
        statements=statements,
        label=label,
        conflict_kind=kind,
        conflict_subtype=subtype,
        conflict_indices=[],
        metadata={
            "n_statements": len(statements),
            "entities": entities,
            "constraints": constraints,
            "hop_count": kappa,
            "emergence_order": emergence_order,
            "n_distractors": max(0, n_distract),
            "surface": surface,
        },
    )


def generate_emergent(kappa_values: tuple[int, ...] = (2, 3, 5),
                      n_total_values: tuple[int, ...] = (4, 8, 16),
                      n_per_cell: int = 100,
                      seed: int = 0,
                      surface: str = "canonical",
                      distractor_pool: list[str] | None = None) -> list[Case]:
    """Generate a balanced (kappa x n_total) grid of consistency cases."""
    rng = random.Random(seed)
    cases: list[Case] = []
    cid = 0
    for kappa in kappa_values:
        for n_total in n_total_values:
            if n_total < kappa:
                continue
            for _ in range(n_per_cell):
                for label in ("inconsistent", "consistent"):
                    cases.append(_make_case(rng, kappa, n_total, cid, label,
                                            surface, distractor_pool))
                    cid += 1
    return cases
