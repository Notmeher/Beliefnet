"""Numeric emergent inconsistency: many-term global-sum consistency.

A running-balance case: an opening amount, a sequence of signed transactions,
and a stated final balance. The set is jointly consistent iff
``final == opening + sum(transactions)``. No pair of statements conflicts --
only the GLOBAL sum reveals an inconsistency -- so the emergence order is the
whole set. Detecting it requires multi-term signed arithmetic, which stresses an
LLM as the depth grows, while the exact checker solves the linear system in
closed form (anchor + sum_eq fragment).
"""
from __future__ import annotations

import random

from beliefnet.data.synthetic.case_schema import Case
from beliefnet.data.synthetic.vocabulary import NEUTRAL_FILLERS


def _make_case(rng: random.Random, depth: int, n_distract: int, cid: int,
               label: str) -> Case:
    v0 = rng.randint(50, 300)
    deltas = [rng.randint(-60, 90) for _ in range(depth)]
    true_final = v0 + sum(deltas)
    if label == "inconsistent":
        err = rng.choice([-1, 1]) * rng.randint(3, 25)  # subtle but real
        final = true_final + err
    else:
        final = true_final

    ents = ["acc0"] + [f"d{i}" for i in range(1, depth + 1)] + \
           [f"acc{i}" for i in range(1, depth + 1)]
    constraints = [{"type": "anchor", "args": ["acc0"], "values": [float(v0)]}]
    for i in range(1, depth + 1):
        constraints.append({"type": "anchor", "args": [f"d{i}"],
                            "values": [float(deltas[i - 1])]})
        constraints.append({"type": "sum_eq",
                            "args": [f"acc{i - 1}", f"d{i}", f"acc{i}"]})
    constraints.append({"type": "anchor", "args": [f"acc{depth}"],
                        "values": [float(final)]})

    core = [f"The account opened with {v0} dollars."]
    for i, d in enumerate(deltas, 1):
        core.append(f"Transaction {i} added {d} dollars." if d >= 0
                    else f"Transaction {i} withdrew {abs(d)} dollars.")
    core.append(f"The final balance was {final} dollars.")

    fillers = [rng.choice(NEUTRAL_FILLERS) for _ in range(max(0, n_distract))]
    middle = core[1:-1] + fillers
    rng.shuffle(middle)
    statements = [core[0]] + middle + [core[-1]]

    return Case(
        id=f"num-{cid:06d}",
        statements=statements,
        label=label,
        conflict_kind="non_pairwise" if label == "inconsistent" else "none",
        conflict_subtype=f"numeric_chain_d{depth}",
        conflict_indices=[],
        metadata={"n_statements": len(statements), "entities": ents,
                  "constraints": constraints, "depth": depth,
                  "hop_count": depth, "n_distractors": max(0, n_distract)},
    )


def generate_numeric(depths: tuple[int, ...] = (3, 6, 10, 15),
                     n_distract: int = 2, n_per_cell: int = 50,
                     seed: int = 0) -> list[Case]:
    rng = random.Random(seed)
    cases: list[Case] = []
    cid = 0
    for depth in depths:
        for _ in range(n_per_cell):
            for label in ("inconsistent", "consistent"):
                cases.append(_make_case(rng, depth, n_distract, cid, label))
                cid += 1
    return cases
