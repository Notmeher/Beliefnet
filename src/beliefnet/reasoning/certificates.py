"""Certificates: minimal unsatisfiable core and minimal correction set.

Given an UNSAT typed constraint set, these functions return the auditable
artifacts that distinguish BeliefNet from a black-box score:

* ``minimal_unsat_core``   - the smallest subset that is *still* unsatisfiable
  (a proof of contradiction), computed with Junker's QuickXplain using the
  exact checker as the consistency oracle;
* ``minimal_correction_set`` - a smallest-by-greedy subset whose removal
  restores satisfiability (a repair: "retract these to resolve the conflict").

Both operate over the same constraint dicts the rest of BeliefNet uses.
"""
from __future__ import annotations

from beliefnet.reasoning.exact_checker import Constraint, is_satisfiable


def minimal_unsat_core(constraints: list[Constraint]) -> list[Constraint]:
    """Return a minimal unsatisfiable subset (empty if the set is satisfiable)."""
    C = list(constraints)
    if is_satisfiable(C):
        return []
    return _qx([], [], C)


def _qx(background: list[Constraint], delta: list[Constraint],
        C: list[Constraint]) -> list[Constraint]:
    """QuickXplain (Junker 2004). ``background`` is the fixed context."""
    if delta and not is_satisfiable(background):
        # background alone already conflicts; the just-added delta is not needed.
        return []
    if len(C) == 1:
        return list(C)
    k = len(C) // 2
    C1, C2 = C[:k], C[k:]
    D2 = _qx(background + C1, C1, C2)
    D1 = _qx(background + D2, D2, C1)
    return D1 + D2


def minimal_correction_set(constraints: list[Constraint]) -> list[Constraint]:
    """Return a minimal set whose removal restores satisfiability.

    Computed as the complement of a greedily-grown maximal satisfiable subset,
    so it is irreducible (removing fewer constraints would not restore SAT).
    """
    C = list(constraints)
    if is_satisfiable(C):
        return []
    kept: list[Constraint] = []
    kept_idx: set[int] = set()
    for i, c in enumerate(C):
        if is_satisfiable(kept + [c]):
            kept.append(c)
            kept_idx.add(i)
    return [c for i, c in enumerate(C) if i not in kept_idx]
