"""Exact feasibility checker for the typed-constraint fragment.

Decides joint satisfiability of a typed constraint set exactly (no learned
parameters, no float slop), so a positive contradiction verdict is a *proof*.
This is the sound reasoning layer that backs the certificates.

Fragment handled (same schema as the rest of BeliefNet):

* strict order:    ``before(a, b)`` => ``u_a < u_b`` ;
                   ``greater(a, b)`` => ``u_a > u_b``
* linear equality: ``anchor(e, v)`` => ``u_e = v`` ;
                   ``sum_eq(a, b, t)`` => ``u_a + u_b = u_t``

Soundness. The order fragment is UNSAT iff its ``<`` digraph has a directed
cycle; the equality fragment is UNSAT iff the linear system is inconsistent
(checked exactly over the rationals via ``fractions.Fraction``). For the
current schema the two fragments share no entities, so the whole set is SAT iff
both fragments are SAT. If a future schema couples them on a shared entity,
``is_satisfiable`` raises rather than return a possibly-unsound answer.
"""
from __future__ import annotations

from fractions import Fraction

Constraint = dict


def _split(constraints: list[Constraint]) -> tuple[list[Constraint], list[Constraint]]:
    order, linear = [], []
    for c in constraints:
        t = c.get("type")
        if t in ("before", "greater"):
            order.append(c)
        elif t in ("anchor", "sum_eq"):
            linear.append(c)
        else:
            raise ValueError(f"unknown constraint type: {t!r}")
    return order, linear


def _order_vars(order: list[Constraint]) -> set[str]:
    vs: set[str] = set()
    for c in order:
        vs.add(c["args"][0])
        vs.add(c["args"][1])
    return vs


def _linear_vars(linear: list[Constraint]) -> set[str]:
    vs: set[str] = set()
    for c in linear:
        vs.update(c["args"])
    return vs


def _has_order_cycle(order: list[Constraint]) -> bool:
    """True iff the strict ``<`` digraph contains a directed cycle."""
    adj: dict[str, list[str]] = {}

    def add(u: str, v: str) -> None:
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, [])

    for c in order:
        a, b = c["args"][0], c["args"][1]
        if c["type"] == "before":
            add(a, b)          # a < b
        else:                  # greater(a, b): a > b => b < a
            add(b, a)

    color: dict[str, int] = {u: 0 for u in adj}  # 0=unvisited 1=in-stack 2=done

    def dfs(u: str) -> bool:
        color[u] = 1
        for w in adj[u]:
            if color[w] == 1:
                return True
            if color[w] == 0 and dfs(w):
                return True
        color[u] = 2
        return False

    return any(color[u] == 0 and dfs(u) for u in list(adj))


def _linear_consistent(linear: list[Constraint]) -> bool:
    """Exact consistency of the anchor/sum_eq linear equalities over Q."""
    if not linear:
        return True
    var_list = sorted(_linear_vars(linear))
    idx = {v: i for i, v in enumerate(var_list)}
    n = len(var_list)

    rows: list[list[Fraction]] = []
    for c in linear:
        row = [Fraction(0)] * (n + 1)
        if c["type"] == "anchor":
            row[idx[c["args"][0]]] += Fraction(1)
            row[n] = Fraction(c["values"][0]).limit_denominator(10**9)
        else:  # sum_eq: a + b - t = 0
            a, b, t = c["args"]
            row[idx[a]] += Fraction(1)
            row[idx[b]] += Fraction(1)
            row[idx[t]] += Fraction(-1)
        rows.append(row)

    # Gaussian elimination to reduced form.
    pivot = 0
    for col in range(n):
        sel = next((r for r in range(pivot, len(rows)) if rows[r][col] != 0), None)
        if sel is None:
            continue
        rows[pivot], rows[sel] = rows[sel], rows[pivot]
        pv = rows[pivot][col]
        rows[pivot] = [x / pv for x in rows[pivot]]
        for r in range(len(rows)):
            if r != pivot and rows[r][col] != 0:
                f = rows[r][col]
                rows[r] = [x - f * y for x, y in zip(rows[r], rows[pivot])]
        pivot += 1
        if pivot == len(rows):
            break

    # Inconsistent iff a row [0 ... 0 | c != 0] survives.
    for row in rows:
        if all(row[col] == 0 for col in range(n)) and row[n] != 0:
            return False
    return True


def is_satisfiable(constraints: list[Constraint]) -> bool:
    """Exactly decide whether the typed constraint set is jointly satisfiable."""
    order, linear = _split(constraints)
    if _order_vars(order) & _linear_vars(linear):
        raise NotImplementedError(
            "exact checker assumes order and linear fragments share no entities; "
            "a shared entity needs a combined LP (not yet implemented)."
        )
    if _has_order_cycle(order):
        return False
    return _linear_consistent(linear)


# Readable alias.
is_consistent = is_satisfiable
