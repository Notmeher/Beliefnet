"""Multi-statement consistency case generator.

A *case* is a set of statements that is either jointly consistent or not. The
point of this dataset is the distinction between two kinds of inconsistency:

* pairwise   - a directly contradictory pair is present;
* non_pairwise - no single pair conflicts, yet the set is jointly unsatisfiable
                 (temporal cycle, global numeric sum violation, comparative
                 cycle).

Each case carries, in ``metadata``, a structured representation:

* ``entities``:    list of unique entity ids referenced in the case;
* ``constraints``: list of typed constraints over those entities
                   (``anchor``, ``before``, ``greater``, ``sum_eq``).
"""
from __future__ import annotations

import random

from beliefnet.data.synthetic.case_schema import Case, MultiStmtConfig
from beliefnet.data.synthetic.vocabulary import (
    ARMS,
    DIAGNOSES,
    DIAGNOSIS_CODES,
    DRUGS,
    NAMES,
    NEUTRAL_FILLERS,
    ORDER_EVENTS,
    SYMPTOMS,
)


# ---------------------------------------------------------------------------
# Statement + structured-constraint builders
# ---------------------------------------------------------------------------
def _rel(a: str, b: str, rel: str) -> str:
    return f"{a} {rel} {b}."


def _cycle(rng, pool, length: int, rel: str):
    """Directed cycle e0->e1->...->e_{L-1}->e0. Returns
    (statements, conflict_indices_full, items, edges)."""
    items = rng.sample(pool, length)
    stmts = [_rel(items[i], items[(i + 1) % length], rel) for i in range(length)]
    edges = [(items[i], items[(i + 1) % length]) for i in range(length)]
    return stmts, list(range(length)), list(items), edges


def _chain(rng, pool, length: int, rel: str):
    """Acyclic order with same statement count as the cycle (consecutive edges
    plus one transitive edge). Returns
    (statements, conflict_indices_empty, items, edges)."""
    items = rng.sample(pool, length)
    stmts = [_rel(items[i], items[i + 1], rel) for i in range(length - 1)]
    stmts.append(_rel(items[0], items[length - 1], rel))
    edges = [(items[i], items[i + 1]) for i in range(length - 1)]
    edges.append((items[0], items[length - 1]))
    return stmts, [], list(items), edges


def _pad(rng, statements, conflict_idx, cfg) -> tuple[list[str], list[int]]:
    """Interleave neutral fillers and return statements + updated conflict indices."""
    k = rng.randint(cfg.min_fillers, cfg.max_fillers)
    fillers = rng.sample(NEUTRAL_FILLERS, min(k, len(NEUTRAL_FILLERS)))
    tagged = [(s, i) for i, s in enumerate(statements)] + [(f, None) for f in fillers]
    rng.shuffle(tagged)
    final = [s for s, _ in tagged]
    new_idx = [pos for pos, (_, orig) in enumerate(tagged) if orig in conflict_idx]
    return final, sorted(new_idx)


def _anchor(entity: str, value: float) -> dict:
    return {"type": "anchor", "args": [entity], "values": [float(value)]}


def _typed(rel_type: str, a: str, b: str) -> dict:
    return {"type": rel_type, "args": [a, b]}


# ---- consistent ----------------------------------------------------------
def _gen_consistent(rng, cfg):
    out = []
    # transitive-consistent temporal chain
    for _ in range(cfg.n_per_subtype):
        stmts, _, ents, edges = _chain(rng, ORDER_EVENTS, rng.choice([3, 4, 5]), "occurred before")
        constraints = [_typed("before", a, b) for a, b in edges]
        s, _ = _pad(rng, stmts, set(), cfg)
        out.append(("consistent", "none", "temporal_chain", [], s, ents, constraints))
    # numeric sum that adds up
    for _ in range(cfg.n_per_subtype):
        x, y = rng.randint(10, 200), rng.randint(10, 200)
        stmts = [
            f"{ARMS[0].capitalize()} enrolled {x} patients.",
            f"{ARMS[1].capitalize()} enrolled {y} patients.",
            f"The trial enrolled {x + y} patients in total.",
        ]
        ents = ["ARM0", "ARM1", "TOTAL"]
        constraints = [_anchor("ARM0", x), _anchor("ARM1", y),
                       _anchor("TOTAL", x + y),
                       {"type": "sum_eq", "args": ["ARM0", "ARM1", "TOTAL"]}]
        s, _ = _pad(rng, stmts, set(), cfg)
        out.append(("consistent", "none", "numeric_sum", [], s, ents, constraints))
    # comparative-consistent chain, length-varied, order-neutral names
    for _ in range(cfg.n_per_subtype):
        stmts, _, ents, edges = _chain(rng, NAMES, rng.choice([3, 4, 5]), "is older than")
        constraints = [_typed("greater", a, b) for a, b in edges]
        s, _ = _pad(rng, stmts, set(), cfg)
        out.append(("consistent", "none", "comparative_chain", [], s, ents, constraints))
    # bag of independent facts
    for _ in range(cfg.n_per_subtype):
        dx, d, sym = rng.choice(DIAGNOSES), rng.choice(DRUGS), rng.choice(SYMPTOMS)
        stmts = [
            f"The patient was diagnosed with {dx}.",
            f"Drug {d} is administered once daily.",
            f"The patient reported {sym}.",
        ]
        ents = ["diagnosis_slot", f"drug_active_{d}", f"presence_{sym}"]
        constraints = [_anchor("diagnosis_slot", DIAGNOSIS_CODES[dx]),
                       _anchor(f"drug_active_{d}", 1.0),
                       _anchor(f"presence_{sym}", 1.0)]
        s, _ = _pad(rng, stmts, set(), cfg)
        out.append(("consistent", "none", "independent", [], s, ents, constraints))
    return out


# ---- pairwise inconsistent ----------------------------------------------
def _gen_pairwise(rng, cfg):
    out = []
    for _ in range(cfg.n_per_subtype):
        sym = rng.choice(SYMPTOMS)
        stmts = ["The patient has " + sym + ".", "The patient does not have " + sym + "."]
        ent = f"presence_{sym}"
        ents = [ent]
        constraints = [_anchor(ent, 1.0), _anchor(ent, -1.0)]
        s, idx = _pad(rng, stmts, {0, 1}, cfg)
        out.append(("inconsistent", "pairwise", "direct_negation", idx, s, ents, constraints))
    for _ in range(cfg.n_per_subtype):
        n1 = rng.randint(40, 400)
        n2 = n1 + rng.randint(40, 200)
        stmts = [f"The trial enrolled {n1} patients.",
                 f"The trial enrolled {n2} patients."]
        ents = ["trial_enrolled"]
        constraints = [_anchor("trial_enrolled", n1), _anchor("trial_enrolled", n2)]
        s, idx = _pad(rng, stmts, {0, 1}, cfg)
        out.append(("inconsistent", "pairwise", "direct_numeric", idx, s, ents, constraints))
    for _ in range(cfg.n_per_subtype):
        a, b = rng.sample(DIAGNOSES, 2)
        stmts = [f"The patient was diagnosed with {a}.",
                 f"The patient was diagnosed with {b}."]
        ents = ["diagnosis_slot"]
        constraints = [_anchor("diagnosis_slot", DIAGNOSIS_CODES[a]),
                       _anchor("diagnosis_slot", DIAGNOSIS_CODES[b])]
        s, idx = _pad(rng, stmts, {0, 1}, cfg)
        out.append(("inconsistent", "pairwise", "entity_swap", idx, s, ents, constraints))
    return out


# ---- non-pairwise inconsistent ------------------------------------------
def _gen_non_pairwise(rng, cfg):
    out = []
    # temporal cycle (every pair individually fine)
    for _ in range(cfg.n_per_subtype):
        stmts, cyc, ents, edges = _cycle(rng, ORDER_EVENTS, rng.choice([3, 4, 5]), "occurred before")
        constraints = [_typed("before", a, b) for a, b in edges]
        s, idx = _pad(rng, stmts, set(cyc), cfg)
        out.append(("inconsistent", "non_pairwise", "temporal_cycle", idx, s, ents, constraints))
    # numeric global: stated total is a VALID sum of different arms
    for _ in range(cfg.n_per_subtype):
        x, y = rng.randint(10, 200), rng.randint(10, 200)
        while True:
            x2, y2 = rng.randint(10, 200), rng.randint(10, 200)
            if x2 + y2 != x + y:
                break
        T = x2 + y2
        stmts = [
            f"{ARMS[0].capitalize()} enrolled {x} patients.",
            f"{ARMS[1].capitalize()} enrolled {y} patients.",
            f"The trial enrolled {T} patients in total.",
        ]
        ents = ["ARM0", "ARM1", "TOTAL"]
        constraints = [_anchor("ARM0", x), _anchor("ARM1", y),
                       _anchor("TOTAL", T),
                       {"type": "sum_eq", "args": ["ARM0", "ARM1", "TOTAL"]}]
        s, idx = _pad(rng, stmts, {0, 1, 2}, cfg)
        out.append(("inconsistent", "non_pairwise", "numeric_sum", idx, s, ents, constraints))
    # comparative cycle, length-varied
    for _ in range(cfg.n_per_subtype):
        stmts, cyc, ents, edges = _cycle(rng, NAMES, rng.choice([3, 4, 5]), "is older than")
        constraints = [_typed("greater", a, b) for a, b in edges]
        s, idx = _pad(rng, stmts, set(cyc), cfg)
        out.append(("inconsistent", "non_pairwise", "comparative_cycle", idx, s, ents, constraints))
    return out


def _stratified_split(cases: list[Case], ratios, rng) -> None:
    from collections import defaultdict
    by_key: dict[tuple, list[Case]] = defaultdict(list)
    for c in cases:
        by_key[(c.label, c.conflict_subtype)].append(c)
    for items in by_key.values():
        rng.shuffle(items)
        n = len(items)
        n_tr, n_va = int(n * ratios[0]), int(n * ratios[1])
        for i, c in enumerate(items):
            c.split = "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")


def generate(config: MultiStmtConfig | None = None) -> list[Case]:
    config = config or MultiStmtConfig()
    rng = random.Random(config.seed)
    raw = (_gen_consistent(rng, config) + _gen_pairwise(rng, config)
           + _gen_non_pairwise(rng, config))

    seen: set[tuple] = set()
    cases: list[Case] = []
    idx = 0
    for label, kind, subtype, cidx, statements, entities, constraints in raw:
        key = tuple(statements)
        if key in seen:
            continue
        seen.add(key)
        cases.append(Case(
            id=f"case-{idx:06d}",
            statements=statements,
            label=label,
            conflict_kind=kind,
            conflict_subtype=subtype,
            conflict_indices=cidx,
            metadata={"n_statements": len(statements),
                      "entities": list(entities),
                      "constraints": list(constraints)},
        ))
        idx += 1

    _stratified_split(cases, config.ratios, rng)
    return cases
