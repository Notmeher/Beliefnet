"""Out-of-distribution synthetic generators.

Parameterized along five axes (vocabulary substitution, template paraphrase,
length variation, noise injection, subtype subset) without touching the case
schema, so the equilibrium reasoner sees the same typed-constraint graph
regardless of the shift.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from beliefnet.data.synthetic.case_schema import Case
from beliefnet.data.synthetic.multi_statement import _pad, _stratified_split
from beliefnet.data.synthetic.templates import (
    CANONICAL_TEMPLATES,
    TemplateSet,
)
from beliefnet.data.synthetic.vocabulary import (
    DIAGNOSES,
    DIAGNOSIS_CODES,
    DRUGS,
    NAMES,
    ORDER_EVENTS,
    SYMPTOMS,
)


# ---------------------------------------------------------------------------
# Configurable vocabularies
# ---------------------------------------------------------------------------
@dataclass
class VocabPool:
    events: list[str] = field(default_factory=lambda: list(ORDER_EVENTS))
    names: list[str] = field(default_factory=lambda: list(NAMES))
    symptoms: list[str] = field(default_factory=lambda: list(SYMPTOMS))
    diagnoses: list[str] = field(default_factory=lambda: list(DIAGNOSES))
    drugs: list[str] = field(default_factory=lambda: list(DRUGS))


ALL_GENERATOR_SUBTYPES = (
    "temporal_chain", "temporal_cycle",
    "comparative_chain", "comparative_cycle",
    "numeric_sum_consistent", "numeric_sum_inconsistent",
    "independent", "direct_negation", "direct_numeric", "entity_swap",
)


@dataclass
class OodConfig:
    n_per_subtype: int = 200
    min_fillers: int = 0
    max_fillers: int = 3
    seed: int = 0
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15)
    vocab: VocabPool = field(default_factory=VocabPool)
    templates: TemplateSet = field(default_factory=lambda: CANONICAL_TEMPLATES)
    length_range: tuple[int, ...] = (3, 4, 5)
    subtypes: set[str] | None = None
    id_prefix: str = "ood"


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
def _pick(rng, options: list[str]) -> str:
    return rng.choice(options)


def _build_chain(rng, vocab, length, templates):
    items = rng.sample(vocab, length)
    stmts = [_pick(rng, templates).format(a=items[i], b=items[i + 1]) for i in range(length - 1)]
    stmts.append(_pick(rng, templates).format(a=items[0], b=items[length - 1]))
    edges = [(items[i], items[i + 1]) for i in range(length - 1)]
    edges.append((items[0], items[length - 1]))
    return stmts, list(items), edges


def _build_cycle(rng, vocab, length, templates):
    items = rng.sample(vocab, length)
    stmts = [_pick(rng, templates).format(a=items[i], b=items[(i + 1) % length]) for i in range(length)]
    edges = [(items[i], items[(i + 1) % length]) for i in range(length)]
    return stmts, list(items), edges


def generate_ood(config: OodConfig) -> list[Case]:
    rng = random.Random(config.seed)
    raw: list[tuple] = []
    subtypes = set(config.subtypes) if config.subtypes is not None else set(ALL_GENERATOR_SUBTYPES)
    t = config.templates
    v = config.vocab

    # consistent ----------------------------------------------------------
    if "temporal_chain" in subtypes and len(v.events) >= max(config.length_range):
        for _ in range(config.n_per_subtype):
            L = rng.choice(config.length_range)
            if L > len(v.events):
                continue
            stmts, ents, edges = _build_chain(rng, v.events, L, t.temporal_before)
            constr = [{"type": "before", "args": [a, b]} for a, b in edges]
            s, _ = _pad(rng, stmts, set(), config)
            raw.append(("consistent", "none", "temporal_chain", [], s, ents, constr))

    if "comparative_chain" in subtypes and len(v.names) >= max(config.length_range):
        for _ in range(config.n_per_subtype):
            L = rng.choice(config.length_range)
            if L > len(v.names):
                continue
            stmts, ents, edges = _build_chain(rng, v.names, L, t.comparative_greater)
            constr = [{"type": "greater", "args": [a, b]} for a, b in edges]
            s, _ = _pad(rng, stmts, set(), config)
            raw.append(("consistent", "none", "comparative_chain", [], s, ents, constr))

    if "numeric_sum_consistent" in subtypes:
        for _ in range(config.n_per_subtype):
            x, y = rng.randint(10, 200), rng.randint(10, 200)
            T = x + y
            stmts = [
                _pick(rng, t.arm0_count).format(n=x),
                _pick(rng, t.arm1_count).format(n=y),
                _pick(rng, t.total_count).format(n=T),
            ]
            ents = ["ARM0", "ARM1", "TOTAL"]
            constr = [
                {"type": "anchor", "args": ["ARM0"], "values": [float(x)]},
                {"type": "anchor", "args": ["ARM1"], "values": [float(y)]},
                {"type": "anchor", "args": ["TOTAL"], "values": [float(T)]},
                {"type": "sum_eq", "args": ["ARM0", "ARM1", "TOTAL"]},
            ]
            s, _ = _pad(rng, stmts, set(), config)
            raw.append(("consistent", "none", "numeric_sum", [], s, ents, constr))

    if "independent" in subtypes and v.diagnoses and v.drugs and v.symptoms:
        for _ in range(config.n_per_subtype):
            dx, d, sym = rng.choice(v.diagnoses), rng.choice(v.drugs), rng.choice(v.symptoms)
            stmts = [
                _pick(rng, t.diagnosis).format(d=dx),
                _pick(rng, t.drug_active).format(d=d),
                _pick(rng, t.presence_pos).format(s=sym),
            ]
            ents = ["diagnosis_slot", f"drug_active_{d}", f"presence_{sym}"]
            constr = [
                {"type": "anchor", "args": ["diagnosis_slot"],
                 "values": [float(DIAGNOSIS_CODES.get(dx, 0.0))]},
                {"type": "anchor", "args": [f"drug_active_{d}"], "values": [1.0]},
                {"type": "anchor", "args": [f"presence_{sym}"], "values": [1.0]},
            ]
            s, _ = _pad(rng, stmts, set(), config)
            raw.append(("consistent", "none", "independent", [], s, ents, constr))

    # pairwise inconsistent ----------------------------------------------
    if "direct_negation" in subtypes and v.symptoms:
        for _ in range(config.n_per_subtype):
            sym = rng.choice(v.symptoms)
            stmts = [
                _pick(rng, t.presence_pos).format(s=sym),
                _pick(rng, t.presence_neg).format(s=sym),
            ]
            ent = f"presence_{sym}"
            constr = [
                {"type": "anchor", "args": [ent], "values": [1.0]},
                {"type": "anchor", "args": [ent], "values": [-1.0]},
            ]
            s, idx = _pad(rng, stmts, {0, 1}, config)
            raw.append(("inconsistent", "pairwise", "direct_negation", idx, s, [ent], constr))

    if "direct_numeric" in subtypes:
        for _ in range(config.n_per_subtype):
            n1 = rng.randint(40, 400)
            n2 = n1 + rng.randint(40, 200)
            stmts = [
                _pick(rng, t.direct_enrolled).format(n=n1),
                _pick(rng, t.direct_enrolled).format(n=n2),
            ]
            constr = [
                {"type": "anchor", "args": ["trial_enrolled"], "values": [float(n1)]},
                {"type": "anchor", "args": ["trial_enrolled"], "values": [float(n2)]},
            ]
            s, idx = _pad(rng, stmts, {0, 1}, config)
            raw.append(("inconsistent", "pairwise", "direct_numeric", idx, s,
                        ["trial_enrolled"], constr))

    if "entity_swap" in subtypes and len(v.diagnoses) >= 2:
        for _ in range(config.n_per_subtype):
            a, b = rng.sample(v.diagnoses, 2)
            stmts = [
                _pick(rng, t.diagnosis).format(d=a),
                _pick(rng, t.diagnosis).format(d=b),
            ]
            constr = [
                {"type": "anchor", "args": ["diagnosis_slot"],
                 "values": [float(DIAGNOSIS_CODES.get(a, 0.0))]},
                {"type": "anchor", "args": ["diagnosis_slot"],
                 "values": [float(DIAGNOSIS_CODES.get(b, 0.0))]},
            ]
            s, idx = _pad(rng, stmts, {0, 1}, config)
            raw.append(("inconsistent", "pairwise", "entity_swap", idx, s,
                        ["diagnosis_slot"], constr))

    # non-pairwise inconsistent ------------------------------------------
    if "temporal_cycle" in subtypes and len(v.events) >= max(config.length_range):
        for _ in range(config.n_per_subtype):
            L = rng.choice(config.length_range)
            if L > len(v.events):
                continue
            stmts, ents, edges = _build_cycle(rng, v.events, L, t.temporal_before)
            constr = [{"type": "before", "args": [a, b]} for a, b in edges]
            s, idx = _pad(rng, stmts, set(range(L)), config)
            raw.append(("inconsistent", "non_pairwise", "temporal_cycle", idx, s, ents, constr))

    if "numeric_sum_inconsistent" in subtypes:
        for _ in range(config.n_per_subtype):
            x, y = rng.randint(10, 200), rng.randint(10, 200)
            while True:
                x2, y2 = rng.randint(10, 200), rng.randint(10, 200)
                if x2 + y2 != x + y:
                    break
            T = x2 + y2
            stmts = [
                _pick(rng, t.arm0_count).format(n=x),
                _pick(rng, t.arm1_count).format(n=y),
                _pick(rng, t.total_count).format(n=T),
            ]
            ents = ["ARM0", "ARM1", "TOTAL"]
            constr = [
                {"type": "anchor", "args": ["ARM0"], "values": [float(x)]},
                {"type": "anchor", "args": ["ARM1"], "values": [float(y)]},
                {"type": "anchor", "args": ["TOTAL"], "values": [float(T)]},
                {"type": "sum_eq", "args": ["ARM0", "ARM1", "TOTAL"]},
            ]
            s, idx = _pad(rng, stmts, {0, 1, 2}, config)
            raw.append(("inconsistent", "non_pairwise", "numeric_sum", idx, s, ents, constr))

    if "comparative_cycle" in subtypes and len(v.names) >= max(config.length_range):
        for _ in range(config.n_per_subtype):
            L = rng.choice(config.length_range)
            if L > len(v.names):
                continue
            stmts, ents, edges = _build_cycle(rng, v.names, L, t.comparative_greater)
            constr = [{"type": "greater", "args": [a, b]} for a, b in edges]
            s, idx = _pad(rng, stmts, set(range(L)), config)
            raw.append(("inconsistent", "non_pairwise", "comparative_cycle", idx, s, ents, constr))

    # dedup and finalize
    seen: set[tuple] = set()
    cases: list[Case] = []
    idx = 0
    for label, kind, subtype, cidx, statements, entities, constraints in raw:
        key = tuple(statements)
        if key in seen:
            continue
        seen.add(key)
        cases.append(Case(
            id=f"{config.id_prefix}-{idx:06d}",
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


# Disjoint vocab splits used for vocabulary-shift OOD experiments.
def disjoint_vocab(test_fraction: float = 0.35, seed: int = 42) -> tuple[VocabPool, VocabPool]:
    rng = random.Random(seed)

    def split(items: list[str]) -> tuple[list[str], list[str]]:
        shuffled = list(items)
        rng.shuffle(shuffled)
        n_test = max(2, int(round(len(shuffled) * test_fraction)))
        return shuffled[n_test:], shuffled[:n_test]

    e_tr, e_te = split(list(ORDER_EVENTS))
    n_tr, n_te = split(list(NAMES))
    s_tr, s_te = split(list(SYMPTOMS))
    d_tr, d_te = split(list(DIAGNOSES))
    dr_tr, dr_te = split(list(DRUGS))
    train_pool = VocabPool(events=e_tr, names=n_tr, symptoms=s_tr,
                           diagnoses=d_tr, drugs=dr_tr)
    test_pool = VocabPool(events=e_te, names=n_te, symptoms=s_te,
                          diagnoses=d_te, drugs=dr_te)
    return train_pool, test_pool
