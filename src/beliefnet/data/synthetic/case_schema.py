"""Schema for synthetic multi-statement consistency cases.

A ``Case`` is a set of statements that is either jointly consistent or not.
Each case carries, in its ``metadata``:

* ``entities``:    list of unique entity ids referenced in the case.
* ``constraints``: list of typed constraints over those entities
                   (``anchor``, ``before``, ``greater``, ``sum_eq``).

The structured representation lets the equilibrium reasoner consume a clean
typed graph independent of how the perception layer recovered it.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

CONSISTENCY_LABELS = ("consistent", "inconsistent")
CONFLICT_KINDS = ("none", "pairwise", "non_pairwise")
CONSTRAINT_TYPES = ("anchor", "before", "greater", "sum_eq")


@dataclass
class Case:
    id: str
    statements: list[str]
    label: str               # consistent | inconsistent
    conflict_kind: str       # none | pairwise | non_pairwise
    conflict_subtype: str
    conflict_indices: list[int] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    split: str | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def write_cases(path: str | Path, cases: Iterable[Case | dict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w") as fh:
        for c in cases:
            fh.write(json.dumps(c.to_json() if isinstance(c, Case) else c) + "\n")
            n += 1
    return n


def read_cases(path: str | Path) -> list[Case]:
    out: list[Case] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(Case(**json.loads(line)))
    return out


@dataclass
class MultiStmtConfig:
    n_per_subtype: int = 200
    min_fillers: int = 0
    max_fillers: int = 3
    seed: int = 0
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15)
