"""Per-statement supervision dicts -> entity / typed-constraint graph.

Pure function (no learnable parameters): given a list of supervision dicts
(predicted by the perception heads or derived from ground-truth), build:

* ``entities``    - the unique entity ids referenced;
* ``constraints`` - typed constraints (``anchor`` / ``before`` / ``greater``
                    / ``sum_eq``).

Anchor-only statements (presence, diagnosis, drug, arm count, total count,
direct enrolled) emit one anchor each. The ``sum_eq`` constraint that links
ARM0 / ARM1 / TOTAL is added at the case level when all three are present.
"""
from __future__ import annotations

from beliefnet.data.synthetic.vocabulary import DIAGNOSIS_CODES


def build_graph_from_supervisions(supervisions: list[dict]) -> tuple[list[str], list[dict]]:
    entities: list[str] = []
    seen: set[str] = set()

    def add_entity(e: str) -> None:
        if e not in seen:
            seen.add(e)
            entities.append(e)

    constraints: list[dict] = []
    has_arm0 = has_arm1 = has_total = False

    for sup in supervisions:
        t = sup.get("type", "filler")
        if t == "filler":
            continue
        if t == "before":
            e1, e2 = sup.get("entity1"), sup.get("entity2")
            if e1 and e2 and e1 != e2:
                add_entity(e1)
                add_entity(e2)
                constraints.append({"type": "before", "args": [e1, e2]})
        elif t == "greater":
            e1, e2 = sup.get("entity1"), sup.get("entity2")
            if e1 and e2 and e1 != e2:
                add_entity(e1)
                add_entity(e2)
                constraints.append({"type": "greater", "args": [e1, e2]})
        elif t == "arm_count":
            arm_id = "ARM0" if int(sup.get("arm", 0)) == 0 else "ARM1"
            add_entity(arm_id)
            constraints.append({"type": "anchor", "args": [arm_id],
                                "values": [float(sup.get("value", 0.0))]})
            if arm_id == "ARM0":
                has_arm0 = True
            else:
                has_arm1 = True
        elif t == "total_count":
            add_entity("TOTAL")
            constraints.append({"type": "anchor", "args": ["TOTAL"],
                                "values": [float(sup.get("value", 0.0))]})
            has_total = True
        elif t == "direct_enrolled":
            add_entity("trial_enrolled")
            constraints.append({"type": "anchor", "args": ["trial_enrolled"],
                                "values": [float(sup.get("value", 0.0))]})
        elif t == "presence_pos":
            ent = f"presence_{sup.get('symptom', '')}"
            add_entity(ent)
            constraints.append({"type": "anchor", "args": [ent], "values": [1.0]})
        elif t == "presence_neg":
            ent = f"presence_{sup.get('symptom', '')}"
            add_entity(ent)
            constraints.append({"type": "anchor", "args": [ent], "values": [-1.0]})
        elif t == "diagnosis":
            ent = "diagnosis_slot"
            add_entity(ent)
            code = float(DIAGNOSIS_CODES.get(sup.get("diagnosis", ""), 0.0))
            constraints.append({"type": "anchor", "args": [ent], "values": [code]})
        elif t == "drug_active":
            ent = f"drug_active_{sup.get('drug', '')}"
            add_entity(ent)
            constraints.append({"type": "anchor", "args": [ent], "values": [1.0]})

    if has_arm0 and has_arm1 and has_total:
        constraints.append({"type": "sum_eq",
                            "args": ["ARM0", "ARM1", "TOTAL"]})

    return entities, constraints


# Legacy alias.
assemble_constraints = build_graph_from_supervisions
