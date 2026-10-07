"""Map a SciFact (claim, cited abstracts, evidence labels) tuple into a
BeliefNet ``Case``.

SciFact is a single-claim verification task: each claim is paired with one
or more cited abstracts, and each abstract carries an evidence label
(``SUPPORT`` / ``CONTRADICT`` / no evidence). We treat each (claim,
abstract) pair as one multi-statement case:

* ``statements = [claim] + abstract_sentences``;
* ``label`` is ``"consistent"`` if the abstract supports the claim,
  ``"inconsistent"`` if it contradicts;
* ``conflict_kind`` is ``"pairwise"`` (the claim contradicts at least one
  evidence sentence) for inconsistent cases, ``"none"`` for consistent;
* ``conflict_subtype`` is ``"scifact_support"`` / ``"scifact_contradict"``;
* ``conflict_indices`` lists positions in ``statements`` of evidence
  sentences flagged by the SciFact annotators.

Typed-constraint extraction (``metadata.entities`` and
``metadata.constraints``) is left empty here. Populating it requires a
SciFact-aware ``BeliefParser``, which is the headline research item for the
real-data adaptation phase. The synthetic-style ``Case`` schema is
preserved exactly so the equilibrium core sees the same structure.
"""
from __future__ import annotations

from beliefnet.data.synthetic.case_schema import Case


def claim_evidence_to_case(
    case_id: str,
    claim: str,
    abstract_sentences: list[str],
    evidence_label: str | None,
    evidence_sentence_idx: list[int] | None = None,
) -> Case:
    """Convert one (claim, abstract, evidence label) tuple into a ``Case``.

    Args:
        case_id: stable identifier (e.g. ``f"scifact-{claim_id}-{doc_id}"``).
        claim: the SciFact claim sentence (becomes ``statements[0]``).
        abstract_sentences: ordered abstract sentences from the cited corpus.
        evidence_label: one of ``"SUPPORT"``, ``"CONTRADICT"``, or ``None``
            (no evidence annotated -> treated as consistent for now).
        evidence_sentence_idx: positions within ``abstract_sentences`` that
            the SciFact annotator flagged as evidence; the indices are
            shifted by ``+1`` to account for the claim at position 0.

    Returns:
        A ``Case`` ready to feed ``pack_graphs`` once a real-data
        ``BeliefParser`` populates ``metadata.constraints``.
    """
    statements = [claim] + list(abstract_sentences)
    if evidence_label == "CONTRADICT":
        label = "inconsistent"
        conflict_kind = "pairwise"
        subtype = "scifact_contradict"
    elif evidence_label == "SUPPORT":
        label = "consistent"
        conflict_kind = "none"
        subtype = "scifact_support"
    else:
        label = "consistent"
        conflict_kind = "none"
        subtype = "scifact_no_evidence"

    if conflict_kind == "pairwise" and evidence_sentence_idx:
        conflict_indices = [0] + [i + 1 for i in evidence_sentence_idx]
    else:
        conflict_indices = []

    return Case(
        id=case_id,
        statements=statements,
        label=label,
        conflict_kind=conflict_kind,
        conflict_subtype=subtype,
        conflict_indices=conflict_indices,
        metadata={
            "n_statements": len(statements),
            "entities": [],
            "constraints": [],
            "source": "scifact",
            "evidence_label": evidence_label,
        },
    )
