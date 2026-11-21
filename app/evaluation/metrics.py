"""Pure retrieval metrics over ranked, scored chunk hits (ADR-0023).

A ranked list ``L`` is built from retrieved chunks: keep hits with
``score >= threshold``, cut to the first ``k`` chunks, then collapse to unique
``(paper_id, section)`` pairs at their first rank. Rank is the 1-based position
in ``L``.

Labels are a mapping ``(paper_id, section) -> grade`` (grade 1 or 2). Off-corpus
questions have no labels and are excluded upstream; empty labels raise here.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

SectionKey = tuple[str, str]
"""A ``(paper_id, section)`` pair identifying a labelled section."""

Labels = Mapping[SectionKey, int]
"""Relevance labels: ``(paper_id, section) -> grade`` with grade in {1, 2}."""

TABLE_CHUNK_TYPE = "table"


@dataclass(frozen=True)
class RankedHit:
    """One retrieved chunk with its similarity score, in rank order."""

    paper_id: str
    section: str
    chunk_index: int
    chunk_type: str
    score: float


def cut(hits: Sequence[RankedHit], k: int, threshold: float) -> list[RankedHit]:
    """Keep hits with ``score >= threshold`` (input order), then the first ``k``."""
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")
    kept = [hit for hit in hits if hit.score >= threshold]
    return kept[:k]


def collapse_sections(hits: Sequence[RankedHit]) -> list[SectionKey]:
    """Unique ``(paper_id, section)`` pairs in order of first appearance."""
    seen: set[SectionKey] = set()
    ranked: list[SectionKey] = []
    for hit in hits:
        key = (hit.paper_id, hit.section)
        if key not in seen:
            seen.add(key)
            ranked.append(key)
    return ranked


def _validated(labels: Labels) -> Labels:
    if not labels:
        raise ValueError(
            "labels must not be empty (off-corpus entries are excluded upstream)"
        )
    for key, grade in labels.items():
        if grade not in (1, 2):
            raise ValueError(f"grade must be 1 or 2, got {grade!r} for {key!r}")
    return labels


def _ranked_sections(
    hits: Sequence[RankedHit], k: int, threshold: float
) -> list[SectionKey]:
    return collapse_sections(cut(hits, k, threshold))


def recall_at_k(
    hits: Sequence[RankedHit], labels: Labels, k: int, threshold: float
) -> float:
    """``|L ∩ R| / |R|`` over labelled sections (any grade)."""
    _validated(labels)
    ranked = _ranked_sections(hits, k, threshold)
    found = sum(1 for key in ranked if key in labels)
    return found / len(labels)


def recall_ceiling_at_k(
    hits: Sequence[RankedHit], labels: Labels, k: int, threshold: float
) -> float:
    """Recall against what ``k`` slots allow: ``found / min(k, |R|)``.

    A question with 7 labels cannot score above 5/7 recall at k = 5. This divides by
    the best possible count instead, so 1.0 means "found as many as k slots allow".
    Reported next to recall; it is not used by the decision rule.
    """
    _validated(labels)
    if k <= 0:
        return 0.0
    ranked = _ranked_sections(hits, k, threshold)
    found = sum(1 for key in ranked if key in labels)
    return found / min(k, len(labels))


def precision_at_k(
    hits: Sequence[RankedHit], labels: Labels, k: int, threshold: float
) -> float:
    """Share of the returned top-``k`` chunks that sit in a labelled section.

    Counted over chunks, not collapsed sections: it measures how much of the context
    that would be sent is on target. The denominator is the number of chunks that
    pass the cut (0 when none do). Reported next to recall; it is not used by the
    decision rule.
    """
    _validated(labels)
    kept = cut(hits, k, threshold)
    if not kept:
        return 0.0
    relevant = sum(1 for hit in kept if (hit.paper_id, hit.section) in labels)
    return relevant / len(kept)


def mrr_at_k(
    hits: Sequence[RankedHit], labels: Labels, k: int, threshold: float
) -> float:
    """``1 / rank`` of the first labelled section (any grade), else 0."""
    _validated(labels)
    for rank, key in enumerate(_ranked_sections(hits, k, threshold), start=1):
        if key in labels:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    hits: Sequence[RankedHit], labels: Labels, k: int, threshold: float
) -> float:
    """nDCG with gain = grade and discount ``log2(rank + 1)``.

    IDCG comes from the labels sorted by grade (descending), cut to ``k``.
    """
    _validated(labels)
    ranked = _ranked_sections(hits, k, threshold)
    dcg = sum(
        labels.get(key, 0) / math.log2(rank + 1)
        for rank, key in enumerate(ranked, start=1)
    )
    ideal = sorted(labels.values(), reverse=True)[:k]
    idcg = sum(grade / math.log2(rank + 1) for rank, grade in enumerate(ideal, start=1))
    return dcg / idcg if idcg > 0 else 0.0


def is_empty(hits: Sequence[RankedHit], threshold: float) -> bool:
    """True when no hit reaches ``threshold`` (the no-result case)."""
    return not any(hit.score >= threshold for hit in hits)


def distinct_papers(hits: Sequence[RankedHit], k: int, threshold: float) -> int:
    """Distinct papers among the top-``k`` chunks above ``threshold`` (crowding)."""
    return len({hit.paper_id for hit in cut(hits, k, threshold)})


def table_hit(
    hits: Sequence[RankedHit], labels: Labels, k: int, threshold: float
) -> bool:
    """True when the top-``k`` chunks hold a ``table`` chunk in a labelled section."""
    _validated(labels)
    return any(
        hit.chunk_type == TABLE_CHUNK_TYPE and (hit.paper_id, hit.section) in labels
        for hit in cut(hits, k, threshold)
    )


def wrong_section_share(
    hits: Sequence[RankedHit], labels: Labels, k: int, threshold: float = 0.0
) -> float:
    """Share of the top-``k`` chunks from a labelled paper but an unlabelled section.

    "Right paper, wrong section": the chunk's paper has at least one label for the
    question, but the chunk's section has none. The denominator is the number of chunks
    that pass the cut (0 when none do). ``threshold`` is optional so a reranked run
    (cut at 0) and a cosine run (cut at 0.6) are both measured the way they are scored.
    """
    _validated(labels)
    kept = cut(hits, k, threshold)
    if not kept:
        return 0.0
    papers = {paper_id for paper_id, _ in labels}
    wrong = sum(
        1
        for hit in kept
        if hit.paper_id in papers and (hit.paper_id, hit.section) not in labels
    )
    return wrong / len(kept)
