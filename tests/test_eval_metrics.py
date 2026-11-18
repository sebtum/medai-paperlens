"""Hand-computed fixtures for the pure retrieval metrics (ADR-0023)."""

import math
from dataclasses import FrozenInstanceError

import pytest

from app.evaluation.metrics import (
    Labels,
    RankedHit,
    collapse_sections,
    cut,
    distinct_papers,
    is_empty,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    recall_ceiling_at_k,
    table_hit,
)


def hit(
    paper_id: str, section: str, score: float, index: int = 0, chunk_type: str = "text"
) -> RankedHit:
    return RankedHit(
        paper_id=paper_id,
        section=section,
        chunk_index=index,
        chunk_type=chunk_type,
        score=score,
    )


# Chunks in rank order. Scores are descending, as Qdrant returns them.
HITS = [
    hit("a", "s1", 0.90, 0),
    hit("a", "s1", 0.85, 1),  # duplicate section
    hit("b", "s2", 0.80, 0),
    hit("a", "s3", 0.70, 2),
    hit("c", "s4", 0.65, 0),
    hit("a", "s5", 0.50, 3),  # below t = 0.6
]
LABELS: Labels = {("a", "s1"): 2, ("a", "s3"): 1, ("d", "x"): 2}

# DCG discount for rank r is 1 / log2(r + 1).
D2 = 1 / math.log2(3)
D3 = 1 / math.log2(4)  # 0.5
IDCG_5 = 2 + 2 * D2 + 1 * D3  # grades [2, 2, 1]
IDCG_2 = 2 + 2 * D2  # grades [2, 2]


def test_rankedhit_is_frozen() -> None:
    with pytest.raises(FrozenInstanceError):
        HITS[0].score = 0.1  # type: ignore[misc]


# cut ---------------------------------------------------------------------------------


def test_cut_filters_by_threshold_then_takes_first_k() -> None:
    kept = cut(HITS, k=3, threshold=0.6)
    assert [(h.paper_id, h.section, h.chunk_index) for h in kept] == [
        ("a", "s1", 0),
        ("a", "s1", 1),
        ("b", "s2", 0),
    ]


def test_cut_threshold_applied_before_k() -> None:
    # k = 10 exceeds what passes the threshold: 5 hits >= 0.6.
    assert len(cut(HITS, k=10, threshold=0.6)) == 5


def test_cut_threshold_is_inclusive() -> None:
    assert [h.score for h in cut(HITS, k=10, threshold=0.65)][-1] == 0.65


def test_cut_does_not_reorder_input() -> None:
    unordered = [hit("a", "s1", 0.5), hit("b", "s2", 0.9)]
    assert [h.paper_id for h in cut(unordered, k=2, threshold=0.0)] == ["a", "b"]


def test_cut_k_zero_and_empty_input() -> None:
    assert cut(HITS, k=0, threshold=0.0) == []
    assert cut([], k=5, threshold=0.0) == []


def test_cut_negative_k_raises() -> None:
    with pytest.raises(ValueError, match="k must be"):
        cut(HITS, k=-1, threshold=0.0)


def test_cut_threshold_above_all_scores_is_empty() -> None:
    assert cut(HITS, k=5, threshold=0.95) == []


# collapse_sections -------------------------------------------------------------------


def test_collapse_sections_keeps_first_rank_of_each_section() -> None:
    assert collapse_sections(HITS[:5]) == [
        ("a", "s1"),
        ("b", "s2"),
        ("a", "s3"),
        ("c", "s4"),
    ]


def test_collapse_sections_same_section_name_in_different_papers_is_distinct() -> None:
    hits = [hit("a", "Methods", 0.9), hit("b", "Methods", 0.8)]
    assert collapse_sections(hits) == [("a", "Methods"), ("b", "Methods")]


# recall / mrr / ndcg -----------------------------------------------------------------


def test_recall_counts_labelled_sections_once() -> None:
    # Ranked L at k=5, t=0.6: s1, s2, s3, s4. Labelled found: s1, s3 of 3 labels.
    assert recall_at_k(HITS, LABELS, k=5, threshold=0.6) == pytest.approx(2 / 3)


def test_duplicate_sections_count_once() -> None:
    labels: Labels = {("a", "s1"): 2}
    hits = [hit("a", "s1", 0.9, 0), hit("a", "s1", 0.8, 1), hit("a", "s1", 0.7, 2)]
    assert recall_at_k(hits, labels, k=5, threshold=0.0) == 1.0
    # DCG counts the section once at rank 1: 2 / 1; IDCG = 2 / 1.
    assert ndcg_at_k(hits, labels, k=5, threshold=0.0) == pytest.approx(1.0)


def test_mrr_first_labelled_rank_in_collapsed_list() -> None:
    # Collapsed L: s1(1), s2(2), s3(3), s4(4). Labelled s3 only -> rank 3.
    labels: Labels = {("a", "s3"): 1}
    assert mrr_at_k(HITS, labels, k=5, threshold=0.6) == pytest.approx(1 / 3)


def test_mrr_uses_collapsed_rank_not_chunk_rank() -> None:
    # In chunk order s3 is 4th, but the duplicate s1 collapses, so its rank is 3.
    labels: Labels = {("a", "s3"): 2}
    assert mrr_at_k(HITS, labels, k=5, threshold=0.0) == pytest.approx(1 / 3)


def test_mrr_any_grade_counts() -> None:
    assert mrr_at_k(HITS, LABELS, k=5, threshold=0.6) == 1.0  # s1 is rank 1
    assert mrr_at_k(HITS, {("a", "s3"): 1}, k=5, threshold=0.6) == pytest.approx(1 / 3)


def test_mrr_zero_when_nothing_labelled_retrieved() -> None:
    assert mrr_at_k(HITS, {("z", "none"): 2}, k=5, threshold=0.0) == 0.0


def test_ndcg_hand_computed() -> None:
    # L: s1 (rank 1, grade 2), s2, s3 (rank 3, grade 1), s4.
    dcg = 2 / math.log2(2) + 1 * D3
    assert dcg == pytest.approx(2.5)
    assert ndcg_at_k(HITS, LABELS, k=5, threshold=0.6) == pytest.approx(dcg / IDCG_5)


def test_ndcg_grade_one_only_hit() -> None:
    # Only the grade-1 section is retrieved, at rank 1: DCG = 1; IDCG over [2, 2, 1].
    hits = [hit("a", "s3", 0.9)]
    assert ndcg_at_k(hits, LABELS, k=5, threshold=0.6) == pytest.approx(1 / IDCG_5)
    assert mrr_at_k(hits, LABELS, k=5, threshold=0.6) == 1.0
    assert recall_at_k(hits, LABELS, k=5, threshold=0.6) == pytest.approx(1 / 3)


def test_ndcg_idcg_is_cut_at_k() -> None:
    # k = 2: L is just s1 (both chunks are s1). IDCG uses the top two grades only.
    assert ndcg_at_k(HITS, LABELS, k=2, threshold=0.6) == pytest.approx(2 / IDCG_2)


def test_ndcg_perfect_ranking_is_one() -> None:
    labels: Labels = {("a", "s1"): 2, ("b", "s2"): 1}
    hits = [hit("a", "s1", 0.9), hit("b", "s2", 0.8)]
    assert ndcg_at_k(hits, labels, k=5, threshold=0.0) == pytest.approx(1.0)


def test_ndcg_lower_when_grade_two_ranked_below_grade_one() -> None:
    labels: Labels = {("a", "s1"): 2, ("b", "s2"): 1}
    hits = [hit("b", "s2", 0.9), hit("a", "s1", 0.8)]
    dcg = 1 / math.log2(2) + 2 / math.log2(3)
    idcg = 2 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k(hits, labels, k=5, threshold=0.0) == pytest.approx(dcg / idcg)


def test_relevant_hit_below_threshold_is_lost() -> None:
    labels: Labels = {("a", "s5"): 2}
    assert recall_at_k(HITS, labels, k=10, threshold=0.6) == 0.0
    assert mrr_at_k(HITS, labels, k=10, threshold=0.6) == 0.0
    assert ndcg_at_k(HITS, labels, k=10, threshold=0.6) == 0.0
    # With a lower threshold the hit is found at collapsed rank 5 (s1..s5).
    assert recall_at_k(HITS, labels, k=10, threshold=0.4) == 1.0
    assert mrr_at_k(HITS, labels, k=10, threshold=0.4) == pytest.approx(1 / 5)


def test_relevant_hit_beyond_k_is_lost() -> None:
    labels: Labels = {("a", "s3"): 2}  # 4th chunk in HITS
    assert recall_at_k(HITS, labels, k=3, threshold=0.0) == 0.0
    assert recall_at_k(HITS, labels, k=4, threshold=0.0) == 1.0


def test_metrics_with_no_hits_are_zero() -> None:
    assert recall_at_k([], LABELS, k=5, threshold=0.6) == 0.0
    assert mrr_at_k([], LABELS, k=5, threshold=0.6) == 0.0
    assert ndcg_at_k([], LABELS, k=5, threshold=0.6) == 0.0


def test_k_zero_gives_zero_metrics() -> None:
    assert recall_at_k(HITS, LABELS, k=0, threshold=0.0) == 0.0
    assert mrr_at_k(HITS, LABELS, k=0, threshold=0.0) == 0.0
    assert ndcg_at_k(HITS, LABELS, k=0, threshold=0.0) == 0.0


@pytest.mark.parametrize(
    "metric",
    [
        recall_at_k,
        mrr_at_k,
        ndcg_at_k,
        table_hit,
        precision_at_k,
        recall_ceiling_at_k,
    ],
)
def test_empty_labels_raise(metric) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError, match="labels must not be empty"):
        metric(HITS, {}, k=5, threshold=0.6)


@pytest.mark.parametrize("grade", [0, 3, -1])
def test_invalid_grade_raises(grade: int) -> None:
    with pytest.raises(ValueError, match="grade must be 1 or 2"):
        ndcg_at_k(HITS, {("a", "s1"): grade}, k=5, threshold=0.6)


# precision_at_k / recall_ceiling_at_k -------------------------------------------------


def test_precision_counts_chunks_in_labelled_sections() -> None:
    # Top 5 chunks above 0.6: a:s1, a:s1, b:s2, a:s3, c:s4. Labelled: s1 twice, s3.
    assert precision_at_k(HITS, LABELS, k=5, threshold=0.6) == pytest.approx(3 / 5)
    # k = 2: both chunks are in s1.
    assert precision_at_k(HITS, LABELS, k=2, threshold=0.6) == 1.0


def test_precision_divides_by_returned_chunks_not_k() -> None:
    # Threshold 0.8 keeps 3 chunks (a:s1, a:s1, b:s2): 2 of 3 labelled.
    assert precision_at_k(HITS, LABELS, k=5, threshold=0.8) == pytest.approx(2 / 3)


def test_precision_is_zero_when_nothing_is_returned() -> None:
    assert precision_at_k(HITS, LABELS, k=5, threshold=0.99) == 0.0
    assert precision_at_k(HITS, LABELS, k=0, threshold=0.0) == 0.0
    assert precision_at_k([], LABELS, k=5, threshold=0.6) == 0.0


def test_recall_ceiling_equals_recall_when_labels_fit_in_k() -> None:
    # 3 labels, k = 5: min(k, |R|) = |R|, so the two agree.
    assert recall_ceiling_at_k(HITS, LABELS, k=5, threshold=0.6) == pytest.approx(
        recall_at_k(HITS, LABELS, k=5, threshold=0.6)
    )


def test_recall_ceiling_divides_by_what_k_slots_allow() -> None:
    labels: Labels = {("a", "s1"): 2, ("a", "s3"): 1} | {
        (f"p{i}", "x"): 1 for i in range(5)
    }  # 7 labels, 2 of them found in the top 5
    assert recall_at_k(HITS, labels, k=5, threshold=0.6) == pytest.approx(2 / 7)
    assert recall_ceiling_at_k(HITS, labels, k=5, threshold=0.6) == pytest.approx(2 / 5)


def test_recall_ceiling_is_one_when_k_slots_are_all_labelled() -> None:
    hits = [hit("a", f"s{i}", 0.9 - i * 0.01) for i in range(5)]
    labels: Labels = {("a", f"s{i}"): 1 for i in range(7)}
    assert recall_at_k(hits, labels, k=5, threshold=0.0) == pytest.approx(5 / 7)
    assert recall_ceiling_at_k(hits, labels, k=5, threshold=0.0) == 1.0


def test_recall_ceiling_is_zero_for_k_zero_and_no_hits() -> None:
    assert recall_ceiling_at_k(HITS, LABELS, k=0, threshold=0.0) == 0.0
    assert recall_ceiling_at_k([], LABELS, k=5, threshold=0.6) == 0.0


# is_empty / distinct_papers / table_hit ----------------------------------------------


def test_is_empty_depends_on_threshold() -> None:
    assert not is_empty(HITS, threshold=0.9)  # top score 0.90 passes (inclusive)
    assert is_empty(HITS, threshold=0.91)
    assert is_empty([], threshold=0.0)


def test_distinct_papers_counts_papers_in_top_k_chunks() -> None:
    # Top 3 chunks: a, a, b -> 2 papers. All 5 above 0.6: a, b, c -> 3.
    assert distinct_papers(HITS, k=3, threshold=0.6) == 2
    assert distinct_papers(HITS, k=5, threshold=0.6) == 3
    assert distinct_papers(HITS, k=5, threshold=0.99) == 0


def test_table_hit_needs_table_chunk_in_labelled_section() -> None:
    hits = [
        hit("a", "s1", 0.9, 0, "text"),
        hit("b", "s2", 0.8, 1, "table"),  # table, but not labelled
        hit("a", "s3", 0.7, 2, "table"),  # table in a labelled section
    ]
    labels: Labels = {("a", "s3"): 2}
    assert table_hit(hits, labels, k=3, threshold=0.6)
    assert not table_hit(hits, labels, k=2, threshold=0.6)  # table chunk is 3rd
    assert not table_hit(hits, labels, k=3, threshold=0.75)  # below threshold
    assert not table_hit(
        hits, {("a", "s1"): 2}, k=3, threshold=0.6
    )  # labelled section is text
