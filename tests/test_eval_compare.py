"""Tests for run results, the paired bootstrap, the verdict and the reports.

All fixtures are built in code; no Qdrant, Ollama or data files are needed.
"""

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from app.evaluation.compare import (
    PRIMARY_CUTOFF,
    Comparison,
    Cutoff,
    QuestionRun,
    RunFormatError,
    RunMeta,
    RunRecord,
    TypeComparison,
    compare_runs,
    crowding,
    curve_thresholds,
    empty_rate_on_corpus,
    evidence_tokens,
    load_run,
    max_off_corpus_top1,
    no_result_rate_off_corpus,
    paired_bootstrap_range,
    question_scores,
    run_from_json,
    run_to_json,
    save_run,
    summarize_run,
    table_hit_rate,
    threshold_curve,
    threshold_verdict,
    verdict,
)
from app.evaluation.golden import GoldenEntry, Label, QuestionType
from app.evaluation.metrics import RankedHit
from app.evaluation.report import (
    R9_SENTENCE,
    render_comparison,
    render_run_summary,
    render_threshold_curve,
)

F, T, OFF = QuestionType.FACTUAL, QuestionType.TABLE, QuestionType.OFF_CORPUS


def _entry(qid: str, qtype: QuestionType) -> GoldenEntry:
    relevant = (
        [] if qtype is OFF else [Label(paper_id="p1", section=f"Sec {qid}", grade=2)]
    )
    return GoldenEntry(
        id=qid,
        question=f"question {qid}",
        type=qtype,
        relevant=relevant,
        source_chunks=[] if qtype is OFF else ["p1:0"],
    )


def _golden(spec: dict[QuestionType, int]) -> list[GoldenEntry]:
    entries: list[GoldenEntry] = []
    for qtype, count in spec.items():
        entries.extend(_entry(f"{qtype.value}-{i}", qtype) for i in range(count))
    return entries


def _hit(
    paper: str, section: str, score: float, chunk_type: str = "text", index: int = 0
) -> RankedHit:
    return RankedHit(paper, section, index, chunk_type, score)


def _hits_for(
    entry: GoldenEntry, rank: int | None, score: float = 0.8
) -> tuple[RankedHit, ...]:
    """Hits where the labelled section sits at ``rank`` (1-based), or is absent."""
    filler = [_hit("p9", f"Other {i}", score - 0.01 * i, index=i) for i in range(1, 6)]
    if rank is None or not entry.relevant:
        return tuple(filler)
    label = entry.relevant[0]
    chunk_type = "table" if entry.type is T else "text"
    target = _hit(label.paper_id, label.section, score, chunk_type, index=50)
    filler.insert(rank - 1, target)
    return tuple(filler)


def _run(
    name: str,
    golden: list[GoldenEntry],
    rank_of: Callable[[GoldenEntry], int | None],
    *,
    off_score: float = 0.5,
) -> RunRecord:
    questions = []
    for entry in golden:
        if entry.type is OFF:
            hits: tuple[RankedHit, ...] = (_hit("p9", "Unrelated", off_score),)
        else:
            hits = _hits_for(entry, rank_of(entry))
        questions.append(QuestionRun(id=entry.id, type=entry.type, hits=hits))
    meta = RunMeta(
        name=name,
        config={"embedder": "bge-small", "rewrite": True},
        golden_sha256="a" * 64,
        git_hash="abc1234",
        chunk_count=600,
    )
    return RunRecord(meta=meta, questions=tuple(questions))


# ------------------------------------------------------------------ run file layout


def test_run_json_roundtrip(tmp_path: Path) -> None:
    golden = _golden({F: 2, T: 1, OFF: 1})
    run = _run("baseline", golden, lambda e: 1)
    with_tokens = QuestionRun(
        id="x",
        type=F,
        hits=(_hit("p1", "S", 0.9),),
        ngram_overlap=3,
        chunk_tokens=(120,),
        query="a query",
    )
    run = RunRecord(meta=run.meta, questions=(*run.questions, with_tokens))
    path = tmp_path / "runs" / "baseline.json"
    save_run(run, path)
    assert load_run(path) == run


def test_run_json_has_documented_layout() -> None:
    run = _run("baseline", _golden({F: 1}), lambda e: 1)
    data = json.loads(run_to_json(run))
    assert set(data) == {
        "name",
        "config",
        "golden_sha256",
        "git_hash",
        "chunk_count",
        "questions",
    }
    assert data["questions"][0]["hits"][0][:2] == ["p1", "Sec factual-0"]
    assert len(data["questions"][0]["hits"][0]) == 5


def test_run_from_json_rejects_bad_hit() -> None:
    text = (
        '{"name": "r", "config": {}, "golden_sha256": "x", "git_hash": "y",'
        ' "chunk_count": 1, "questions": ['
        '{"id": "q", "type": "factual", "hits": [["p", "s"]]}]}'
    )
    with pytest.raises(RunFormatError, match="hit must be"):
        run_from_json(text, source="bad.json")


def test_run_from_json_rejects_invalid_json() -> None:
    with pytest.raises(RunFormatError, match="invalid JSON"):
        run_from_json("{", source="bad.json")


# ------------------------------------------------------------------ question scores


def test_question_scores_exclude_off_corpus() -> None:
    golden = _golden({F: 2, OFF: 2})
    run = _run("a", golden, lambda e: 1)
    scores = question_scores(run, golden, metric="ndcg", cutoff=PRIMARY_CUTOFF)
    assert set(scores) == {"factual-0", "factual-1"}
    assert all(value == 1.0 for value in scores.values())


def test_question_scores_follow_cutoff() -> None:
    golden = _golden({F: 1})
    run = _run("a", golden, lambda e: 4)  # labelled section at rank 4, score 0.8-ish
    assert question_scores(run, golden, metric="recall", cutoff=Cutoff(3, 0.0)) == {
        "factual-0": 0.0
    }
    assert question_scores(run, golden, metric="recall", cutoff=Cutoff(5, 0.0)) == {
        "factual-0": 1.0
    }


def test_mismatched_question_ids_fail_loudly() -> None:
    golden = _golden({F: 2})
    run = _run("a", golden[:1], lambda e: 1)
    with pytest.raises(ValueError, match="factual-1"):
        question_scores(run, golden, metric="ndcg", cutoff=PRIMARY_CUTOFF)


def test_type_mismatch_between_run_and_golden_fails() -> None:
    golden = _golden({F: 1})
    run = _run("a", golden, lambda e: 1)
    other = [_entry("factual-0", T)]
    with pytest.raises(ValueError, match="type"):
        question_scores(run, other, metric="ndcg", cutoff=PRIMARY_CUTOFF)


# ------------------------------------------------------------------ bootstrap


def test_bootstrap_all_zero_deltas_gives_zero_range() -> None:
    assert paired_bootstrap_range({"a": [0.0] * 5, "b": [0.0] * 3}) == (0.0, 0.0)


def test_bootstrap_requires_questions() -> None:
    with pytest.raises(ValueError, match="no questions"):
        paired_bootstrap_range({})


def test_bootstrap_is_deterministic_and_seed_dependent() -> None:
    deltas = {"a": [0.5, -0.2, 0.1, 0.0, 0.3, -0.4], "b": [0.2, 0.2, -0.1, 0.6]}
    first = paired_bootstrap_range(deltas, seed=0)
    assert paired_bootstrap_range(deltas, seed=0) == first
    assert paired_bootstrap_range(deltas, seed=1) != first
    lo, hi = first
    assert lo < hi


def test_bootstrap_strata_are_resampled_within_type() -> None:
    # Type "a" is always +1, type "b" always -1, equal sizes: every resample has mean 0.
    # An unstratified bootstrap over the pooled deltas would spread widely.
    lo, hi = paired_bootstrap_range({"a": [1.0] * 4, "b": [-1.0] * 4})
    assert (lo, hi) == (0.0, 0.0)


# ------------------------------------------------------------ comparison + verdict


def test_identical_runs_have_zero_range_all_ties_and_are_not_adopted() -> None:
    golden = _golden({F: 4, T: 3, OFF: 2})
    run = _run("a", golden, lambda e: 1)
    comparison = compare_runs(run, run, golden)
    assert (comparison.lo, comparison.hi, comparison.mean_delta) == (0.0, 0.0, 0.0)
    assert (comparison.wins, comparison.losses, comparison.ties) == (0, 0, 7)
    assert all(item.ties == item.n for item in comparison.per_type)
    assert not verdict(comparison).adopt


def test_better_on_every_question_is_adopted() -> None:
    golden = _golden({F: 5, T: 4, OFF: 2})
    run_a = _run("a", golden, lambda e: None)
    run_b = _run("b", golden, lambda e: 1)
    comparison = compare_runs(run_a, run_b, golden)
    assert comparison.lo > 0
    assert comparison.mean_delta == pytest.approx(1.0)
    assert (comparison.wins, comparison.losses, comparison.ties) == (9, 0, 0)
    result = verdict(comparison)
    assert result.adopt and result.lo_above_zero and result.clearly_worse == ()


def test_one_type_clearly_worse_is_rejected_despite_positive_lower_end() -> None:
    golden = _golden({F: 12, T: 3})
    run_a = _run("a", golden, lambda e: None if e.type is F else 1)
    run_b = _run("b", golden, lambda e: 1 if e.type is F else 2)
    comparison = compare_runs(run_a, run_b, golden)
    assert comparison.lo > 0
    worse = {item.type: item for item in comparison.per_type}[T]
    assert worse.mean_delta < -0.05 and worse.losses == 3 and worse.wins == 0
    result = verdict(comparison)
    assert not result.adopt
    assert result.lo_above_zero
    assert result.clearly_worse == (T,)
    assert "table" in result.reason


def test_clearly_worse_needs_both_conditions() -> None:
    def item(delta: float, wins: int, losses: int) -> TypeComparison:
        return TypeComparison(
            F, 10, 0.5, 0.5 + delta, delta, wins, losses, 10 - wins - losses
        )

    assert item(-0.05, 1, 5).clearly_worse  # boundary is inclusive
    assert not item(-0.04, 1, 5).clearly_worse  # drop too small
    assert not item(-0.30, 5, 5).clearly_worse  # not more losses than wins
    assert not item(-0.30, 6, 4).clearly_worse


def test_off_corpus_is_excluded_from_the_bootstrap() -> None:
    golden = _golden({F: 4, OFF: 3})
    run_a = _run("a", golden, lambda e: 2, off_score=0.5)
    run_b = _run("b", golden, lambda e: 2, off_score=0.9)  # only off-corpus differs
    comparison = compare_runs(run_a, run_b, golden)
    assert comparison.n == 4
    assert comparison.n_off_corpus == 3
    assert (comparison.lo, comparison.hi) == (0.0, 0.0)
    assert comparison.ties == 4
    assert {item.type for item in comparison.per_type} == {F}
    assert comparison.no_result_rate_a == 1.0
    assert comparison.no_result_rate_b == 0.0


def test_comparison_is_deterministic() -> None:
    golden = _golden({F: 6, T: 4, OFF: 2})
    run_a = _run("a", golden, lambda e: 2 if e.id.endswith(("0", "2")) else None)
    run_b = _run("b", golden, lambda e: 1 if e.id.endswith(("0", "1")) else 3)
    assert compare_runs(run_a, run_b, golden) == compare_runs(run_a, run_b, golden)


def test_threshold_comparison_same_run_two_cutoffs() -> None:
    golden = _golden({F: 5, OFF: 3})
    run = _run("a", golden, lambda e: 1, off_score=0.55)
    comparison = compare_runs(
        run, run, golden, cutoff_a=Cutoff(5, 0.6), cutoff_b=Cutoff(5, 0.5)
    )
    assert comparison.no_result_rate_a == 1.0
    assert comparison.no_result_rate_b == 0.0
    assert not threshold_verdict(comparison).adopt
    assert "no-result rate drops" in threshold_verdict(comparison).reason


def test_threshold_verdict_matches_verdict_when_rate_holds() -> None:
    golden = _golden({F: 5, OFF: 3})
    run_a = _run("a", golden, lambda e: None)
    run_b = _run("b", golden, lambda e: 1)
    comparison = compare_runs(run_a, run_b, golden)
    assert threshold_verdict(comparison) == verdict(comparison)


# ------------------------------------------------------------------ run-level measures


def test_crowding_and_table_hit_rate() -> None:
    golden = _golden({F: 1, T: 2})
    # filler hits are all paper p9; the labelled hit is paper p1.
    run = _run("a", golden, lambda e: 1 if e.id == "table-0" else 2)
    cutoff = Cutoff(3, 0.0)
    assert crowding(run, golden, cutoff) == 2.0
    # table-0 and table-1 both have a table chunk in the labelled section within k=3.
    assert table_hit_rate(run, golden, cutoff) == 1.0
    assert table_hit_rate(run, golden, Cutoff(1, 0.0)) == 0.5
    assert (
        table_hit_rate(_run("a", _golden({F: 1}), lambda e: 1), _golden({F: 1}), cutoff)
        is None
    )


def test_empty_rates_and_max_off_corpus_top1() -> None:
    golden = _golden({F: 2, OFF: 2})
    run = _run("a", golden, lambda e: 1, off_score=0.65)
    assert empty_rate_on_corpus(run, golden, 0.6) == 0.0
    assert empty_rate_on_corpus(run, golden, 0.9) == 1.0
    assert no_result_rate_off_corpus(run, golden, 0.6) == 0.0
    assert no_result_rate_off_corpus(run, golden, 0.7) == 1.0
    assert max_off_corpus_top1(run, golden) == 0.65
    on_only = _golden({F: 2})
    on_only_run = _run("b", on_only, lambda e: 1)
    assert no_result_rate_off_corpus(on_only_run, on_only, 0.6) is None
    assert max_off_corpus_top1(on_only_run, on_only) is None


def test_evidence_tokens_uses_chunk_tokens_when_present() -> None:
    golden = _golden({F: 1})
    run = _run("a", golden, lambda e: 1)
    assert (
        evidence_tokens(run, golden, Cutoff(3, 0.0)) is None
    )  # no token counts stored
    question = run.questions[0]
    counted = QuestionRun(
        id=question.id,
        type=question.type,
        hits=question.hits,
        chunk_tokens=tuple(100 * (i + 1) for i in range(len(question.hits))),
    )
    with_tokens = RunRecord(meta=run.meta, questions=(counted,))
    assert evidence_tokens(with_tokens, golden, Cutoff(3, 0.0)) == 600.0
    assert evidence_tokens(with_tokens, golden, Cutoff(2, 0.0)) == 300.0


def test_threshold_curve_has_expected_steps_and_monotone_empty_rate() -> None:
    thresholds = curve_thresholds()
    assert thresholds[0] == 0.4 and thresholds[-1] == 0.8 and len(thresholds) == 21
    golden = _golden({F: 3, OFF: 2})
    run = _run("a", golden, lambda e: 1, off_score=0.55)
    curve = threshold_curve(run, golden)
    assert [point.threshold for point in curve.points] == list(thresholds)
    empties = [p.empty_rate_on_corpus for p in curve.points]
    assert all(rate == 0.0 for rate in empties)  # top hit scores 0.8, kept at t <= 0.8
    narrow = threshold_curve(run, golden, thresholds=[0.7, 0.85])
    assert [p.empty_rate_on_corpus for p in narrow.points] == [0.0, 1.0]
    low, high = curve.points[0], curve.points[-1]
    assert (
        low.no_result_rate_off_corpus == 0.0 and high.no_result_rate_off_corpus == 1.0
    )
    assert low.max_off_corpus_top1 == 0.55
    assert curve.n_on_corpus == 3 and curve.n_off_corpus == 2


def test_summarize_run_numbers() -> None:
    golden = _golden({F: 2, T: 2, OFF: 1})
    run = _run("baseline", golden, lambda e: 1)
    # annotate overlaps
    annotated = RunRecord(
        meta=run.meta,
        questions=tuple(
            QuestionRun(
                id=q.id,
                type=q.type,
                hits=q.hits,
                ngram_overlap=None if q.type is OFF else (2 if q.type is F else 6),
            )
            for q in run.questions
        ),
    )
    summary = summarize_run(annotated, golden)
    assert [row.k for row in summary.by_k] == [3, 5, 10]
    assert all(row.recall == 1.0 and row.ndcg == 1.0 for row in summary.by_k)
    assert summary.n_on_corpus == 4 and summary.n_off_corpus == 1
    by_type = {row.type: row for row in summary.by_type}
    assert by_type[F].mean_overlap == 2.0 and by_type[T].max_overlap == 6
    assert by_type[OFF].ndcg is None and by_type[OFF].mean_overlap is None
    assert by_type[OFF].empty_rate == 1.0  # off-score 0.5 < 0.6
    assert summary.no_result_rate_off_corpus == 1.0
    assert summary.by_k[0].table_hit_rate == 1.0
    # Precision and recall vs ceiling are reported next to the other scores.
    assert all(0.0 <= row.precision <= 1.0 for row in summary.by_k)
    assert all(row.recall_ceiling >= row.recall for row in summary.by_k)
    assert by_type[F].precision is not None and by_type[F].recall_ceiling is not None
    assert by_type[OFF].precision is None and by_type[OFF].recall_ceiling is None


# ------------------------------------------------------------------ reports


def test_run_summary_report_states_config_and_caveat() -> None:
    golden = _golden({F: 2, T: 2, OFF: 1})
    run = _run("baseline", golden, lambda e: 1)
    text = render_run_summary(summarize_run(run, golden))
    assert R9_SENTENCE in text
    assert (
        '"embedder": "bge-small"' in text and "abc1234" in text and "600 chunks" in text
    )
    assert "| k | Recall | nDCG |" in text
    for k in ("| 3 |", "| 5 |", "| 10 |"):
        assert k in text
    assert "Crowding" in text and "Table hit rate" in text and "Mean overlap" in text
    assert "Few questions" in text and "off_corpus (n=1)" in text


def test_comparison_report_states_decision_config_and_caveats() -> None:
    golden = _golden({F: 5, T: 4, OFF: 2})
    run_a = _run("a", golden, lambda e: None)
    run_b = _run("b", golden, lambda e: 1)
    comparison: Comparison = compare_runs(run_a, run_b, golden)
    text = render_comparison(comparison)
    assert R9_SENTENCE in text
    assert "**B replaces A**" in text
    assert "config" in text and "default_rng(0)" in text and "1000 resamples" in text
    assert "Few questions in" in text
    assert "Only 2 off-corpus questions" in text and "coarse" in text

    same = render_comparison(compare_runs(run_a, run_a, golden))
    assert "**A stays**" in same


def test_threshold_curve_report() -> None:
    golden = _golden({F: 3, OFF: 2})
    run = _run("a", golden, lambda e: 1, off_score=0.55)
    text = render_threshold_curve(threshold_curve(run, golden))
    assert R9_SENTENCE in text and "config" in text
    assert "| 0.40 |" in text and "| 0.80 |" in text
    assert "coarse" in text


def test_reports_contain_no_non_ascii_symbols() -> None:
    golden = _golden({F: 3, T: 2, OFF: 2})
    run = _run("a", golden, lambda e: 1)
    texts = [
        render_run_summary(summarize_run(run, golden)),
        render_comparison(compare_runs(run, run, golden)),
        render_threshold_curve(threshold_curve(run, golden)),
    ]
    assert all(text.isascii() for text in texts)
