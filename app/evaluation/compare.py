"""Run results, per-question scoring, paired bootstrap and the decision rule (ADR-0023).

A *run* is one retrieval configuration scored on the whole golden set: for every
question the top-20 chunks at threshold 0, with scores kept. Any ``(k, threshold)`` is
applied afterwards with the pure functions in ``metrics`` (``cut`` keeps hits with
``score >= t`` and then the first ``k``).

``RunRecord`` is the in-memory form of ``data/eval/runs/<name>.json``. It holds no
paper text: only ids, section labels, scores and counts. ``save_run`` / ``load_run``
convert it to and from JSON. This module never talks to Qdrant, so a comparison works
from two run files plus ``golden.jsonl`` alone.

Off-corpus entries have no labels. They are excluded from nDCG, MRR and Recall and only
feed the no-result rate (and the max off-corpus top-1 score).

Decision rule (``verdict``): primary metric nDCG@5 at t = 0.6. B replaces A only if the
lower end of the 95 % bootstrap range of ``mean(B - A)`` is above 0 and no question type
is *clearly worse* (mean delta <= -0.05 and more losses than wins).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, get_args

import numpy as np

from app.evaluation.golden import GoldenEntry, QuestionType
from app.evaluation.metrics import (
    Labels,
    RankedHit,
    distinct_papers,
    is_empty,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    recall_ceiling_at_k,
    table_hit,
    wrong_section_share,
)

Metric = Literal["recall", "mrr", "ndcg", "precision", "recall_ceiling"]

PRIMARY_METRIC: Metric = "ndcg"
DEFAULT_RESAMPLES = 1000
DEFAULT_SEED = 0
CI_LEVELS = (2.5, 97.5)
TIE_EPSILON = 1e-9
CLEARLY_WORSE_DELTA = -0.05
REPORT_KS: tuple[int, ...] = (3, 5, 10)
# A type with fewer questions than this gets a "few questions" note in reports.
FEW_QUESTIONS = 10

_METRIC_FUNCTIONS: dict[
    str, Callable[[Sequence[RankedHit], Labels, int, float], float]
] = {
    "recall": recall_at_k,
    "mrr": mrr_at_k,
    "ndcg": ndcg_at_k,
    "precision": precision_at_k,
    "recall_ceiling": recall_ceiling_at_k,
}


@dataclass(frozen=True)
class Cutoff:
    """A ``(k, threshold)`` pair applied to the stored top-20 hits."""

    k: int
    threshold: float


PRIMARY_CUTOFF = Cutoff(k=5, threshold=0.6)

RERANK_THRESHOLD = 0.0
"""Default cut-off for a reranked run: no cut. The cosine pre-filter already ran."""


# --------------------------------------------------------------------------- run model


class RunFormatError(ValueError):
    """A run file is unreadable or does not match the run layout."""


@dataclass(frozen=True)
class RunMeta:
    """What a run was (R4): config, golden version, code version, corpus size."""

    name: str
    config: dict[str, Any]
    golden_sha256: str
    git_hash: str
    chunk_count: int

    @property
    def score_kind(self) -> str:
        """What the hit scores mean: ``cosine`` or ``rerank``.

        Runs written before Phase 10 have no ``score_kind`` in their config; they are
        cosine runs.
        """
        return str(self.config.get("score_kind", "cosine"))


def default_threshold(meta: RunMeta) -> float:
    """Default cut-off threshold for a run: 0.6 for cosine scores, 0 for reranked."""
    return RERANK_THRESHOLD if meta.score_kind == "rerank" else PRIMARY_CUTOFF.threshold


@dataclass(frozen=True)
class QuestionRun:
    """One question's retrieval result.

    ``hits`` are in rank order (best first, top-20 at threshold 0).
    ``ngram_overlap`` is the longest word n-gram shared with the source chunks (U3
    probe), ``None`` when not measured (off-corpus). ``chunk_tokens`` is an optional
    token estimate per hit, parallel to ``hits`` (no text stored); empty when unknown.
    ``query`` is the query text sent to the embedder (our own text, may be empty).
    ``rerank_ms`` is the reranker time for this question in milliseconds (``None``
    when no reranker ran).
    """

    id: str
    type: QuestionType
    hits: tuple[RankedHit, ...]
    ngram_overlap: int | None = None
    chunk_tokens: tuple[int, ...] = ()
    query: str = ""
    rerank_ms: float | None = None


@dataclass(frozen=True)
class RunRecord:
    """A whole run: metadata plus one ``QuestionRun`` per golden question."""

    meta: RunMeta
    questions: tuple[QuestionRun, ...] = field(default_factory=tuple)


def _question_to_json(question: QuestionRun) -> dict[str, Any]:
    data: dict[str, Any] = {
        "id": question.id,
        "type": question.type.value,
        "query": question.query,
        "ngram_overlap": question.ngram_overlap,
        "hits": [
            [h.paper_id, h.section, h.chunk_index, h.chunk_type, h.score]
            for h in question.hits
        ],
    }
    if question.chunk_tokens:
        data["chunk_tokens"] = list(question.chunk_tokens)
    if question.rerank_ms is not None:
        data["rerank_ms"] = round(question.rerank_ms, 3)
    return data


def run_to_json(run: RunRecord) -> str:
    """Serialise a run: a small header, then one question per line (diff-friendly)."""
    head = {
        "name": run.meta.name,
        "config": run.meta.config,
        "golden_sha256": run.meta.golden_sha256,
        "git_hash": run.meta.git_hash,
        "chunk_count": run.meta.chunk_count,
    }
    lines = [
        json.dumps(_question_to_json(q), ensure_ascii=False) for q in run.questions
    ]
    head_text = json.dumps(head, indent=2, ensure_ascii=False)
    body = ",\n".join(f"    {line}" for line in lines)
    # Splice "questions" into the header object, before its closing brace.
    return f'{head_text[:-2]},\n  "questions": [\n{body}\n  ]\n}}\n'


def save_run(run: RunRecord, path: Path | str) -> None:
    """Write ``run`` to ``path`` (parent directories are created)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(run_to_json(run), encoding="utf-8")


def _hit_from_json(raw: Any, where: str) -> RankedHit:
    if not isinstance(raw, list) or len(raw) != 5:
        raise RunFormatError(
            f"{where}: a hit must be "
            "[paper_id, section, chunk_index, chunk_type, score]"
        )
    paper_id, section, chunk_index, chunk_type, score = raw
    if not (
        isinstance(paper_id, str)
        and isinstance(section, str)
        and isinstance(chunk_index, int)
        and isinstance(chunk_type, str)
        and isinstance(score, (int, float))
    ):
        raise RunFormatError(f"{where}: hit has a field of the wrong type: {raw!r}")
    return RankedHit(paper_id, section, chunk_index, chunk_type, float(score))


def run_from_json(text: str, *, source: str = "<string>") -> RunRecord:
    """Parse the JSON written by ``run_to_json``; errors name ``source``."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RunFormatError(f"{source}: invalid JSON: {exc.msg}") from exc
    if not isinstance(data, dict):
        raise RunFormatError(f"{source}: top level must be an object")
    try:
        meta = RunMeta(
            name=str(data["name"]),
            config=dict(data["config"]),
            golden_sha256=str(data["golden_sha256"]),
            git_hash=str(data["git_hash"]),
            chunk_count=int(data["chunk_count"]),
        )
        raw_questions = data["questions"]
    except (KeyError, TypeError, ValueError) as exc:
        raise RunFormatError(f"{source}: bad or missing header field: {exc}") from exc
    if not isinstance(raw_questions, list):
        raise RunFormatError(f"{source}: questions must be a list")
    questions: list[QuestionRun] = []
    for position, raw in enumerate(raw_questions):
        where = f"{source}: questions[{position}]"
        try:
            hits = tuple(
                _hit_from_json(item, f"{where}.hits[{i}]")
                for i, item in enumerate(raw["hits"])
            )
            overlap = raw.get("ngram_overlap")
            tokens = tuple(int(t) for t in raw.get("chunk_tokens", ()))
            rerank_ms = raw.get("rerank_ms")
            questions.append(
                QuestionRun(
                    id=str(raw["id"]),
                    type=QuestionType(raw["type"]),
                    hits=hits,
                    ngram_overlap=None if overlap is None else int(overlap),
                    chunk_tokens=tokens,
                    query=str(raw.get("query", "")),
                    rerank_ms=None if rerank_ms is None else float(rerank_ms),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RunFormatError(f"{where}: {exc}") from exc
    return RunRecord(meta=meta, questions=tuple(questions))


def load_run(path: Path | str) -> RunRecord:
    """Read a run file written by ``save_run``."""
    source = Path(path)
    return run_from_json(source.read_text(encoding="utf-8"), source=str(source))


# ------------------------------------------------------------------ per-question scores


def labels_of(entry: GoldenEntry) -> dict[tuple[str, str], int]:
    """``{(paper_id, section): grade}`` as the metric functions expect."""
    return {(label.paper_id, label.section): label.grade for label in entry.relevant}


def _joined(
    run: RunRecord, golden: Sequence[GoldenEntry]
) -> list[tuple[GoldenEntry, QuestionRun]]:
    """Pair every golden entry with its question in ``run``; fail on any mismatch."""
    by_id = {q.id: q for q in run.questions}
    problems: list[str] = []
    if len(by_id) != len(run.questions):
        problems.append("run has duplicate question ids")
    golden_ids = {entry.id for entry in golden}
    for missing in sorted(golden_ids - by_id.keys()):
        problems.append(f"golden question {missing} is not in run {run.meta.name!r}")
    for extra in sorted(by_id.keys() - golden_ids):
        problems.append(f"run {run.meta.name!r} has unknown question {extra}")
    pairs: list[tuple[GoldenEntry, QuestionRun]] = []
    for entry in golden:
        question = by_id.get(entry.id)
        if question is None:
            continue
        if question.type is not entry.type:
            problems.append(
                f"{entry.id}: type {question.type.value} in run, "
                f"{entry.type.value} in golden"
            )
        pairs.append((entry, question))
    if problems:
        raise ValueError("; ".join(problems))
    return pairs


def on_corpus(golden: Sequence[GoldenEntry]) -> list[GoldenEntry]:
    """Golden entries that have labels (everything except off-corpus)."""
    return [entry for entry in golden if entry.type is not QuestionType.OFF_CORPUS]


def question_scores(
    run: RunRecord,
    golden: Sequence[GoldenEntry],
    *,
    metric: Metric = PRIMARY_METRIC,
    cutoff: Cutoff = PRIMARY_CUTOFF,
) -> dict[str, float]:
    """Per-question score ``{question id: value}`` for on-corpus questions only."""
    if metric not in get_args(Metric):
        raise ValueError(f"unknown metric {metric!r}")
    function = _METRIC_FUNCTIONS[metric]
    return {
        entry.id: function(question.hits, labels_of(entry), cutoff.k, cutoff.threshold)
        for entry, question in _joined(run, golden)
        if entry.type is not QuestionType.OFF_CORPUS
    }


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _rate(flags: Sequence[bool]) -> float | None:
    return sum(flags) / len(flags) if flags else None


def no_result_rate_off_corpus(
    run: RunRecord, golden: Sequence[GoldenEntry], threshold: float
) -> float | None:
    """Share of off-corpus questions with no hit at ``threshold`` (higher is better)."""
    flags = [
        is_empty(question.hits, threshold)
        for entry, question in _joined(run, golden)
        if entry.type is QuestionType.OFF_CORPUS
    ]
    return _rate(flags)


def empty_rate_on_corpus(
    run: RunRecord, golden: Sequence[GoldenEntry], threshold: float
) -> float | None:
    """Share of on-corpus questions with no hit at ``threshold`` (lower is better)."""
    flags = [
        is_empty(question.hits, threshold)
        for entry, question in _joined(run, golden)
        if entry.type is not QuestionType.OFF_CORPUS
    ]
    return _rate(flags)


def max_off_corpus_top1(run: RunRecord, golden: Sequence[GoldenEntry]) -> float | None:
    """Highest top-1 score over off-corpus questions; ``None`` if none has any hit."""
    scores = [
        max(hit.score for hit in question.hits)
        for entry, question in _joined(run, golden)
        if entry.type is QuestionType.OFF_CORPUS and question.hits
    ]
    return max(scores) if scores else None


def crowding(
    run: RunRecord, golden: Sequence[GoldenEntry], cutoff: Cutoff
) -> float | None:
    """Mean distinct papers in the top-k chunks over on-corpus questions."""
    values = [
        float(distinct_papers(question.hits, cutoff.k, cutoff.threshold))
        for entry, question in _joined(run, golden)
        if entry.type is not QuestionType.OFF_CORPUS
    ]
    return _mean(values) if values else None


def table_hit_rate(
    run: RunRecord, golden: Sequence[GoldenEntry], cutoff: Cutoff
) -> float | None:
    """Share of ``table`` questions with a table chunk in a labelled top-k section."""
    flags = [
        table_hit(question.hits, labels_of(entry), cutoff.k, cutoff.threshold)
        for entry, question in _joined(run, golden)
        if entry.type is QuestionType.TABLE
    ]
    return _rate(flags)


def evidence_tokens(
    run: RunRecord, golden: Sequence[GoldenEntry], cutoff: Cutoff
) -> float | None:
    """Mean token estimate of the top-k evidence over on-corpus questions.

    Needs ``QuestionRun.chunk_tokens`` (the runner fills it from the chunk text it reads
    at run time; no text is stored). ``None`` when no question carries token counts.
    This is the raw sum of the kept chunks; the generator's budget guard is not applied.
    """
    values: list[float] = []
    for entry, question in _joined(run, golden):
        if entry.type is QuestionType.OFF_CORPUS or not question.chunk_tokens:
            continue
        if len(question.chunk_tokens) != len(question.hits):
            raise ValueError(
                f"{question.id}: chunk_tokens has {len(question.chunk_tokens)} entries "
                f"for {len(question.hits)} hits"
            )
        kept = [
            tokens
            for hit, tokens in zip(question.hits, question.chunk_tokens, strict=True)
            if hit.score >= cutoff.threshold
        ][: cutoff.k]
        values.append(float(sum(kept)))
    return _mean(values) if values else None


# --------------------------------------------------------------------- run summary


@dataclass(frozen=True)
class KRow:
    """Run-level numbers at one k (threshold fixed by the summary)."""

    k: int
    recall: float
    ndcg: float
    mrr: float
    precision: float
    recall_ceiling: float
    crowding: float | None
    table_hit_rate: float | None
    evidence_tokens: float | None
    wrong_section_share: float | None = None


@dataclass(frozen=True)
class RerankTiming:
    """Reranker time per question in milliseconds: median and 90th percentile."""

    n: int
    median_ms: float
    p90_ms: float


@dataclass(frozen=True)
class TypeRow:
    """Per-type numbers at the primary cut-off; overlap is the U3 n-gram probe."""

    type: QuestionType
    n: int
    recall: float | None
    ndcg: float | None
    mrr: float | None
    precision: float | None
    recall_ceiling: float | None
    empty_rate: float
    mean_overlap: float | None
    max_overlap: int | None


@dataclass(frozen=True)
class RunSummary:
    meta: RunMeta
    threshold: float
    n_on_corpus: int
    n_off_corpus: int
    by_k: tuple[KRow, ...]
    by_type: tuple[TypeRow, ...]
    empty_rate_on_corpus: float | None
    no_result_rate_off_corpus: float | None
    max_off_corpus_top1: float | None
    rerank_timing: RerankTiming | None = None


def wrong_section_rate(
    run: RunRecord, golden: Sequence[GoldenEntry], cutoff: Cutoff
) -> float | None:
    """Mean "right paper, wrong section" share of the top-k over on-corpus questions."""
    values = [
        wrong_section_share(question.hits, labels_of(entry), cutoff.k, cutoff.threshold)
        for entry, question in _joined(run, golden)
        if entry.type is not QuestionType.OFF_CORPUS
    ]
    return _mean(values) if values else None


def rerank_timing(run: RunRecord) -> RerankTiming | None:
    """Median and 90th percentile of the reranker time; ``None`` for a dense run."""
    times = [q.rerank_ms for q in run.questions if q.rerank_ms is not None]
    if not times:
        return None
    median, p90 = np.percentile(times, [50, 90])
    return RerankTiming(n=len(times), median_ms=float(median), p90_ms=float(p90))


def _types_in(golden: Sequence[GoldenEntry]) -> list[QuestionType]:
    present = {entry.type for entry in golden}
    return [qtype for qtype in QuestionType if qtype in present]


def summarize_run(
    run: RunRecord,
    golden: Sequence[GoldenEntry],
    *,
    threshold: float | None = None,
    ks: Sequence[int] = REPORT_KS,
    primary_k: int = PRIMARY_CUTOFF.k,
) -> RunSummary:
    """Everything the run summary report shows, at ``threshold``.

    Without ``threshold`` the cut-off follows the run's score kind
    (``default_threshold``): 0.6 for cosine runs, 0 for reranked runs.
    """
    if threshold is None:
        threshold = default_threshold(run.meta)
    pairs = _joined(run, golden)
    labelled = [
        (entry, question)
        for entry, question in pairs
        if entry.type is not QuestionType.OFF_CORPUS
    ]

    by_k: list[KRow] = []
    for k in ks:
        cutoff = Cutoff(k, threshold)
        scores = {
            metric: list(
                question_scores(run, golden, metric=metric, cutoff=cutoff).values()
            )
            for metric in get_args(Metric)
        }
        by_k.append(
            KRow(
                k=k,
                recall=_mean(scores["recall"]),
                ndcg=_mean(scores["ndcg"]),
                mrr=_mean(scores["mrr"]),
                precision=_mean(scores["precision"]),
                recall_ceiling=_mean(scores["recall_ceiling"]),
                crowding=crowding(run, golden, cutoff),
                table_hit_rate=table_hit_rate(run, golden, cutoff),
                evidence_tokens=evidence_tokens(run, golden, cutoff),
                wrong_section_share=wrong_section_rate(run, golden, cutoff),
            )
        )

    primary = Cutoff(primary_k, threshold)
    per_type: list[TypeRow] = []
    for qtype in _types_in(golden):
        members = [(e, q) for e, q in pairs if e.type is qtype]
        overlaps = [q.ngram_overlap for _, q in members if q.ngram_overlap is not None]
        empty = _mean([float(is_empty(q.hits, threshold)) for _, q in members])
        if qtype is QuestionType.OFF_CORPUS:
            recall = ndcg = mrr = precision = recall_ceiling = None
        else:
            recall = _mean(
                [
                    recall_at_k(q.hits, labels_of(e), primary.k, threshold)
                    for e, q in members
                ]
            )
            ndcg = _mean(
                [
                    ndcg_at_k(q.hits, labels_of(e), primary.k, threshold)
                    for e, q in members
                ]
            )
            mrr = _mean(
                [
                    mrr_at_k(q.hits, labels_of(e), primary.k, threshold)
                    for e, q in members
                ]
            )
            precision = _mean(
                [
                    precision_at_k(q.hits, labels_of(e), primary.k, threshold)
                    for e, q in members
                ]
            )
            recall_ceiling = _mean(
                [
                    recall_ceiling_at_k(q.hits, labels_of(e), primary.k, threshold)
                    for e, q in members
                ]
            )
        per_type.append(
            TypeRow(
                type=qtype,
                n=len(members),
                recall=recall,
                ndcg=ndcg,
                mrr=mrr,
                precision=precision,
                recall_ceiling=recall_ceiling,
                empty_rate=empty,
                mean_overlap=_mean([float(o) for o in overlaps]) if overlaps else None,
                max_overlap=max(overlaps) if overlaps else None,
            )
        )

    return RunSummary(
        meta=run.meta,
        threshold=threshold,
        n_on_corpus=len(labelled),
        n_off_corpus=len(pairs) - len(labelled),
        by_k=tuple(by_k),
        by_type=tuple(per_type),
        empty_rate_on_corpus=empty_rate_on_corpus(run, golden, threshold),
        no_result_rate_off_corpus=no_result_rate_off_corpus(run, golden, threshold),
        max_off_corpus_top1=max_off_corpus_top1(run, golden),
        rerank_timing=rerank_timing(run),
    )


# ------------------------------------------------------------------ threshold curve


def curve_thresholds() -> tuple[float, ...]:
    """0.40 .. 0.80 in 0.02 steps (21 values)."""
    return tuple(round(0.40 + 0.02 * step, 2) for step in range(21))


@dataclass(frozen=True)
class CurvePoint:
    threshold: float
    recall: float
    ndcg: float
    empty_rate_on_corpus: float | None
    no_result_rate_off_corpus: float | None
    max_off_corpus_top1: float | None


@dataclass(frozen=True)
class ThresholdCurve:
    meta: RunMeta
    k: int
    n_on_corpus: int
    n_off_corpus: int
    points: tuple[CurvePoint, ...]


def threshold_curve(
    run: RunRecord,
    golden: Sequence[GoldenEntry],
    *,
    k: int = PRIMARY_CUTOFF.k,
    thresholds: Sequence[float] | None = None,
) -> ThresholdCurve:
    """On-corpus Recall/nDCG at k and the empty / no-result rates for each threshold."""
    steps = tuple(thresholds) if thresholds is not None else curve_thresholds()
    top1 = max_off_corpus_top1(run, golden)
    points: list[CurvePoint] = []
    for threshold in steps:
        cutoff = Cutoff(k, threshold)
        recalls = question_scores(run, golden, metric="recall", cutoff=cutoff)
        ndcgs = question_scores(run, golden, metric="ndcg", cutoff=cutoff)
        points.append(
            CurvePoint(
                threshold=threshold,
                recall=_mean(list(recalls.values())),
                ndcg=_mean(list(ndcgs.values())),
                empty_rate_on_corpus=empty_rate_on_corpus(run, golden, threshold),
                no_result_rate_off_corpus=no_result_rate_off_corpus(
                    run, golden, threshold
                ),
                max_off_corpus_top1=top1,
            )
        )
    n_on = len(on_corpus(golden))
    return ThresholdCurve(
        meta=run.meta,
        k=k,
        n_on_corpus=n_on,
        n_off_corpus=len(golden) - n_on,
        points=tuple(points),
    )


# --------------------------------------------------------------------- gate curve


def gate_thresholds() -> tuple[float, ...]:
    """0 .. 1 in 0.05 steps, plus 0.01, 0.02, 0.98 and 0.99 for the steep ends.

    Reranker scores pass through a sigmoid, so many sit very close to 0 or 1.
    """
    steps = {round(0.05 * step, 2) for step in range(21)}
    steps.update({0.01, 0.02, 0.98, 0.99})
    return tuple(sorted(steps))


@dataclass(frozen=True)
class GatePoint:
    """Refusal at one threshold ``t``.

    A question is refused when its top-1 score is below ``t`` or it has no hit at all.
    """

    threshold: float
    off_refused: int
    off_total: int
    on_lost: int
    on_total: int

    @property
    def off_refused_rate(self) -> float | None:
        return self.off_refused / self.off_total if self.off_total else None

    @property
    def on_lost_rate(self) -> float | None:
        return self.on_lost / self.on_total if self.on_total else None


@dataclass(frozen=True)
class GateCurve:
    meta: RunMeta
    n_on_corpus: int
    n_off_corpus: int
    points: tuple[GatePoint, ...]


def gate_curve(
    run: RunRecord,
    golden: Sequence[GoldenEntry],
    *,
    thresholds: Sequence[float] | None = None,
) -> GateCurve:
    """Refusal on the top-1 score only, swept over ``thresholds`` (default 0 to 1).

    For each ``t``: how many off-corpus questions would be refused (good) and how many
    on-corpus questions would be refused too (lost). Quality metrics are not part of
    this curve; the cut-off is chosen by the owner after seeing it.
    """
    steps = tuple(thresholds) if thresholds is not None else gate_thresholds()
    pairs = _joined(run, golden)
    off = [q for e, q in pairs if e.type is QuestionType.OFF_CORPUS]
    on = [q for e, q in pairs if e.type is not QuestionType.OFF_CORPUS]
    points = tuple(
        GatePoint(
            threshold=t,
            off_refused=sum(is_empty(q.hits, t) for q in off),
            off_total=len(off),
            on_lost=sum(is_empty(q.hits, t) for q in on),
            on_total=len(on),
        )
        for t in steps
    )
    return GateCurve(
        meta=run.meta, n_on_corpus=len(on), n_off_corpus=len(off), points=points
    )


# ------------------------------------------------------------- bootstrap and verdict


def paired_bootstrap_range(
    deltas_by_type: Mapping[str, Sequence[float]],
    *,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> tuple[float, float]:
    """95 % percentile range of ``mean(delta)`` under a stratified bootstrap.

    Each resample draws, within every type, as many deltas as the type has (with
    replacement); the resample mean is the grand total over all questions divided by
    their count. ``numpy.random.default_rng(seed)`` makes the result deterministic.
    """
    strata = {
        name: np.asarray(values, dtype=float)
        for name, values in deltas_by_type.items()
        if len(values) > 0
    }
    if not strata:
        raise ValueError("no questions to resample")
    n_total = sum(len(values) for values in strata.values())
    rng = np.random.default_rng(seed)
    totals = np.zeros(resamples, dtype=float)
    for name in sorted(strata):
        values = strata[name]
        indices = rng.integers(0, len(values), size=(resamples, len(values)))
        totals += values[indices].sum(axis=1)
    lo, hi = np.percentile(totals / n_total, CI_LEVELS)
    return float(lo), float(hi)


@dataclass(frozen=True)
class TypeComparison:
    """Per-type means and win / loss / tie counts of B against A."""

    type: QuestionType
    n: int
    mean_a: float
    mean_b: float
    mean_delta: float
    wins: int
    losses: int
    ties: int

    @property
    def clearly_worse(self) -> bool:
        """Mean delta at or below -0.05 and more losses than wins."""
        return (
            self.mean_delta <= CLEARLY_WORSE_DELTA + TIE_EPSILON
            and self.losses > self.wins
        )


@dataclass(frozen=True)
class Comparison:
    """A paired comparison of B against A on one metric, on-corpus questions only."""

    a: RunMeta
    b: RunMeta
    metric: Metric
    cutoff_a: Cutoff
    cutoff_b: Cutoff
    n: int
    mean_a: float
    mean_b: float
    mean_delta: float
    lo: float
    hi: float
    wins: int
    losses: int
    ties: int
    resamples: int
    seed: int
    per_type: tuple[TypeComparison, ...]
    n_off_corpus: int
    no_result_rate_a: float | None
    no_result_rate_b: float | None


@dataclass(frozen=True)
class Verdict:
    adopt: bool
    lo_above_zero: bool
    clearly_worse: tuple[QuestionType, ...]
    reason: str


def _outcome(delta: float) -> Literal["win", "loss", "tie"]:
    if abs(delta) < TIE_EPSILON:
        return "tie"
    return "win" if delta > 0 else "loss"


def compare_runs(
    run_a: RunRecord,
    run_b: RunRecord,
    golden: Sequence[GoldenEntry],
    *,
    metric: Metric = PRIMARY_METRIC,
    cutoff_a: Cutoff = PRIMARY_CUTOFF,
    cutoff_b: Cutoff | None = None,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> Comparison:
    """Paired comparison of ``run_b`` against ``run_a`` (delta = B - A).

    ``cutoff_b`` defaults to ``cutoff_a``. Passing the same run twice with different
    thresholds gives the threshold comparison.
    """
    cutoff_b = cutoff_b or cutoff_a
    scores_a = question_scores(run_a, golden, metric=metric, cutoff=cutoff_a)
    scores_b = question_scores(run_b, golden, metric=metric, cutoff=cutoff_b)
    entries = on_corpus(golden)
    if not entries:
        raise ValueError("golden set has no on-corpus questions")

    deltas_by_type: dict[str, list[float]] = {}
    per_type_scores: dict[QuestionType, list[tuple[float, float]]] = {}
    for entry in entries:
        a, b = scores_a[entry.id], scores_b[entry.id]
        deltas_by_type.setdefault(entry.type.value, []).append(b - a)
        per_type_scores.setdefault(entry.type, []).append((a, b))

    lo, hi = paired_bootstrap_range(deltas_by_type, resamples=resamples, seed=seed)

    per_type: list[TypeComparison] = []
    for qtype in _types_in(entries):
        pairs = per_type_scores[qtype]
        outcomes = [_outcome(b - a) for a, b in pairs]
        per_type.append(
            TypeComparison(
                type=qtype,
                n=len(pairs),
                mean_a=_mean([a for a, _ in pairs]),
                mean_b=_mean([b for _, b in pairs]),
                mean_delta=_mean([b - a for a, b in pairs]),
                wins=outcomes.count("win"),
                losses=outcomes.count("loss"),
                ties=outcomes.count("tie"),
            )
        )

    all_a = [scores_a[e.id] for e in entries]
    all_b = [scores_b[e.id] for e in entries]
    return Comparison(
        a=run_a.meta,
        b=run_b.meta,
        metric=metric,
        cutoff_a=cutoff_a,
        cutoff_b=cutoff_b,
        n=len(entries),
        mean_a=_mean(all_a),
        mean_b=_mean(all_b),
        mean_delta=_mean([b - a for a, b in zip(all_a, all_b, strict=True)]),
        lo=lo,
        hi=hi,
        wins=sum(t.wins for t in per_type),
        losses=sum(t.losses for t in per_type),
        ties=sum(t.ties for t in per_type),
        resamples=resamples,
        seed=seed,
        per_type=tuple(per_type),
        n_off_corpus=len(golden) - len(entries),
        no_result_rate_a=no_result_rate_off_corpus(run_a, golden, cutoff_a.threshold),
        no_result_rate_b=no_result_rate_off_corpus(run_b, golden, cutoff_b.threshold),
    )


def verdict(comparison: Comparison) -> Verdict:
    """Decision rule (ADR-0023): B replaces A if lo > 0 and no type is clearly worse.

    Meant for the primary metric (nDCG@5 at t = 0.6); the caller picks the comparison.
    """
    lo_above_zero = comparison.lo > 0
    worse = tuple(t.type for t in comparison.per_type if t.clearly_worse)
    adopt = lo_above_zero and not worse
    if adopt:
        reason = "lower end of the range is above 0 and no type is clearly worse"
    else:
        parts: list[str] = []
        if not lo_above_zero:
            parts.append(f"lower end of the range is {comparison.lo:+.4f}, not above 0")
        if worse:
            names = ", ".join(t.value for t in worse)
            parts.append(f"clearly worse on: {names}")
        reason = "; ".join(parts)
    return Verdict(
        adopt=adopt, lo_above_zero=lo_above_zero, clearly_worse=worse, reason=reason
    )


def threshold_verdict(comparison: Comparison) -> Verdict:
    """Threshold rule: the verdict rule, and the off-corpus no-result rate holds.

    ``comparison`` is the same run at two thresholds (A = current, B = lower).
    """
    base = verdict(comparison)
    rate_a, rate_b = comparison.no_result_rate_a, comparison.no_result_rate_b
    dropped = (
        rate_a is not None and rate_b is not None and rate_b < rate_a - TIE_EPSILON
    )
    if not dropped:
        return base
    reason = f"off-corpus no-result rate drops from {rate_a:.2f} to {rate_b:.2f}"
    if not base.adopt:
        reason = f"{base.reason}; {reason}"
    return Verdict(
        adopt=False,
        lo_above_zero=base.lo_above_zero,
        clearly_worse=base.clearly_worse,
        reason=reason,
    )
