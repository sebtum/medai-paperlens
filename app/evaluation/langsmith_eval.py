"""LangSmith experiment for the retrieval evaluation (ADR-0024).

`run_experiment` hands the same retrieval target (`retrieve_ranked`) to LangSmith's
`aevaluate`, together with thin evaluator adapters over the pure metric functions in
`app.evaluation.metrics`. The adapters add no maths of their own: they rebuild the hits
and labels from the example, call the metric function and return the number.

- Experiments run at one fixed k = 5. The threshold follows the run's score kind:
  0.6 for cosine scores (`EXPERIMENT_THRESHOLD`), 0 for reranked and hybrid runs
  (`RERANK_THRESHOLD`; their scores are not cosine). Curves and k-sweeps are local only.
- The experiment metadata holds the run config (R3/R4): config, golden sha256, git hash,
  chunk count, k, threshold.
- Every target or evaluator error is recorded in a `ScoreBook` and re-raised. LangSmith
  logs and swallows such errors by default, so `run_experiment` checks the book after
  the experiment and fails loudly (R5).
- The adapters also record each score they computed. `check_against_local` compares
  those with the scores of the local path for the same run and raises on any
  difference.

`aevaluate` is injected (`aevaluate_fn`), so tests need no LangSmith account or network.
This module imports no `langsmith` code.
"""

from __future__ import annotations

import functools
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from qdrant_client import AsyncQdrantClient

from app.evaluation.compare import (
    Cutoff,
    Metric,
    RunMeta,
    RunRecord,
    default_threshold,
    question_scores,
)
from app.evaluation.golden import GoldenEntry, QuestionType
from app.evaluation.langsmith_sync import DATASET_NAME
from app.evaluation.metrics import (
    RankedHit,
    is_empty,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)
from app.evaluation.target import (
    RankedResult,
    RetrievalConfig,
    RewriteCache,
    retrieve_ranked,
)
from app.llm.base import LlmProvider
from app.retrieval.embedding import EmbeddingProvider
from app.retrieval.rerank import Reranker

EXPERIMENT_K = 5
EXPERIMENT_THRESHOLD = 0.6
SCORE_TOLERANCE = 1e-9

NO_RESULT_KEY = "no_result_off_corpus"
EMPTY_KEY = "empty_on_corpus"

AevaluateFn = Callable[..., Awaitable[Any]]
"""`langsmith.aevaluate`; only the keyword arguments used below are relied on."""


class ExperimentError(RuntimeError):
    """The experiment hit an error in the target or an evaluator."""


class ExperimentMismatchError(ExperimentError):
    """The experiment's scores differ from the local path's scores for the same run."""


def metric_key(metric: str, k: int) -> str:
    """Feedback key of a ranking metric, e.g. `ndcg@5`."""
    return f"{metric}@{k}"


# ---------------------------------------------------------------------------
# Wire format: what the target returns and the evaluators read
# ---------------------------------------------------------------------------


def result_to_outputs(result: RankedResult) -> dict[str, Any]:
    """Target outputs: the query used and the ranked hits as plain lists (JSON-safe)."""
    return {
        "query": result.query,
        "rerank_ms": result.rerank_ms,
        "hits": [
            [h.paper_id, h.section, h.chunk_index, h.chunk_type, h.score]
            for h in result.hits
        ],
    }


def hits_from_outputs(outputs: Mapping[str, Any]) -> list[RankedHit]:
    """Rebuild the ranked hits from target outputs; a malformed hit raises."""
    raw_hits = outputs.get("hits")
    if not isinstance(raw_hits, list):
        raise ExperimentError("target outputs have no 'hits' list")
    hits: list[RankedHit] = []
    for position, raw in enumerate(raw_hits):
        if not isinstance(raw, (list, tuple)) or len(raw) != 5:
            raise ExperimentError(f"hits[{position}] is not a 5-field hit: {raw!r}")
        paper_id, section, chunk_index, chunk_type, score = raw
        hits.append(
            RankedHit(
                paper_id=str(paper_id),
                section=str(section),
                chunk_index=int(chunk_index),
                chunk_type=str(chunk_type),
                score=float(score),
            )
        )
    return hits


def labels_from_reference(
    reference_outputs: Mapping[str, Any],
) -> dict[tuple[str, str], int]:
    """`{(paper_id, section): grade}` from the example's reference outputs."""
    return {
        (str(item["paper_id"]), str(item["section"])): int(item["grade"])
        for item in reference_outputs.get("relevant", [])
    }


def _is_off_corpus(reference_outputs: Mapping[str, Any]) -> bool:
    return reference_outputs.get("type") == QuestionType.OFF_CORPUS.value


# ---------------------------------------------------------------------------
# Score book and evaluator adapters
# ---------------------------------------------------------------------------


@dataclass
class ScoreBook:
    """What the experiment computed: `scores[key][question id]`, and every error."""

    scores: dict[str, dict[str, float]] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)

    def record(self, key: str, question_id: str, score: float) -> None:
        self.scores.setdefault(key, {})[question_id] = score

    def fail(self, where: str, exc: Exception) -> None:
        self.failures.append(f"{where}: {type(exc).__name__}: {exc}")


NOT_APPLICABLE: dict[str, Any] = {"results": []}
"""Evaluator result that writes no feedback (used for the other question kind)."""

_Evaluator = Callable[..., dict[str, Any]]
_RankingFunction = Callable[
    [Sequence[RankedHit], Mapping[tuple[str, str], int], int, float], float
]


def _guarded(book: ScoreBook, name: str, function: _Evaluator) -> _Evaluator:
    """Record an evaluator error in the book before re-raising it.

    `functools.wraps` keeps the parameter names (`inputs`, `outputs`,
    `reference_outputs`), which LangSmith reads to decide what to pass in.
    """

    @functools.wraps(function)
    def wrapper(*args: Any, **kwargs: Any) -> dict[str, Any]:
        try:
            return function(*args, **kwargs)
        except Exception as exc:
            book.fail(f"evaluator {name}", exc)
            raise

    return wrapper


def make_evaluators(
    book: ScoreBook, *, k: int = EXPERIMENT_K, threshold: float = EXPERIMENT_THRESHOLD
) -> list[_Evaluator]:
    """Thin evaluator adapters over the T1 metric functions.

    Ranking metrics (recall, MRR, nDCG) apply to on-corpus questions only. The empty
    on-corpus rate and the off-corpus no-result rate are 0/1 scores per question, so the
    experiment's mean is the rate. A question of the other kind gets no feedback.
    """

    def ranking_adapter(metric: str, function: _RankingFunction) -> _Evaluator:
        key = metric_key(metric, k)

        def evaluator(
            inputs: dict[str, Any],
            outputs: dict[str, Any],
            reference_outputs: dict[str, Any],
        ) -> dict[str, Any]:
            if _is_off_corpus(reference_outputs):
                return NOT_APPLICABLE
            score = function(
                hits_from_outputs(outputs),
                labels_from_reference(reference_outputs),
                k,
                threshold,
            )
            book.record(key, str(inputs["id"]), score)
            return {"key": key, "score": score}

        return _guarded(book, key, evaluator)

    def empty_adapter(key: str, *, off_corpus: bool) -> _Evaluator:
        def evaluator(
            inputs: dict[str, Any],
            outputs: dict[str, Any],
            reference_outputs: dict[str, Any],
        ) -> dict[str, Any]:
            if _is_off_corpus(reference_outputs) is not off_corpus:
                return NOT_APPLICABLE
            score = 1.0 if is_empty(hits_from_outputs(outputs), threshold) else 0.0
            book.record(key, str(inputs["id"]), score)
            return {"key": key, "score": score}

        return _guarded(book, key, evaluator)

    return [
        ranking_adapter("recall", recall_at_k),
        ranking_adapter("mrr", mrr_at_k),
        ranking_adapter("ndcg", ndcg_at_k),
        empty_adapter(EMPTY_KEY, off_corpus=False),
        empty_adapter(NO_RESULT_KEY, off_corpus=True),
    ]


# ---------------------------------------------------------------------------
# Target
# ---------------------------------------------------------------------------


def make_target(
    entries: Sequence[GoldenEntry],
    config: RetrievalConfig,
    book: ScoreBook,
    *,
    embedder: EmbeddingProvider,
    client: AsyncQdrantClient,
    cache: RewriteCache | None,
    llm: LlmProvider | None,
    reranker: Reranker | None = None,
) -> Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]:
    """The async target `aevaluate` calls once per example.

    The example's `inputs` are `{id, question}`. The entry comes from the local golden
    file by id; a question text that differs from the dataset's means the dataset is
    stale, which raises instead of scoring the wrong question.
    """
    by_id = {entry.id: entry for entry in entries}

    async def target(inputs: dict[str, Any]) -> dict[str, Any]:
        question_id = str(inputs.get("id", ""))
        try:
            entry = by_id.get(question_id)
            if entry is None:
                raise ExperimentError(
                    f"dataset example {question_id!r} is not in the local golden file"
                )
            if entry.question != inputs.get("question"):
                raise ExperimentError(
                    f"dataset question for {question_id} differs from the local "
                    "golden file; run scripts.sync_golden"
                )
            result = await retrieve_ranked(
                entry,
                config,
                embedder=embedder,
                client=client,
                cache=cache,
                llm=llm,
                reranker=reranker,
            )
        except Exception as exc:
            book.fail(f"target {question_id or '<no id>'}", exc)
            raise
        return result_to_outputs(result)

    return target


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------


def experiment_metadata(
    meta: RunMeta, *, k: int = EXPERIMENT_K, threshold: float = EXPERIMENT_THRESHOLD
) -> dict[str, Any]:
    """Experiment metadata = the run config (R3) plus golden and code version."""
    return {
        "run_name": meta.name,
        "score_kind": meta.score_kind,
        "config": dict(meta.config),
        "golden_sha256": meta.golden_sha256,
        "git_hash": meta.git_hash,
        "chunk_count": meta.chunk_count,
        "k": k,
        "threshold": threshold,
    }


def experiment_prefix(meta: RunMeta) -> str:
    return f"retrieval-{meta.name}"


@dataclass(frozen=True)
class ExperimentOutcome:
    """Per-question scores the evaluators computed, keyed by feedback key."""

    scores: dict[str, dict[str, float]]


async def run_experiment(
    entries: Sequence[GoldenEntry],
    config: RetrievalConfig,
    meta: RunMeta,
    *,
    embedder: EmbeddingProvider,
    client: AsyncQdrantClient,
    cache: RewriteCache | None,
    llm: LlmProvider | None,
    aevaluate_fn: AevaluateFn,
    dataset_name: str = DATASET_NAME,
    k: int = EXPERIMENT_K,
    threshold: float | None = None,
    reranker: Reranker | None = None,
) -> ExperimentOutcome:
    """Run one LangSmith experiment (`aevaluate`) for `config` at (k, threshold).

    Without `threshold` it follows the run's score kind (`default_threshold`): 0.6 for
    cosine runs, 0 for reranked and hybrid runs.

    Raises `ExperimentError` if the target or any evaluator failed, including errors
    LangSmith would only have logged.
    """
    if threshold is None:
        threshold = default_threshold(meta)
    book = ScoreBook()
    target = make_target(
        entries,
        config,
        book,
        embedder=embedder,
        client=client,
        cache=cache,
        llm=llm,
        reranker=reranker,
    )
    evaluators = make_evaluators(book, k=k, threshold=threshold)
    try:
        await aevaluate_fn(
            target,
            data=dataset_name,
            evaluators=evaluators,
            metadata=experiment_metadata(meta, k=k, threshold=threshold),
            experiment_prefix=experiment_prefix(meta),
            description=(
                f"Retrieval eval run {meta.name} at k={k}, t={threshold} "
                "(golden set LLM-labelled (Claude), not human-verified)"
            ),
            max_concurrency=1,
        )
    except Exception as exc:
        book.fail("aevaluate", exc)
    if book.failures:
        raise ExperimentError(
            f"{len(book.failures)} experiment error(s):\n"
            + "\n".join(f"- {line}" for line in book.failures)
        )
    return ExperimentOutcome(scores=book.scores)


# ---------------------------------------------------------------------------
# Local cross-check
# ---------------------------------------------------------------------------


def local_experiment_scores(
    run: RunRecord,
    golden: Sequence[GoldenEntry],
    *,
    k: int = EXPERIMENT_K,
    threshold: float | None = None,
) -> dict[str, dict[str, float]]:
    """The scores the evaluators should produce, computed by the local path.

    Without `threshold` it follows the run's score kind, like `run_experiment`.
    """
    if threshold is None:
        threshold = default_threshold(run.meta)
    cutoff = Cutoff(k, threshold)
    expected: dict[str, dict[str, float]] = {}
    metrics: tuple[Metric, ...] = ("recall", "mrr", "ndcg")
    for metric in metrics:
        expected[metric_key(metric, k)] = question_scores(
            run, golden, metric=metric, cutoff=cutoff
        )
    types = {entry.id: entry.type for entry in golden}
    empty: dict[str, float] = {}
    no_result: dict[str, float] = {}
    for question in run.questions:
        flag = 1.0 if is_empty(question.hits, threshold) else 0.0
        if types[question.id] is QuestionType.OFF_CORPUS:
            no_result[question.id] = flag
        else:
            empty[question.id] = flag
    expected[EMPTY_KEY] = empty
    expected[NO_RESULT_KEY] = no_result
    return expected


def check_against_local(
    outcome: ExperimentOutcome, expected: Mapping[str, Mapping[str, float]]
) -> None:
    """Raise `ExperimentMismatchError` listing each score that differs or is missing."""
    problems: list[str] = []
    for key, per_question in expected.items():
        got = outcome.scores.get(key, {})
        for question_id, value in per_question.items():
            if question_id not in got:
                problems.append(f"{key} {question_id}: no experiment score")
            elif abs(got[question_id] - value) > SCORE_TOLERANCE:
                problems.append(
                    f"{key} {question_id}: experiment {got[question_id]:.6f}, "
                    f"local {value:.6f}"
                )
        for question_id in got.keys() - per_question.keys():
            problems.append(f"{key} {question_id}: experiment score without local one")
    if problems:
        raise ExperimentMismatchError(
            f"{len(problems)} score(s) differ between experiment and local run:\n"
            + "\n".join(f"- {line}" for line in problems)
        )
