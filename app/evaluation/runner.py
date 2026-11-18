"""Retrieval evaluation runner: named configs, `run`, `compare`, `curve` (ADR-0024).

The logic behind `scripts/eval_retrieval.py`; the script only parses arguments, loads
`.env` and prints. Every command returns the lines to print instead of printing, and
anything that touches the outside world (Qdrant, the embedder, the rewrite LLM,
LangSmith, git) comes in through `Services`, so tests need none of them.

- `run_command`: pre-flight, retrieve, write `runs/<name>.json` and a report; with
  `langsmith` also an experiment whose scores must equal the local ones. A config with a
  reranker loads it once, times the load, makes one warm-up call, then times one
  reranker call per question (Phase 10).
- `compare_command`, `curve_command`: reports from stored runs only. Their default
  cut-off follows each run's score kind (0.6 cosine, 0 reranked).
- Failures raise (`EVAL_ERRORS` are the expected ones); nothing here swallows an error.
- No paper text is written anywhere: chunk text is read into memory for the token
  estimate and the n-gram probe only.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable, Iterable, Sequence
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qdrant_client import AsyncQdrantClient

from app.evaluation.compare import (
    PRIMARY_CUTOFF,
    Cutoff,
    Metric,
    QuestionRun,
    RunFormatError,
    RunMeta,
    RunRecord,
    compare_runs,
    default_threshold,
    gate_curve,
    load_run,
    save_run,
    summarize_run,
    threshold_curve,
    threshold_verdict,
    verdict,
)
from app.evaluation.golden import (
    GoldenEntry,
    GoldenFormatError,
    golden_sha256,
    load_golden,
    load_manifests,
    validate_golden,
)
from app.evaluation.langsmith_eval import (
    EXPERIMENT_K,
    AevaluateFn,
    ExperimentError,
    check_against_local,
    local_experiment_scores,
    run_experiment,
)
from app.evaluation.langsmith_sync import DATASET_NAME, GitError, git_short_hash
from app.evaluation.report import (
    render_comparison,
    render_gate_curve,
    render_run_summary,
    render_threshold_curve,
)
from app.evaluation.target import (
    DEFAULT_DEPTH,
    DEFAULT_PREFILTER_THRESHOLD,
    DEFAULT_REWRITES_PATH,
    PreflightError,
    RerankQuery,
    RetrievalConfig,
    RewriteCache,
    RewriteCacheMissError,
    RewriteFallbackError,
    RewriteKey,
    fetch_chunk_payloads,
    longest_shared_ngram,
    parse_source_chunk,
    preflight,
    retrieve_ranked,
    rewrite_prompt_sha256,
)
from app.llm.base import LlmProvider
from app.retrieval.embedding import (
    DEFAULT_EMBEDDING_MODEL,
    EmbeddingProvider,
    SentenceTransformerProvider,
)
from app.retrieval.ingestion import COLLECTION_NAME
from app.retrieval.rerank import Reranker
from app.workflow.generate import _estimated_tokens

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = REPO_ROOT / "data" / "eval"
DEFAULT_GOLDEN = EVAL_DIR / "golden.jsonl"
DEFAULT_PAPERS_DIR = REPO_ROOT / "data" / "papers"
DEFAULT_RUNS_DIR = EVAL_DIR / "runs"
DEFAULT_REPORTS_DIR = EVAL_DIR / "reports"
DEFAULT_REWRITES = REPO_ROOT / DEFAULT_REWRITES_PATH

# Frozen copy of the experiment's prefix (recorded in the run files). The app's own
# default lives in app.retrieval.embedding (ADR-0025); a test keeps the two equal.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
BGE_BASE_MODEL = "BAAI/bge-base-en-v1.5"
BGE_BASE_COLLECTION = "eval_bge-base-en-v1.5"

RERANKER_MODELS: dict[str, str] = {
    "minilm": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "bge": "BAAI/bge-reranker-base",
    "medcpt": "ncbi/MedCPT-Cross-Encoder",
}
"""The three candidate rerankers (Phase 10); `--reranker` also takes a short alias."""

GitRunner = Callable[[Sequence[str]], str]


class RunError(RuntimeError):
    """The eval cannot continue; the message is the whole explanation."""


EVAL_ERRORS: tuple[type[Exception], ...] = (
    RunError,
    PreflightError,
    RewriteCacheMissError,
    RewriteFallbackError,
    ExperimentError,
    GoldenFormatError,
    RunFormatError,
)
"""Expected failures: the CLI logs the message and exits 1."""

# ---------------------------------------------------------------------------
# Named configs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConfigSpec:
    """A named run configuration (the plan's four runs, then the reranker runs)."""

    embedder: str
    collection: str
    prefix: str = ""
    rewrite: bool = True
    reranker: str | None = None


NAMED_CONFIGS: dict[str, ConfigSpec] = {
    "baseline": ConfigSpec(DEFAULT_EMBEDDING_MODEL, COLLECTION_NAME),
    "no_rewrite": ConfigSpec(DEFAULT_EMBEDDING_MODEL, COLLECTION_NAME, rewrite=False),
    "prefix": ConfigSpec(
        DEFAULT_EMBEDDING_MODEL, COLLECTION_NAME, prefix=BGE_QUERY_PREFIX
    ),
    "bge_base": ConfigSpec(BGE_BASE_MODEL, BGE_BASE_COLLECTION),
    # The reranker runs are built on `prefix`, the default dense setup (ADR-0025).
    "rerank_minilm": ConfigSpec(
        DEFAULT_EMBEDDING_MODEL,
        COLLECTION_NAME,
        prefix=BGE_QUERY_PREFIX,
        reranker=RERANKER_MODELS["minilm"],
    ),
    "rerank_bge": ConfigSpec(
        DEFAULT_EMBEDDING_MODEL,
        COLLECTION_NAME,
        prefix=BGE_QUERY_PREFIX,
        reranker=RERANKER_MODELS["bge"],
    ),
    "rerank_medcpt": ConfigSpec(
        DEFAULT_EMBEDDING_MODEL,
        COLLECTION_NAME,
        prefix=BGE_QUERY_PREFIX,
        reranker=RERANKER_MODELS["medcpt"],
    ),
}


def default_rewrite_model() -> str:
    """Name of the model that makes rewrites, from the same env the app reads."""
    provider = os.environ.get("LLM_PROVIDER", "ollama").strip().lower()
    if provider == "gemini":
        return os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    return os.environ.get("OLLAMA_MODEL", "qwen3:0.6b")


# ---------------------------------------------------------------------------
# Services (injectable, so tests need no Qdrant, model, Ollama or LangSmith)
# ---------------------------------------------------------------------------


def _default_client() -> AsyncQdrantClient:
    from app.retrieval.client import get_async_client

    return get_async_client()


def _default_embedder(model_name: str) -> EmbeddingProvider:
    from sentence_transformers import SentenceTransformer

    # Always an empty provider prefix: the target prepends `config.prefix` itself, so
    # each named config sets its prefix exactly once. The app's default prefix
    # (ADR-0025) must not leak into `baseline`, `no_rewrite` or `bge_base`, nor double
    # up in `prefix`.
    return SentenceTransformerProvider(SentenceTransformer(model_name), query_prefix="")


def _default_reranker(model_name: str) -> Reranker:
    from app.retrieval import rerank

    # Through the module, so the test fixture that blocks real model loads applies.
    return rerank.CrossEncoderReranker(
        rerank.CrossEncoder(model_name, max_length=rerank.RERANKER_MAX_LENGTH),
        model_name=model_name,
    )


def _default_reranker_disk_bytes(model_name: str) -> int | None:
    """Size of the model in the local Hugging Face cache in bytes, `None` if absent.

    This is the whole cached repository, every stored revision and file together.
    """
    try:
        from huggingface_hub import scan_cache_dir

        for repo in scan_cache_dir().repos:
            if repo.repo_type == "model" and repo.repo_id == model_name:
                return int(repo.size_on_disk)
    except Exception:
        logger.warning("Cannot read the Hugging Face cache for %s", model_name)
    return None


def _default_llm() -> Any:
    from app.llm.factory import make_llm_provider

    return make_llm_provider()


def _default_aevaluate() -> AevaluateFn:
    from langsmith import aevaluate

    return aevaluate


@dataclass
class Services:
    """Everything that touches the outside world, replaceable in tests.

    `llm_factory` returns an async context manager with a `model` attribute (the real
    `OllamaClient` / `GeminiClient`); it is only called when a rewrite is missing.
    """

    git: GitRunner
    client_factory: Callable[[], AsyncQdrantClient] = _default_client
    embedder_factory: Callable[[str], EmbeddingProvider] = _default_embedder
    llm_factory: Callable[[], Any] = _default_llm
    aevaluate_factory: Callable[[], AevaluateFn] = _default_aevaluate
    reranker_factory: Callable[[str], Reranker] = _default_reranker
    reranker_disk_bytes: Callable[[str], int | None] = _default_reranker_disk_bytes


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def git_version(git: GitRunner) -> str:
    """Short hash of HEAD, `-dirty` when tracked files have uncommitted changes."""
    try:
        short = git_short_hash(git)
        dirty = git(["status", "--porcelain", "--untracked-files=no"]).strip()
    except GitError as exc:
        logger.warning("Cannot read the git version: %s", exc)
        return "unknown"
    return f"{short}-dirty" if dirty else short


def _load_entries(
    golden_path: Path, papers_dir: Path, *, validate: bool
) -> tuple[list[GoldenEntry], str]:
    entries = load_golden(golden_path)
    if not entries:
        raise RunError(f"{golden_path} contains no entries")
    if validate:
        violations = validate_golden(entries, load_manifests(papers_dir))
        if violations:
            raise RunError(
                f"golden set has {len(violations)} violation(s):\n"
                + "\n".join(f"- {v}" for v in violations)
                + "\n(--skip-validate runs a work-in-progress set anyway)"
            )
    return entries, golden_sha256(golden_path)


def _resolve_run_path(value: str, runs_dir: Path) -> Path:
    """A run is given as a file path or as a name under the runs directory."""
    direct = Path(value)
    if direct.is_file():
        return direct
    named = runs_dir / (value if value.endswith(".json") else f"{value}.json")
    if named.is_file():
        return named
    raise RunError(f"run not found: {value!r} (looked at {direct} and {named})")


def load_runs(
    values: Sequence[str], runs_dir: Path, golden_sha: str, golden_path: Path
) -> list[RunRecord]:
    """Load runs and refuse any that was made on a different golden file."""
    runs: list[RunRecord] = []
    for value in values:
        path = _resolve_run_path(value, runs_dir)
        run = load_run(path)
        if run.meta.golden_sha256 != golden_sha:
            raise RunError(
                f"run {run.meta.name!r} ({path}) was made on golden sha256 "
                f"{run.meta.golden_sha256[:12]}, but {golden_path} is "
                f"{golden_sha[:12]}; re-run it on the current golden set"
            )
        runs.append(run)
    return runs


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _rewrite_missing(
    entries: Iterable[GoldenEntry], cache: RewriteCache, rewrite_model: str
) -> list[str]:
    return [
        entry.id
        for entry in entries
        if cache.get(RewriteKey.for_entry(entry, rewrite_model)) is None
    ]


# ---------------------------------------------------------------------------

# run
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunOptions:
    name: str
    config_name: str
    config: RetrievalConfig
    golden_path: Path
    papers_dir: Path
    rewrites_path: Path
    runs_dir: Path
    reports_dir: Path
    skip_validate: bool = False
    preflight_only: bool = False
    no_llm: bool = False
    langsmith: bool = False
    dataset_name: str | None = None


def _config_dict(
    options: RunOptions, reranker_info: dict[str, Any] | None = None
) -> dict[str, Any]:
    """The run header's config. `reranker_info` adds load time and disk size."""
    config = options.config
    data: dict[str, Any] = {
        "config_name": options.config_name,
        "embedder": config.embedder,
        "collection": config.collection,
        "prefix": config.prefix,
        "rewrite": config.rewrite,
        "rewrite_model": config.rewrite_model,
        "rewrite_prompt_sha256": rewrite_prompt_sha256() if config.rewrite else None,
        "depth": config.depth,
        "retrieval_threshold": 0.0,
        "golden_validated": not options.skip_validate,
        "score_kind": config.score_kind,
        "reranker": config.reranker,
    }
    if config.reranker is not None:
        data["rerank_query"] = config.rerank_query
        data["prefilter_threshold"] = config.prefilter_threshold
        data.update(reranker_info or {})
    return data


async def fetch_chunk_texts(
    client: AsyncQdrantClient, collection: str, refs: Iterable[tuple[str, int]]
) -> dict[tuple[str, int], str]:
    """Chunk text by `(paper_id, chunk_index)` for the chunks that have any.

    The text only lives in memory: it feeds the token estimate and the n-gram probe.
    The payload read (title and text) is shared with the reranker's passages.
    """
    payloads = await fetch_chunk_payloads(client, collection, refs)
    return {ref: chunk.text for ref, chunk in payloads.items()}


def _source_refs(entry: GoldenEntry) -> list[tuple[str, int]]:
    return [
        parsed
        for ref in entry.source_chunks
        if (parsed := parse_source_chunk(ref)) is not None
    ]


async def _collect_question(
    entry: GoldenEntry,
    options: RunOptions,
    *,
    embedder: EmbeddingProvider,
    client: AsyncQdrantClient,
    cache: RewriteCache | None,
    llm: LlmProvider | None,
    reranker: Reranker | None = None,
) -> QuestionRun:
    result = await retrieve_ranked(
        entry,
        options.config,
        embedder=embedder,
        client=client,
        cache=cache,
        llm=llm,
        reranker=reranker,
    )
    sources = _source_refs(entry)
    texts = await fetch_chunk_texts(
        client,
        options.config.collection,
        [*((h.paper_id, h.chunk_index) for h in result.hits), *sources],
    )

    tokens: list[int] = []
    for hit in result.hits:
        text = texts.get((hit.paper_id, hit.chunk_index))
        if text is None:
            raise RunError(
                f"{entry.id}: hit {hit.paper_id}:{hit.chunk_index} has no text in "
                f"collection {options.config.collection!r}"
            )
        tokens.append(_estimated_tokens(text))

    overlap: int | None = None
    if sources:
        source_texts = [texts[ref] for ref in sources if ref in texts]
        if len(source_texts) != len(sources):
            raise RunError(f"{entry.id}: a source chunk has no text in the collection")
        overlap = longest_shared_ngram(entry.question, source_texts)

    return QuestionRun(
        id=entry.id,
        type=entry.type,
        hits=tuple(result.hits),
        ngram_overlap=overlap,
        chunk_tokens=tuple(tokens),
        query=result.query,
        rerank_ms=result.rerank_ms,
    )


WARMUP_TEXT = "warm-up"


async def _load_reranker(
    model_name: str, services: Services
) -> tuple[Reranker, dict[str, Any]]:
    """Load the reranker, time the load, make one warm-up call, measure the disk size.

    The warm-up call keeps the first-call cost (lazy setup, thread start) out of the
    per-question times. Returns the reranker and the run-header fields about it.
    """
    started = time.perf_counter()
    reranker = services.reranker_factory(model_name)
    load_s = time.perf_counter() - started
    await reranker.score(WARMUP_TEXT, [WARMUP_TEXT])
    info: dict[str, Any] = {
        "reranker_load_s": round(load_s, 2),
        "reranker_disk_bytes": services.reranker_disk_bytes(model_name),
    }
    logger.info("Reranker %s loaded in %.1f s and warmed up", model_name, load_s)
    return reranker, info


async def run_command(options: RunOptions, services: Services) -> list[str]:
    """Pre-flight, retrieve, write the run file and report; optional experiment."""
    config = options.config
    if options.langsmith and not os.getenv("LANGSMITH_API_KEY"):
        raise RunError(
            "LANGSMITH_API_KEY is not set. Set it in the environment or .env "
            "(see .env.example) to use --langsmith."
        )

    entries, sha256 = _load_entries(
        options.golden_path, options.papers_dir, validate=not options.skip_validate
    )
    client = services.client_factory()

    index = await preflight(entries, client, config)
    logger.info(
        "Pre-flight passed: %d chunks in %r, %d golden questions",
        index.chunk_count,
        config.collection,
        len(entries),
    )
    if options.preflight_only:
        return [
            f"Pre-flight passed for {options.config_name}: {len(entries)} questions, "
            f"{index.chunk_count} chunks in {config.collection!r}."
        ]

    cache = RewriteCache(options.rewrites_path) if config.rewrite else None
    async with AsyncExitStack() as stack:
        llm: LlmProvider | None = None
        if cache is not None:
            missing = _rewrite_missing(entries, cache, config.rewrite_model)
            if missing and options.no_llm:
                raise RewriteCacheMissError(
                    f"{len(missing)} rewrite(s) are not cached for model "
                    f"{config.rewrite_model!r} and --no-llm is set: "
                    + ", ".join(missing)
                )
            if missing:
                logger.info(
                    "%d of %d rewrites are not cached; using the LLM for them",
                    len(missing),
                    len(entries),
                )
                llm = await stack.enter_async_context(services.llm_factory())
                served = getattr(llm, "model", None)
                if served != config.rewrite_model:
                    raise RunError(
                        f"the LLM serves model {served!r}, but the run's rewrite model "
                        f"is {config.rewrite_model!r}; they must match because the "
                        "rewrite cache is keyed by model"
                    )
            else:
                logger.info("All %d rewrites are cached; no LLM needed", len(entries))

        embedder = services.embedder_factory(config.embedder)
        reranker: Reranker | None = None
        reranker_info: dict[str, Any] | None = None
        if config.reranker is not None:
            reranker, reranker_info = await _load_reranker(config.reranker, services)
        questions: list[QuestionRun] = []
        for position, entry in enumerate(entries, start=1):
            questions.append(
                await _collect_question(
                    entry,
                    options,
                    embedder=embedder,
                    client=client,
                    cache=cache,
                    llm=llm,
                    reranker=reranker,
                )
            )
            if position % 10 == 0 or position == len(entries):
                logger.info("Retrieved %d / %d questions", position, len(entries))

        meta = RunMeta(
            name=options.name,
            config=_config_dict(options, reranker_info),
            golden_sha256=sha256,
            git_hash=git_version(services.git),
            chunk_count=index.chunk_count,
        )
        record = RunRecord(meta=meta, questions=tuple(questions))
        run_path = options.runs_dir / f"{options.name}.json"
        save_run(record, run_path)
        report_path = options.reports_dir / f"run-{options.name}.md"
        write_text(report_path, render_run_summary(summarize_run(record, entries)))
        lines = [f"Run written: {run_path}", f"Report written: {report_path}"]
        logger.info("Run written: %s; report: %s", run_path, report_path)

        if options.langsmith:
            threshold = default_threshold(meta)
            outcome = await run_experiment(
                entries,
                config,
                meta,
                embedder=embedder,
                client=client,
                cache=cache,
                llm=llm,
                aevaluate_fn=services.aevaluate_factory(),
                dataset_name=options.dataset_name or DATASET_NAME,
                threshold=threshold,
                reranker=reranker,
            )
            check_against_local(
                outcome, local_experiment_scores(record, entries, threshold=threshold)
            )
            lines.append(
                f"LangSmith experiment finished at k={EXPERIMENT_K}, "
                f"t={threshold} ({meta.score_kind} scores); its scores equal the "
                "local run."
            )
    return lines


def make_run_options(
    config_name: str,
    *,
    name: str | None = None,
    collection: str | None = None,
    embedder: str | None = None,
    rewrite_model: str | None = None,
    golden_path: Path = DEFAULT_GOLDEN,
    papers_dir: Path = DEFAULT_PAPERS_DIR,
    rewrites_path: Path = DEFAULT_REWRITES,
    runs_dir: Path = DEFAULT_RUNS_DIR,
    reports_dir: Path = DEFAULT_REPORTS_DIR,
    skip_validate: bool = False,
    preflight_only: bool = False,
    no_llm: bool = False,
    langsmith: bool = False,
    dataset_name: str | None = None,
    depth: int | None = None,
    rerank_query: RerankQuery | None = None,
    reranker: str | None = None,
) -> RunOptions:
    """Resolve a named config plus overrides into `RunOptions`.

    `depth`, `rerank_query` and `reranker` change what the named config measures, so
    they need an explicit `name`: without one the run would overwrite the named
    config's own run file. `reranker` is a model name or an alias of `RERANKER_MODELS`.
    """
    spec = NAMED_CONFIGS[config_name]
    changed = [
        flag
        for flag, value in (
            ("--depth", depth),
            ("--rerank-query", rerank_query),
            ("--reranker", reranker),
        )
        if value is not None
    ]
    if changed and not name:
        raise RunError(
            f"{', '.join(changed)} change(s) the config, so --name is required "
            f"(otherwise the run would overwrite {config_name}.json)"
        )
    model = RERANKER_MODELS.get(reranker, reranker) if reranker else spec.reranker
    if rerank_query is not None and model is None:
        raise RunError(
            "--rerank-query needs a reranker (--config rerank_* or --reranker)"
        )
    config = RetrievalConfig(
        embedder=embedder or spec.embedder,
        collection=collection or spec.collection,
        prefix=spec.prefix,
        rewrite=spec.rewrite,
        rewrite_model=(
            (rewrite_model or default_rewrite_model()) if spec.rewrite else ""
        ),
        depth=depth if depth is not None else DEFAULT_DEPTH,
        reranker=model,
        rerank_query=rerank_query or "rewritten",
        prefilter_threshold=DEFAULT_PREFILTER_THRESHOLD,
    )
    return RunOptions(
        name=name or config_name,
        config_name=config_name,
        config=config,
        golden_path=golden_path,
        papers_dir=papers_dir,
        rewrites_path=rewrites_path,
        runs_dir=runs_dir,
        reports_dir=reports_dir,
        skip_validate=skip_validate,
        preflight_only=preflight_only,
        no_llm=no_llm,
        langsmith=langsmith,
        dataset_name=dataset_name,
    )


# ---------------------------------------------------------------------------
# compare / curve
# ---------------------------------------------------------------------------


def compare_command(
    run_a: str,
    run_b: str,
    *,
    golden_path: Path,
    runs_dir: Path,
    reports_dir: Path,
    metric: Metric = "ndcg",
    k: int = PRIMARY_CUTOFF.k,
    threshold: float | None = None,
    threshold_b: float | None = None,
    threshold_rule: bool = False,
    resamples: int = 1000,
    out: Path | None = None,
) -> list[str]:
    """Compare run B against run A; write the report, return the verdict lines.

    Without `threshold`, each run is cut at the default for its score kind: 0.6 for
    cosine runs, 0 for reranked runs (the cosine pre-filter already ran). `threshold`
    sets both sides, `threshold_b` overrides B.
    """
    entries = load_golden(golden_path)
    sha256 = golden_sha256(golden_path)
    record_a, record_b = load_runs([run_a, run_b], runs_dir, sha256, golden_path)
    cut_a = threshold if threshold is not None else default_threshold(record_a.meta)
    if threshold_b is not None:
        cut_b = threshold_b
    elif threshold is not None:
        cut_b = threshold
    else:
        cut_b = default_threshold(record_b.meta)
    comparison = compare_runs(
        record_a,
        record_b,
        entries,
        metric=metric,
        cutoff_a=Cutoff(k, cut_a),
        cutoff_b=Cutoff(k, cut_b),
        resamples=resamples,
    )
    use_threshold_rule = threshold_rule or threshold_b is not None
    text = render_comparison(comparison, threshold_rule=use_threshold_rule)
    suffix = f"-t{threshold_b:.2f}" if threshold_b is not None else ""
    target = out or (
        reports_dir / f"compare-{record_b.meta.name}-vs-{record_a.meta.name}{suffix}.md"
    )
    write_text(target, text)
    result = (
        threshold_verdict(comparison) if use_threshold_rule else verdict(comparison)
    )
    return [
        f"Report written: {target}",
        f"{'B replaces A' if result.adopt else 'A stays'}: {result.reason} "
        f"(mean delta {comparison.mean_delta:+.3f}, "
        f"range [{comparison.lo:+.4f}, {comparison.hi:+.4f}])",
    ]


def curve_command(
    run: str,
    *,
    golden_path: Path,
    runs_dir: Path,
    reports_dir: Path,
    k: int = PRIMARY_CUTOFF.k,
    out: Path | None = None,
    gate: bool = False,
) -> list[str]:
    """Threshold curve of one run; write the report, return the lines to print.

    With `gate` it is the refusal curve instead: the top-1 score swept from 0 to 1,
    with the off-corpus questions refused and the on-corpus questions lost per `t`.
    """
    entries = load_golden(golden_path)
    sha256 = golden_sha256(golden_path)
    (record,) = load_runs([run], runs_dir, sha256, golden_path)
    if gate:
        gate_target = out or reports_dir / f"gate-{record.meta.name}.md"
        write_text(gate_target, render_gate_curve(gate_curve(record, entries)))
        return [f"Report written: {gate_target}"]
    target = out or reports_dir / f"curve-{record.meta.name}.md"
    write_text(target, render_threshold_curve(threshold_curve(record, entries, k=k)))
    return [f"Report written: {target}"]
