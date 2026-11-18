"""Retrieval target for the evaluation: rewrite cache, ranked retrieval, pre-flight.

Plan decisions (Phase 9): U2 (pre-flight against Qdrant), U4 (one collection per
embedder), U6 (rewrites are cached and committed).

- `retrieve_ranked` runs one question through one `RetrievalConfig`: optional query
  rewrite (served from the cache when possible), the bge-style query prefix, one
  top-`depth` search at threshold 0. Cut-offs (k, threshold) are applied afterwards with
  `app.evaluation.metrics.cut`.
- `RewriteCache` stores rewrites in `data/eval/rewrites.jsonl`. A miss calls the real
  `make_rewrite_node`. That node silently falls back to the original question on an LLM
  error, which would make rewrite-on equal rewrite-off, so any fallback raises
  `RewriteFallbackError` and the run fails.
- `preflight` scrolls the whole target collection once and reports every labelling
  problem in a single `PreflightError`.
- `longest_shared_ngram` / `source_overlap` measure the consecutive words a question
  shares with its source chunks (the U3 paraphrase probe).

Nothing here writes paper text to disk; the rewrite cache holds hashes and our own
rewritten questions only.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections import Counter
from collections.abc import AsyncIterator, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, cast

from qdrant_client import AsyncQdrantClient

from app.evaluation.golden import GoldenEntry, QuestionType
from app.evaluation.metrics import TABLE_CHUNK_TYPE, RankedHit
from app.llm.base import LlmProvider, T
from app.retrieval.embedding import EmbeddingProvider
from app.retrieval.ingestion import point_id
from app.retrieval.rerank import Reranker, rerank_passage
from app.retrieval.search import search_hits
from app.workflow.rewrite import _PROMPT as REWRITE_PROMPT
from app.workflow.rewrite import (
    REWRITE_CLEANUP_VERSION,
    clean_rewrite,
    make_rewrite_node,
)
from app.workflow.state import WorkflowState

DEFAULT_DEPTH = 20
"""Chunks kept per question; any k up to this is derived afterwards."""

NO_SCORE_THRESHOLD = 0.0
"""Retrieval runs at threshold 0; the real threshold is applied after the fact."""

DEFAULT_PREFILTER_THRESHOLD = 0.6
"""Cosine pre-filter before the reranker; the same loose cut-off the app keeps."""

RerankQuery = Literal["rewritten", "original"]
ScoreKind = Literal["cosine", "rerank"]

REWRITE_TEMPERATURE = 0
REWRITE_SEED = 0

DEFAULT_REWRITES_PATH = Path("data/eval/rewrites.jsonl")

PREFLIGHT_PAYLOAD_FIELDS = (
    "paper_id",
    "section",
    "chunk_index",
    "chunk_type",
    "embedding_model",
)
_SCROLL_PAGE_SIZE = 256
_MISSING_MODEL = "<missing>"

_SOURCE_CHUNK_RE = re.compile(r"^(?P<paper>[A-Za-z0-9_\-]+):(?P<index>\d+)$")
_WORD_RE = re.compile(r"[a-z0-9]+")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RetrievalConfig:
    """One evaluation run: (embedder, prefix, rewrite) over one collection.

    `embedder` is the embedding model name that must match the collection's stored
    `embedding_model`. `rewrite_model` names the LLM that makes the rewrites and is part
    of the rewrite-cache key; it is required when `rewrite` is on.

    `reranker` names a reranker model (Phase 10); `None` is the dense-only run. With a
    reranker the dense top-`depth` hits are first cut at `prefilter_threshold` (cosine),
    then re-scored against `rerank_query` ("rewritten": the query the embedder saw;
    "original": the golden question) and sorted by the reranker score.
    """

    embedder: str
    collection: str
    prefix: str = ""
    rewrite: bool = True
    rewrite_model: str = ""
    depth: int = DEFAULT_DEPTH
    reranker: str | None = None
    rerank_query: RerankQuery = "rewritten"
    prefilter_threshold: float = DEFAULT_PREFILTER_THRESHOLD

    def __post_init__(self) -> None:
        if self.depth < 1:
            raise ValueError(f"depth must be >= 1, got {self.depth}")
        if self.rewrite and not self.rewrite_model:
            raise ValueError("rewrite_model is required when rewrite is on")
        if self.reranker is not None and not self.reranker.strip():
            raise ValueError("reranker must be a model name or None")
        if self.rerank_query not in ("rewritten", "original"):
            raise ValueError(
                f"rerank_query must be 'rewritten' or 'original', got "
                f"{self.rerank_query!r}"
            )
        if not 0.0 <= self.prefilter_threshold <= 1.0:
            raise ValueError(
                f"prefilter_threshold must be in 0..1, got {self.prefilter_threshold}"
            )

    @property
    def score_kind(self) -> ScoreKind:
        """What the hit scores mean: cosine similarity, or reranker output (0..1)."""
        return "rerank" if self.reranker is not None else "cosine"


@dataclass(frozen=True)
class RankedResult:
    """What one question retrieved: the query used (before the prefix) and its hits.

    `rerank_ms` is the time the reranker took for this question in milliseconds; `None`
    when no reranker ran (dense-only run, or nothing left after the pre-filter).
    """

    query: str
    hits: list[RankedHit]
    rerank_ms: float | None = None


# ---------------------------------------------------------------------------
# Rewrite cache
# ---------------------------------------------------------------------------


class RewriteCacheMissError(RuntimeError):
    """A rewrite is not cached and no LLM was provided to make it."""


class RewriteFallbackError(RuntimeError):
    """The rewriter fell back to the original question (LLM error or empty output)."""


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def rewrite_prompt_sha256() -> str:
    """Hash of the real rewrite prompt plus the cleanup version.

    Editing the prompt or bumping `REWRITE_CLEANUP_VERSION` invalidates the cache.
    """
    return _sha256(f"{REWRITE_PROMPT}\n{REWRITE_CLEANUP_VERSION}")


@dataclass(frozen=True)
class RewriteKey:
    """Cache key: question id, sha256(question), rewrite model, rewrite prompt hash.

    The prompt hash covers the prompt and `REWRITE_CLEANUP_VERSION`.
    """

    question_id: str
    question_sha256: str
    model: str
    prompt_sha256: str

    @classmethod
    def for_entry(cls, entry: GoldenEntry, rewrite_model: str) -> RewriteKey:
        return cls(
            question_id=entry.id,
            question_sha256=_sha256(entry.question),
            model=rewrite_model,
            prompt_sha256=rewrite_prompt_sha256(),
        )


class RewriteCache:
    """JSONL file of rewrites keyed by `RewriteKey`; a missing file is empty."""

    def __init__(self, path: Path | str = DEFAULT_REWRITES_PATH) -> None:
        self.path = Path(path)
        self._rewrites: dict[RewriteKey, str] = {}
        if self.path.exists():
            self._load()

    def _load(self) -> None:
        with self.path.open(encoding="utf-8") as handle:
            for line_no, raw in enumerate(handle, start=1):
                line = raw.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    key = RewriteKey(
                        question_id=row["id"],
                        question_sha256=row["question_sha256"],
                        model=row["model"],
                        prompt_sha256=row["prompt_sha256"],
                    )
                    self._rewrites[key] = str(row["rewrite"])
                except (json.JSONDecodeError, KeyError, TypeError) as exc:
                    raise ValueError(
                        f"{self.path}:{line_no}: bad rewrite cache line: {exc}"
                    ) from exc

    def __len__(self) -> int:
        return len(self._rewrites)

    def get(self, key: RewriteKey) -> str | None:
        return self._rewrites.get(key)

    def put(self, key: RewriteKey, rewrite: str) -> None:
        """Remember a rewrite and append it to the file."""
        self._rewrites[key] = rewrite
        self.path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "id": key.question_id,
            "question_sha256": key.question_sha256,
            "model": key.model,
            "prompt_sha256": key.prompt_sha256,
            "rewrite": rewrite,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


class _RecordingRewriteLlm:
    """Wraps the LLM for the rewrite node: pins temperature/seed and records failures.

    `rewrite_node` swallows LLM errors and empty output and returns the original
    question; `failures` lets the caller notice that. Output that is empty only after
    `clean_rewrite` (for example a bare "Rewritten question:") counts as empty too.
    """

    def __init__(self, inner: LlmProvider) -> None:
        self._inner = inner
        self.failures: list[str] = []

    async def generate(
        self, prompt: str | Sequence[Mapping[str, str]], **kwargs: Any
    ) -> str:
        options = {
            **kwargs,
            "temperature": REWRITE_TEMPERATURE,
            "seed": REWRITE_SEED,
        }
        try:
            text = await self._inner.generate(prompt, **options)
        except Exception as exc:
            self.failures.append(f"{type(exc).__name__}: {exc}")
            raise
        if not clean_rewrite(text):
            self.failures.append("empty rewrite")
        return text

    def stream(
        self, prompt: str | Sequence[Mapping[str, str]], **kwargs: Any
    ) -> AsyncIterator[str]:
        return self._inner.stream(prompt, **kwargs)

    async def generate_structured(
        self,
        prompt: str | Sequence[Mapping[str, str]],
        response_model: type[T],
        **kwargs: Any,
    ) -> T:
        return await self._inner.generate_structured(prompt, response_model, **kwargs)


async def rewrite_query(
    entry: GoldenEntry,
    rewrite_model: str,
    cache: RewriteCache,
    llm: LlmProvider | None,
) -> str:
    """The cached rewrite for `entry`, or a fresh one from the real rewrite node.

    Raises `RewriteCacheMissError` on a miss without an LLM, and `RewriteFallbackError`
    when the rewrite node fell back to the original question.
    """
    key = RewriteKey.for_entry(entry, rewrite_model)
    cached = cache.get(key)
    if cached is not None:
        return cached
    if llm is None:
        raise RewriteCacheMissError(
            f"no cached rewrite for {entry.id} (model {rewrite_model!r}, "
            f"prompt {key.prompt_sha256[:8]}) and no LLM provided"
        )

    recorder = _RecordingRewriteLlm(llm)
    node = make_rewrite_node(recorder)
    state = cast(WorkflowState, {"question": entry.question})
    result = await node(state)
    if recorder.failures:
        raise RewriteFallbackError(
            f"rewrite for {entry.id} fell back to the original question: "
            + "; ".join(recorder.failures)
        )
    rewritten = str(result["rewritten"])
    cache.put(key, rewritten)
    return rewritten


# ---------------------------------------------------------------------------
# Ranked retrieval
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChunkPayload:
    """The payload fields of one chunk that the eval reads: title and text."""

    title: str
    text: str


_PAYLOAD_FIELDS = ["title", "text", "paper_id", "chunk_index"]


async def fetch_chunk_payloads(
    client: AsyncQdrantClient, collection: str, refs: Iterable[tuple[str, int]]
) -> dict[tuple[str, int], ChunkPayload]:
    """Title and text by `(paper_id, chunk_index)` for the chunks that have any text.

    Read from the Qdrant payload and kept in memory only: it feeds the reranker, the
    token estimate and the n-gram probe. A missing title becomes an empty string.
    """
    wanted = sorted(set(refs))
    if not wanted:
        return {}
    records = await client.retrieve(
        collection_name=collection,
        ids=[point_id(paper_id, index) for paper_id, index in wanted],
        with_payload=_PAYLOAD_FIELDS,
        with_vectors=False,
    )
    found: dict[tuple[str, int], ChunkPayload] = {}
    for record in records:
        payload = record.payload or {}
        paper_id, index, text = (
            payload.get("paper_id"),
            payload.get("chunk_index"),
            payload.get("text"),
        )
        if (
            isinstance(paper_id, str)
            and isinstance(index, int)
            and isinstance(text, str)
        ):
            title = payload.get("title")
            found[(paper_id, index)] = ChunkPayload(
                title=title if isinstance(title, str) else "", text=text
            )
    return found


async def rerank_hits(
    hits: Sequence[RankedHit],
    query: str,
    reranker: Reranker,
    *,
    client: AsyncQdrantClient,
    collection: str,
) -> tuple[list[RankedHit], float | None]:
    """Re-score `hits` with `reranker`; returns the hits sorted by the new score.

    The reranker reads `rerank_passage(title, section, text)`, the same text the app
    builds. The sort is stable, so ties keep the dense order. The second value is the
    time the reranker call took in milliseconds (`None` when there was nothing to
    score, because no model call was made). Raises `ValueError` when the reranker
    returns a different number of scores than passages, or a chunk has no text.
    """
    if not hits:
        return [], None
    payloads = await fetch_chunk_payloads(
        client, collection, [(h.paper_id, h.chunk_index) for h in hits]
    )
    passages: list[str] = []
    for hit in hits:
        chunk = payloads.get((hit.paper_id, hit.chunk_index))
        if chunk is None:
            raise ValueError(
                f"hit {hit.paper_id}:{hit.chunk_index} has no text in "
                f"collection {collection!r}; cannot rerank"
            )
        passages.append(rerank_passage(chunk.title, hit.section, chunk.text))

    started = time.perf_counter()
    scores = await reranker.score(query, passages)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if len(scores) != len(hits):
        raise ValueError(
            f"reranker returned {len(scores)} scores for {len(hits)} passages"
        )
    rescored = [
        replace(hit, score=float(s)) for hit, s in zip(hits, scores, strict=True)
    ]
    rescored.sort(key=lambda hit: hit.score, reverse=True)  # stable: ties keep order
    return rescored, elapsed_ms


async def retrieve_ranked(
    entry: GoldenEntry,
    config: RetrievalConfig,
    *,
    embedder: EmbeddingProvider,
    client: AsyncQdrantClient,
    cache: RewriteCache | None = None,
    llm: LlmProvider | None = None,
    reranker: Reranker | None = None,
) -> RankedResult:
    """Top-`config.depth` hits for one question at threshold 0, scores kept.

    With `config.rewrite` the query is the cached (or freshly made) rewrite, else the
    question itself. `config.prefix` is prepended to that text before embedding; the
    returned `query` is the text without the prefix.

    With `config.reranker` the dense hits below `config.prefilter_threshold` are
    dropped, the rest are re-scored by `reranker` and sorted by that score (ties keep
    the dense order); `RankedHit.score` is then the reranker score. The reranker reads
    the rewritten query or the original question (`config.rerank_query`). The
    result's `rerank_ms` is the time the reranker took.
    """
    if config.reranker is not None and reranker is None:
        raise ValueError(
            f"config asks for reranker {config.reranker!r} but none was provided"
        )
    if config.rewrite:
        if cache is None:
            raise RewriteCacheMissError(
                "rewrite is on but no rewrite cache was provided"
            )
        query = await rewrite_query(entry, config.rewrite_model, cache, llm)
    else:
        query = entry.question

    vector = await embedder.embed(config.prefix + query)
    hits = await search_hits(
        vector,
        client,
        top_k=config.depth,
        score_threshold=NO_SCORE_THRESHOLD,
        collection_name=config.collection,
    )
    if config.reranker is None or reranker is None:
        return RankedResult(query=query, hits=hits)

    candidates = [h for h in hits if h.score >= config.prefilter_threshold]
    rerank_text = entry.question if config.rerank_query == "original" else query
    rescored, rerank_ms = await rerank_hits(
        candidates, rerank_text, reranker, client=client, collection=config.collection
    )
    return RankedResult(query=query, hits=rescored, rerank_ms=rerank_ms)


# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CorpusIndex:
    """What the target collection holds, built from one payload-only scroll.

    `sections`: paper_id -> section -> sorted chunk indices.
    `chunk_types`: (paper_id, chunk_index) -> chunk type.
    """

    collection: str
    chunk_count: int
    sections: dict[str, dict[str, tuple[int, ...]]]
    chunk_types: dict[tuple[str, int], str]
    embedding_models: frozenset[str]
    skipped_points: int = 0
    type_counts: Counter[str] = field(default_factory=Counter)

    def has_section(self, paper_id: str, section: str) -> bool:
        return section in self.sections.get(paper_id, {})


class PreflightError(ValueError):
    """The golden set or collection is not fit to run; `problems` lists all of them."""

    def __init__(self, problems: Sequence[str]) -> None:
        self.problems = list(problems)
        super().__init__(
            f"{len(self.problems)} pre-flight problem(s):\n"
            + "\n".join(f"- {p}" for p in self.problems)
        )


def parse_source_chunk(ref: str) -> tuple[str, int] | None:
    """`"paper_id:chunk_index"` -> `(paper_id, chunk_index)`, or `None` if malformed."""
    match = _SOURCE_CHUNK_RE.match(ref)
    if match is None:
        return None
    return match["paper"], int(match["index"])


async def scan_collection(client: AsyncQdrantClient, collection: str) -> CorpusIndex:
    """Scroll every point of `collection` (payload fields only) into a `CorpusIndex`."""
    sections: dict[str, dict[str, set[int]]] = {}
    chunk_types: dict[tuple[str, int], str] = {}
    models: set[str] = set()
    skipped = 0
    count = 0

    offset: Any = None
    while True:
        points, offset = await client.scroll(
            collection_name=collection,
            limit=_SCROLL_PAGE_SIZE,
            offset=offset,
            with_payload=list(PREFLIGHT_PAYLOAD_FIELDS),
            with_vectors=False,
        )
        for point in points:
            count += 1
            payload = point.payload or {}
            model = payload.get("embedding_model")
            models.add(model if isinstance(model, str) and model else _MISSING_MODEL)
            paper_id = payload.get("paper_id")
            section = payload.get("section")
            chunk_index = payload.get("chunk_index")
            if (
                not isinstance(paper_id, str)
                or not paper_id
                or not isinstance(section, str)
                or not section
                or not isinstance(chunk_index, int)
                or isinstance(chunk_index, bool)
            ):
                skipped += 1
                continue
            chunk_type = payload.get("chunk_type")
            sections.setdefault(paper_id, {}).setdefault(section, set()).add(
                chunk_index
            )
            chunk_types[(paper_id, chunk_index)] = (
                chunk_type if isinstance(chunk_type, str) and chunk_type else "text"
            )
        if offset is None:
            break

    return CorpusIndex(
        collection=collection,
        chunk_count=count,
        sections={
            paper: {sec: tuple(sorted(idx)) for sec, idx in secs.items()}
            for paper, secs in sections.items()
        },
        chunk_types=chunk_types,
        embedding_models=frozenset(models),
        skipped_points=skipped,
        type_counts=Counter(chunk_types.values()),
    )


def preflight_problems(
    entries: Iterable[GoldenEntry], index: CorpusIndex, embedder: str
) -> list[str]:
    """Every reason the golden set cannot be scored against `index` (empty = fit)."""
    problems: list[str] = []

    if index.chunk_count == 0:
        problems.append(f"collection {index.collection!r} has no points")
    if index.skipped_points:
        problems.append(
            f"{index.skipped_points} point(s) in {index.collection!r} lack paper_id, "
            "section or chunk_index (stale index?)"
        )
    if index.embedding_models and index.embedding_models != {embedder}:
        found = ", ".join(sorted(index.embedding_models))
        problems.append(
            f"collection {index.collection!r} was embedded with {found}, "
            f"but this run uses {embedder!r}"
        )

    for entry in entries:
        for label in entry.relevant:
            if not index.has_section(label.paper_id, label.section):
                problems.append(
                    f"{entry.id}: labelled section does not exist: "
                    f"{label.paper_id} / {label.section!r}"
                )

        source_types: list[str] = []
        for ref in entry.source_chunks:
            parsed = parse_source_chunk(ref)
            chunk_type = index.chunk_types.get(parsed) if parsed else None
            if chunk_type is None:
                problems.append(f"{entry.id}: source chunk does not exist: {ref!r}")
            else:
                source_types.append(chunk_type)
        if entry.type is QuestionType.TABLE and TABLE_CHUNK_TYPE not in source_types:
            problems.append(
                f"{entry.id}: table entry has no source chunk of type "
                f"{TABLE_CHUNK_TYPE!r}"
            )
    return problems


async def preflight(
    entries: Sequence[GoldenEntry],
    client: AsyncQdrantClient,
    config: RetrievalConfig,
) -> CorpusIndex:
    """Check the golden set against the target collection before any scoring.

    Raises one `PreflightError` listing all problems; otherwise returns the corpus index
    (sections per paper with chunk ids, chunk count, chunk types).
    """
    index = await scan_collection(client, config.collection)
    problems = preflight_problems(entries, index, config.embedder)
    if problems:
        raise PreflightError(problems)
    return index


# ---------------------------------------------------------------------------
# N-gram overlap (U3 paraphrase probe)
# ---------------------------------------------------------------------------


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def longest_shared_ngram(question: str, texts: Iterable[str]) -> int:
    """Length in words of the longest word sequence the question shares with any text.

    Case-insensitive, punctuation ignored. 0 when nothing is shared.
    """
    q_words = _words(question)
    best = 0
    for text in texts:
        t_words = _words(text)
        # Longest common substring over word lists, one row of the DP table at a time.
        previous = [0] * (len(t_words) + 1)
        for q_word in q_words:
            current = [0] * (len(t_words) + 1)
            for j, t_word in enumerate(t_words, start=1):
                if q_word == t_word:
                    current[j] = previous[j - 1] + 1
                    best = max(best, current[j])
            previous = current
    return best


async def source_overlap(
    entry: GoldenEntry, client: AsyncQdrantClient, collection: str
) -> int:
    """Longest word n-gram shared by the question and its source chunks' text.

    The text is read from the Qdrant payload (`text`) and never stored. An entry without
    source chunks (off-corpus) scores 0; a source chunk that does not exist is ignored
    (the pre-flight reports it).
    """
    ids = [
        point_id(*parsed)
        for ref in entry.source_chunks
        if (parsed := parse_source_chunk(ref)) is not None
    ]
    if not ids:
        return 0
    records = await client.retrieve(
        collection_name=collection,
        ids=ids,
        with_payload=["text"],
        with_vectors=False,
    )
    texts = [
        text
        for record in records
        if isinstance(text := (record.payload or {}).get("text"), str)
    ]
    return longest_shared_ngram(entry.question, texts)
