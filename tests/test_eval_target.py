"""Tests for the retrieval target: rewrite cache, ranked retrieval, pre-flight, overlap.

No Qdrant, Ollama or network: a fake async Qdrant, a fake embedder and `FakeLlm`.
"""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from app.evaluation import target
from app.evaluation.golden import GoldenEntry, QuestionType
from app.evaluation.target import (
    PreflightError,
    RetrievalConfig,
    RewriteCache,
    RewriteCacheMissError,
    RewriteFallbackError,
    RewriteKey,
    longest_shared_ngram,
    parse_source_chunk,
    preflight,
    retrieve_ranked,
    rewrite_prompt_sha256,
    rewrite_query,
    source_overlap,
)
from app.retrieval.ingestion import point_id
from tests.conftest import FakeLlm

MODEL = "BAAI/bge-small-en-v1.5"
COLLECTION = "papers"
REWRITER = "qwen3:0.6b"


# ---------------------------------------------------------------------------
# Test doubles and builders
# ---------------------------------------------------------------------------


class RecordingLlm(FakeLlm):
    """`FakeLlm` that also keeps the keyword arguments of every `generate` call."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.kwargs: list[dict[str, Any]] = []

    async def generate(
        self, prompt: str | Sequence[Mapping[str, str]], **kwargs: Any
    ) -> str:
        self.kwargs.append(kwargs)
        return await super().generate(prompt, **kwargs)


class FakeEmbedder:
    def __init__(self) -> None:
        self.texts: list[str] = []

    async def embed(self, text: str) -> npt.NDArray[np.float32]:
        self.texts.append(text)
        return np.zeros(3, dtype=np.float32)


def make_point(
    paper_id: str,
    section: str,
    chunk_index: int,
    *,
    chunk_type: str = "text",
    model: str | None = MODEL,
    text: str = "",
    score: float = 0.9,
) -> SimpleNamespace:
    payload: dict[str, Any] = {
        "paper_id": paper_id,
        "section": section,
        "chunk_index": chunk_index,
        "chunk_type": chunk_type,
        "text": text,
    }
    if model is not None:
        payload["embedding_model"] = model
    return SimpleNamespace(
        id=point_id(paper_id, chunk_index), score=score, payload=payload
    )


class FakeAsyncQdrant:
    """The slice of `AsyncQdrantClient` used: query_points, scroll, retrieve."""

    def __init__(self, points: list[SimpleNamespace]) -> None:
        self.points = points
        self.query_calls: list[dict[str, Any]] = []
        self.scroll_calls: list[dict[str, Any]] = []

    async def query_points(self, **kwargs: Any) -> SimpleNamespace:
        self.query_calls.append(kwargs)
        kept = [p for p in self.points if p.score >= kwargs["score_threshold"]]
        kept.sort(key=lambda p: p.score, reverse=True)
        return SimpleNamespace(points=kept[: kwargs["limit"]])

    async def scroll(self, **kwargs: Any) -> tuple[list[SimpleNamespace], int | None]:
        self.scroll_calls.append(kwargs)
        start = kwargs["offset"] or 0
        end = start + kwargs["limit"]
        page = [SimpleNamespace(payload=p.payload) for p in self.points[start:end]]
        return page, (end if end < len(self.points) else None)

    async def retrieve(self, **kwargs: Any) -> list[SimpleNamespace]:
        wanted = set(kwargs["ids"])
        return [p for p in self.points if p.id in wanted]


def make_entry(
    entry_id: str = "q001",
    *,
    question: str = "How well does the model detect pneumothorax?",
    type_: QuestionType = QuestionType.FACTUAL,
    relevant: list[tuple[str, str, int]] | None = None,
    source_chunks: list[str] | None = None,
) -> GoldenEntry:
    return GoldenEntry.model_validate(
        {
            "id": entry_id,
            "question": question,
            "type": type_,
            "relevant": [
                {"paper_id": p, "section": s, "grade": g} for p, s, g in relevant or []
            ],
            "source_chunks": source_chunks or [],
        }
    )


def make_config(**overrides: Any) -> RetrievalConfig:
    params: dict[str, Any] = {
        "embedder": MODEL,
        "collection": COLLECTION,
        "rewrite": False,
    }
    params.update(overrides)
    return RetrievalConfig(**params)


@pytest.fixture
def corpus() -> FakeAsyncQdrant:
    return FakeAsyncQdrant(
        [
            make_point("maira_2", "Results", 0, text="alpha beta gamma delta epsilon"),
            make_point("maira_2", "Methods", 1, text="zeta eta theta", score=0.8),
            make_point("conch", "Results > Table 2", 0, chunk_type="table", score=0.7),
        ]
    )


# ---------------------------------------------------------------------------
# RetrievalConfig
# ---------------------------------------------------------------------------


class TestRetrievalConfig:
    def test_defaults(self) -> None:
        config = make_config()
        assert config.depth == 20
        assert config.prefix == ""

    def test_rewrite_requires_a_model(self) -> None:
        with pytest.raises(ValueError, match="rewrite_model"):
            RetrievalConfig(embedder=MODEL, collection=COLLECTION, rewrite=True)

    def test_depth_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="depth"):
            make_config(depth=0)


# ---------------------------------------------------------------------------
# Rewrite cache
# ---------------------------------------------------------------------------


class TestRewriteCache:
    async def test_hit_does_not_call_the_llm(self, tmp_path: Path) -> None:
        entry = make_entry()
        cache = RewriteCache(tmp_path / "rewrites.jsonl")
        cache.put(RewriteKey.for_entry(entry, REWRITER), "cached rewrite")
        llm = RecordingLlm("fresh rewrite")

        result = await rewrite_query(entry, REWRITER, cache, llm)

        assert result == "cached rewrite"
        assert llm.calls == []

    async def test_miss_calls_the_real_rewrite_node_and_stores_it(
        self, tmp_path: Path
    ) -> None:
        entry = make_entry()
        path = tmp_path / "data" / "eval" / "rewrites.jsonl"
        cache = RewriteCache(path)
        llm = RecordingLlm("  pneumothorax detection performance \n")

        result = await rewrite_query(entry, REWRITER, cache, llm)

        assert result == "pneumothorax detection performance"
        # The real prompt template was used, with the question filled in.
        assert len(llm.calls) == 1
        assert isinstance(llm.calls[0], str)
        assert entry.question in llm.calls[0]
        assert "medical literature search assistant" in llm.calls[0]
        # Deterministic generation options.
        assert llm.kwargs == [{"temperature": 0, "seed": 0}]
        # Persisted, and found again by a fresh cache without any LLM.
        reloaded = RewriteCache(path)
        assert await rewrite_query(entry, REWRITER, reloaded, None) == result
        row = json.loads(path.read_text(encoding="utf-8").strip())
        assert row["id"] == "q001"
        assert row["model"] == REWRITER
        assert row["prompt_sha256"] == rewrite_prompt_sha256()
        assert entry.question not in path.read_text(encoding="utf-8")

    async def test_miss_without_llm_raises(self, tmp_path: Path) -> None:
        cache = RewriteCache(tmp_path / "rewrites.jsonl")
        with pytest.raises(RewriteCacheMissError, match="q001"):
            await rewrite_query(make_entry(), REWRITER, cache, None)

    async def test_llm_error_is_a_failure_not_a_silent_fallback(
        self, tmp_path: Path
    ) -> None:
        cache = RewriteCache(tmp_path / "rewrites.jsonl")
        llm = FakeLlm(error=RuntimeError("ollama down"))

        with pytest.raises(RewriteFallbackError, match="ollama down"):
            await rewrite_query(make_entry(), REWRITER, cache, llm)

        assert len(cache) == 0
        assert not (tmp_path / "rewrites.jsonl").exists()

    async def test_empty_rewrite_is_a_failure(self, tmp_path: Path) -> None:
        cache = RewriteCache(tmp_path / "rewrites.jsonl")
        with pytest.raises(RewriteFallbackError, match="empty rewrite"):
            await rewrite_query(make_entry(), REWRITER, cache, FakeLlm(answer="   "))
        assert len(cache) == 0

    async def test_a_rewrite_equal_to_the_question_is_accepted(
        self, tmp_path: Path
    ) -> None:
        entry = make_entry()
        cache = RewriteCache(tmp_path / "rewrites.jsonl")
        result = await rewrite_query(entry, REWRITER, cache, FakeLlm(entry.question))
        assert result == entry.question

    async def test_key_includes_question_model_and_prompt(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        entry = make_entry()
        cache = RewriteCache(tmp_path / "rewrites.jsonl")
        cache.put(RewriteKey.for_entry(entry, REWRITER), "old")

        # Another rewrite model, an edited question and an edited prompt all miss.
        with pytest.raises(RewriteCacheMissError):
            await rewrite_query(entry, "other-model", cache, None)
        edited = make_entry(question=entry.question + " today")
        with pytest.raises(RewriteCacheMissError):
            await rewrite_query(edited, REWRITER, cache, None)
        monkeypatch.setattr(target, "REWRITE_PROMPT", "different prompt {question}")
        with pytest.raises(RewriteCacheMissError):
            await rewrite_query(entry, REWRITER, cache, None)

    def test_prompt_hash_covers_the_real_prompt_and_the_cleanup_version(self) -> None:
        import hashlib

        from app.workflow import rewrite

        text = f"{rewrite._PROMPT}\n{rewrite.REWRITE_CLEANUP_VERSION}"
        expected = hashlib.sha256(text.encode("utf-8")).hexdigest()
        assert rewrite_prompt_sha256() == expected

    async def test_a_new_cleanup_version_misses_the_cache(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        entry = make_entry()
        cache = RewriteCache(tmp_path / "rewrites.jsonl")
        cache.put(RewriteKey.for_entry(entry, REWRITER), "old")

        monkeypatch.setattr(target, "REWRITE_CLEANUP_VERSION", "next")
        with pytest.raises(RewriteCacheMissError):
            await rewrite_query(entry, REWRITER, cache, None)

    async def test_a_leaked_label_is_cleaned_before_it_is_stored(
        self, tmp_path: Path
    ) -> None:
        entry = make_entry()
        path = tmp_path / "rewrites.jsonl"
        cache = RewriteCache(path)
        llm = FakeLlm(answer="Rewritten question: pneumothorax detection")

        assert await rewrite_query(entry, REWRITER, cache, llm) == (
            "pneumothorax detection"
        )
        assert json.loads(path.read_text(encoding="utf-8"))["rewrite"] == (
            "pneumothorax detection"
        )

    async def test_output_empty_after_cleaning_is_a_failure(
        self, tmp_path: Path
    ) -> None:
        cache = RewriteCache(tmp_path / "rewrites.jsonl")
        llm = FakeLlm(answer="Rewritten question:")
        with pytest.raises(RewriteFallbackError, match="empty rewrite"):
            await rewrite_query(make_entry(), REWRITER, cache, llm)
        assert len(cache) == 0

    def test_committed_rewrites_have_no_leaked_label(self) -> None:
        cache = RewriteCache(target.DEFAULT_REWRITES_PATH)
        assert len(cache) > 0
        leaked = [
            key.question_id
            for key in cache._rewrites
            if cache._rewrites[key].lower().lstrip().startswith("rewritten question")
        ]
        assert leaked == []

    def test_committed_rewrites_use_the_current_prompt_hash(self) -> None:
        cache = RewriteCache(target.DEFAULT_REWRITES_PATH)
        hashes = {key.prompt_sha256 for key in cache._rewrites}
        assert hashes == {rewrite_prompt_sha256()}

    def test_bad_cache_line_names_the_line(self, tmp_path: Path) -> None:
        path = tmp_path / "rewrites.jsonl"
        path.write_text('{"id": "q001"}\n', encoding="utf-8")
        with pytest.raises(ValueError, match=r"rewrites\.jsonl:1"):
            RewriteCache(path)


# ---------------------------------------------------------------------------
# retrieve_ranked
# ---------------------------------------------------------------------------


class TestRetrieveRanked:
    async def test_returns_scored_hits_at_threshold_zero_with_depth(
        self, corpus: FakeAsyncQdrant
    ) -> None:
        entry = make_entry()
        result = await retrieve_ranked(
            entry,
            make_config(collection="eval_x", depth=2),
            embedder=FakeEmbedder(),
            client=corpus,  # type: ignore[arg-type]
        )

        assert result.query == entry.question
        assert [(h.paper_id, h.section, h.score) for h in result.hits] == [
            ("maira_2", "Results", 0.9),
            ("maira_2", "Methods", 0.8),
        ]
        call = corpus.query_calls[0]
        assert call["score_threshold"] == 0.0
        assert call["limit"] == 2
        assert call["collection_name"] == "eval_x"

    async def test_default_depth_is_20(self, corpus: FakeAsyncQdrant) -> None:
        await retrieve_ranked(
            make_entry(),
            make_config(),
            embedder=FakeEmbedder(),
            client=corpus,  # type: ignore[arg-type]
        )
        assert corpus.query_calls[0]["limit"] == 20

    async def test_chunk_type_is_kept(self, corpus: FakeAsyncQdrant) -> None:
        result = await retrieve_ranked(
            make_entry(),
            make_config(),
            embedder=FakeEmbedder(),
            client=corpus,  # type: ignore[arg-type]
        )
        assert [h.chunk_type for h in result.hits] == ["text", "text", "table"]

    async def test_prefix_is_prepended_to_the_embedded_text(
        self, corpus: FakeAsyncQdrant
    ) -> None:
        entry = make_entry()
        embedder = FakeEmbedder()
        result = await retrieve_ranked(
            entry,
            make_config(prefix="Represent this: "),
            embedder=embedder,
            client=corpus,  # type: ignore[arg-type]
        )
        assert embedder.texts == ["Represent this: " + entry.question]
        # The returned query is the text without the prefix.
        assert result.query == entry.question

    async def test_prefix_applies_to_the_rewritten_query(
        self, corpus: FakeAsyncQdrant, tmp_path: Path
    ) -> None:
        entry = make_entry()
        cache = RewriteCache(tmp_path / "rewrites.jsonl")
        cache.put(RewriteKey.for_entry(entry, REWRITER), "rewritten text")
        embedder = FakeEmbedder()

        result = await retrieve_ranked(
            entry,
            make_config(rewrite=True, rewrite_model=REWRITER, prefix="P: "),
            embedder=embedder,
            client=corpus,  # type: ignore[arg-type]
            cache=cache,
        )

        assert embedder.texts == ["P: rewritten text"]
        assert result.query == "rewritten text"

    async def test_rewrite_off_never_touches_the_cache_or_llm(
        self, corpus: FakeAsyncQdrant
    ) -> None:
        llm = FakeLlm()
        result = await retrieve_ranked(
            make_entry(),
            make_config(rewrite=False),
            embedder=FakeEmbedder(),
            client=corpus,  # type: ignore[arg-type]
            llm=llm,
        )
        assert llm.calls == []
        assert result.query == make_entry().question

    async def test_rewrite_miss_without_llm_raises(
        self, corpus: FakeAsyncQdrant, tmp_path: Path
    ) -> None:
        with pytest.raises(RewriteCacheMissError):
            await retrieve_ranked(
                make_entry(),
                make_config(rewrite=True, rewrite_model=REWRITER),
                embedder=FakeEmbedder(),
                client=corpus,  # type: ignore[arg-type]
                cache=RewriteCache(tmp_path / "rewrites.jsonl"),
            )

    async def test_rewrite_fallback_raises_instead_of_retrieving(
        self, corpus: FakeAsyncQdrant, tmp_path: Path
    ) -> None:
        with pytest.raises(RewriteFallbackError):
            await retrieve_ranked(
                make_entry(),
                make_config(rewrite=True, rewrite_model=REWRITER),
                embedder=FakeEmbedder(),
                client=corpus,  # type: ignore[arg-type]
                cache=RewriteCache(tmp_path / "rewrites.jsonl"),
                llm=FakeLlm(error=ConnectionError("refused")),
            )
        assert corpus.query_calls == []

    async def test_rewrite_on_without_a_cache_raises(
        self, corpus: FakeAsyncQdrant
    ) -> None:
        with pytest.raises(RewriteCacheMissError):
            await retrieve_ranked(
                make_entry(),
                make_config(rewrite=True, rewrite_model=REWRITER),
                embedder=FakeEmbedder(),
                client=corpus,  # type: ignore[arg-type]
                llm=FakeLlm(),
            )


# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------


def good_entries() -> list[GoldenEntry]:
    return [
        make_entry(
            "q001",
            relevant=[("maira_2", "Results", 2), ("maira_2", "Methods", 1)],
            source_chunks=["maira_2:0"],
        ),
        make_entry(
            "q002",
            type_=QuestionType.TABLE,
            relevant=[("conch", "Results > Table 2", 2)],
            source_chunks=["conch:0"],
        ),
        make_entry("q003", type_=QuestionType.OFF_CORPUS),
    ]


class TestPreflight:
    async def test_passes_and_returns_the_corpus_index(
        self, corpus: FakeAsyncQdrant
    ) -> None:
        index = await preflight(
            good_entries(),
            corpus,  # type: ignore[arg-type]
            make_config(),
        )

        assert index.chunk_count == 3
        assert index.sections == {
            "maira_2": {"Results": (0,), "Methods": (1,)},
            "conch": {"Results > Table 2": (0,)},
        }
        assert index.chunk_types[("conch", 0)] == "table"
        assert index.type_counts == {"text": 2, "table": 1}
        assert index.embedding_models == {MODEL}

    async def test_scrolls_payload_only_and_follows_pages(
        self, corpus: FakeAsyncQdrant, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(target, "_SCROLL_PAGE_SIZE", 2)
        index = await preflight(
            good_entries(),
            corpus,  # type: ignore[arg-type]
            make_config(),
        )

        assert index.chunk_count == 3
        assert len(corpus.scroll_calls) == 2
        first = corpus.scroll_calls[0]
        assert first["with_vectors"] is False
        assert first["with_payload"] == [
            "paper_id",
            "section",
            "chunk_index",
            "chunk_type",
            "embedding_model",
        ]

    async def test_missing_section_is_reported(self, corpus: FakeAsyncQdrant) -> None:
        entries = [make_entry(relevant=[("maira_2", "Nope", 1)])]
        with pytest.raises(PreflightError, match="section does not exist") as info:
            await preflight(entries, corpus, make_config())  # type: ignore[arg-type]
        assert "maira_2 / 'Nope'" in str(info.value)

    async def test_section_of_another_paper_does_not_count(
        self, corpus: FakeAsyncQdrant
    ) -> None:
        entries = [make_entry(relevant=[("conch", "Results", 1)])]
        with pytest.raises(PreflightError, match="section does not exist"):
            await preflight(entries, corpus, make_config())  # type: ignore[arg-type]

    async def test_missing_source_chunk_is_reported(
        self, corpus: FakeAsyncQdrant
    ) -> None:
        entries = [make_entry(source_chunks=["maira_2:99", "garbage"])]
        with pytest.raises(PreflightError) as info:
            await preflight(entries, corpus, make_config())  # type: ignore[arg-type]
        assert "'maira_2:99'" in str(info.value)
        assert "'garbage'" in str(info.value)

    async def test_table_entry_without_table_source_chunk_is_reported(
        self, corpus: FakeAsyncQdrant
    ) -> None:
        entries = [
            make_entry(
                "q009",
                type_=QuestionType.TABLE,
                relevant=[("maira_2", "Results", 2)],
                source_chunks=["maira_2:0"],  # exists, but is a text chunk
            )
        ]
        with pytest.raises(PreflightError, match="q009: table entry"):
            await preflight(entries, corpus, make_config())  # type: ignore[arg-type]

    async def test_table_entry_with_one_table_source_chunk_is_fine(
        self, corpus: FakeAsyncQdrant
    ) -> None:
        entries = [
            make_entry(
                type_=QuestionType.TABLE,
                relevant=[("conch", "Results > Table 2", 2)],
                source_chunks=["maira_2:0", "conch:0"],
            )
        ]
        await preflight(entries, corpus, make_config())  # type: ignore[arg-type]

    async def test_embedding_model_mismatch_is_reported(
        self, corpus: FakeAsyncQdrant
    ) -> None:
        config = make_config(embedder="BAAI/bge-base-en-v1.5")
        with pytest.raises(PreflightError) as info:
            await preflight(good_entries(), corpus, config)  # type: ignore[arg-type]
        message = str(info.value)
        assert MODEL in message
        assert "BAAI/bge-base-en-v1.5" in message

    async def test_mixed_embedding_models_are_reported(self) -> None:
        client = FakeAsyncQdrant(
            [
                make_point("maira_2", "Results", 0, model=MODEL),
                make_point("maira_2", "Results", 1, model="other/model"),
            ]
        )
        with pytest.raises(PreflightError, match="other/model"):
            await preflight([], client, make_config())  # type: ignore[arg-type]

    async def test_points_without_embedding_model_are_reported(self) -> None:
        client = FakeAsyncQdrant([make_point("maira_2", "Results", 0, model=None)])
        with pytest.raises(PreflightError, match="<missing>"):
            await preflight([], client, make_config())  # type: ignore[arg-type]

    async def test_empty_collection_is_reported(self) -> None:
        with pytest.raises(PreflightError, match="has no points"):
            await preflight([], FakeAsyncQdrant([]), make_config())  # type: ignore[arg-type]

    async def test_points_without_chunk_identity_are_reported(self) -> None:
        stale = make_point("maira_2", "Results", 0)
        del stale.payload["section"]
        client = FakeAsyncQdrant([stale, make_point("maira_2", "Results", 1)])
        with pytest.raises(PreflightError, match="lack paper_id"):
            await preflight([], client, make_config())  # type: ignore[arg-type]

    async def test_all_problems_are_reported_in_one_error(
        self, corpus: FakeAsyncQdrant
    ) -> None:
        entries = [
            make_entry("q001", relevant=[("maira_2", "Nope", 1)]),
            make_entry("q002", source_chunks=["maira_2:99"]),
            make_entry(
                "q003",
                type_=QuestionType.TABLE,
                relevant=[("maira_2", "Results", 2)],
                source_chunks=["maira_2:0"],
            ),
        ]
        config = make_config(embedder="BAAI/bge-base-en-v1.5")

        with pytest.raises(PreflightError) as info:
            await preflight(entries, corpus, config)  # type: ignore[arg-type]

        problems = info.value.problems
        assert len(problems) == 4
        text = str(info.value)
        assert "q001: labelled section does not exist" in text
        assert "q002: source chunk does not exist" in text
        assert "q003: table entry" in text
        assert "was embedded with" in text
        assert text.startswith("4 pre-flight problem(s)")


# ---------------------------------------------------------------------------
# N-gram overlap
# ---------------------------------------------------------------------------


class TestParseSourceChunk:
    def test_parses_paper_and_index(self) -> None:
        assert parse_source_chunk("maira_2:12") == ("maira_2", 12)

    @pytest.mark.parametrize("ref", ["maira_2", "maira_2:x", ":3", "a:b:1", ""])
    def test_malformed_is_none(self, ref: str) -> None:
        assert parse_source_chunk(ref) is None


class TestLongestSharedNgram:
    def test_counts_the_longest_consecutive_run(self) -> None:
        question = "Which model reaches high recall on chest x-ray findings?"
        text = "We report that it reaches high recall on chest x-ray findings overall."
        assert longest_shared_ngram(question, [text]) == 8

    def test_ignores_case_and_punctuation(self) -> None:
        assert (
            longest_shared_ngram("Chest X-Ray, Findings!", ["chest x ray findings"])
            == 4
        )

    def test_scattered_words_do_not_add_up(self) -> None:
        assert (
            longest_shared_ngram("alpha beta gamma", ["alpha delta beta epsilon gamma"])
            == 1
        )

    def test_takes_the_best_text(self) -> None:
        texts = ["alpha unrelated", "alpha beta gamma here"]
        assert longest_shared_ngram("alpha beta gamma", texts) == 3

    def test_no_overlap_and_no_texts_are_zero(self) -> None:
        assert longest_shared_ngram("alpha beta", ["gamma delta"]) == 0
        assert longest_shared_ngram("alpha beta", []) == 0
        assert longest_shared_ngram("", ["gamma"]) == 0

    async def test_source_overlap_reads_the_text_from_qdrant(
        self, corpus: FakeAsyncQdrant
    ) -> None:
        entry = make_entry(
            question="what about beta gamma delta in this study",
            source_chunks=["maira_2:0", "maira_2:1"],
        )
        overlap = await source_overlap(entry, corpus, COLLECTION)  # type: ignore[arg-type]
        assert overlap == 3

    async def test_entry_without_source_chunks_is_zero(
        self, corpus: FakeAsyncQdrant
    ) -> None:
        entry = make_entry(type_=QuestionType.OFF_CORPUS)
        assert await source_overlap(entry, corpus, COLLECTION) == 0  # type: ignore[arg-type]

    async def test_missing_source_chunk_is_ignored(
        self, corpus: FakeAsyncQdrant
    ) -> None:
        entry = make_entry(
            question="beta gamma", source_chunks=["maira_2:0", "maira_2:99"]
        )
        assert await source_overlap(entry, corpus, COLLECTION) == 2  # type: ignore[arg-type]
