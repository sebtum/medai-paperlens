import json
import logging
import uuid
from collections.abc import AsyncGenerator
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from asgi_lifespan import LifespanManager
from conftest import TEST_VECTOR_DIM, FakeLlm
from docling_builders import DocBuilder, WhitespaceTokenizer, words
from httpx import ASGITransport, AsyncClient
from qdrant_client.models import (
    CollectionDescription,
    CollectionsResponse,
    Distance,
    FilterSelector,
    PointStruct,
    VectorParams,
)

from app.llm.factory import get_llm_provider
from app.main import app
from app.models.query import DEFAULT_SCORE_THRESHOLD, QueryRequest
from app.retrieval.client import get_async_client
from app.retrieval.embedding import embed, get_embedding_provider, get_model_name
from app.retrieval.ingestion import COLLECTION_NAME, ingest, point_id
from app.retrieval.search import search

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_qdrant() -> MagicMock:
    client = MagicMock()
    client.get_collections.return_value = CollectionsResponse(collections=[])
    return client


@pytest.fixture
def async_qdrant() -> AsyncMock:
    client = AsyncMock()
    client.query_points.return_value.points = []
    return client


@pytest.fixture
def mock_model() -> MagicMock:
    model = MagicMock()
    model.encode.return_value = np.zeros(TEST_VECTOR_DIM, dtype=np.float32)
    model.get_sentence_embedding_dimension.return_value = TEST_VECTOR_DIM
    return model


@pytest.fixture
def fake_llm() -> FakeLlm:
    return FakeLlm(answer="Mocked answer.")


@pytest.fixture
async def http_client(
    async_qdrant: AsyncMock, fake_llm: FakeLlm
) -> AsyncGenerator[AsyncClient]:
    mock_provider = AsyncMock()
    mock_provider.embed.return_value = np.zeros(TEST_VECTOR_DIM, dtype=np.float32)

    app.dependency_overrides[get_async_client] = lambda: async_qdrant
    app.dependency_overrides[get_embedding_provider] = lambda: mock_provider
    app.dependency_overrides[get_llm_provider] = lambda: fake_llm

    async with (
        LifespanManager(app) as manager,
        AsyncClient(
            transport=ASGITransport(app=manager.app), base_url="http://test"
        ) as client,
    ):
        yield client

    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# TestEmbed
# ---------------------------------------------------------------------------


class TestEmbed:
    def test_returns_ndarray_from_model_encode(self, mock_model: MagicMock) -> None:
        result = embed("test query", mock_model)
        assert isinstance(result, np.ndarray)
        assert result.shape == (TEST_VECTOR_DIM,)
        mock_model.encode.assert_called_once_with("test query")


class TestEmbeddingModelName:
    def test_defaults_to_bge_small(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
        assert get_model_name() == "BAAI/bge-small-en-v1.5"

    def test_env_var_overrides_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
        assert get_model_name() == "BAAI/bge-base-en-v1.5"

    def test_empty_env_var_falls_back_to_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("EMBEDDING_MODEL", "")
        assert get_model_name() == "BAAI/bge-small-en-v1.5"


# ---------------------------------------------------------------------------
# TestIngest
# ---------------------------------------------------------------------------


class FakeQdrant:
    """In-memory stand-in for the parts of `QdrantClient` that ingestion uses."""

    def __init__(self) -> None:
        self.collections: dict[str, VectorParams] = {}
        self.points: dict[str, PointStruct] = {}
        self.calls: list[str] = []

    def get_collections(self) -> CollectionsResponse:
        return CollectionsResponse(
            collections=[CollectionDescription(name=n) for n in self.collections]
        )

    def create_collection(
        self, collection_name: str, vectors_config: VectorParams
    ) -> None:
        self.calls.append("create_collection")
        self.collections[collection_name] = vectors_config

    def delete_collection(self, collection_name: str) -> None:
        self.calls.append("delete_collection")
        self.collections.pop(collection_name)
        self.points.clear()

    def get_collection(self, collection_name: str) -> SimpleNamespace:
        vectors = self.collections[collection_name]
        params = SimpleNamespace(vectors=vectors)
        return SimpleNamespace(config=SimpleNamespace(params=params))

    def scroll(
        self, collection_name: str, limit: int, **kwargs: Any
    ) -> tuple[list[SimpleNamespace], None]:
        sample = list(self.points.values())[:limit]
        return [SimpleNamespace(payload=p.payload) for p in sample], None

    def delete(self, collection_name: str, points_selector: FilterSelector) -> None:
        self.calls.append("delete_points")
        condition = points_selector.filter.must[0]
        paper_id = condition.match.value
        self.points = {
            pid: p
            for pid, p in self.points.items()
            if (p.payload or {}).get("paper_id") != paper_id
        }

    def upsert(self, collection_name: str, points: list[PointStruct]) -> None:
        self.calls.append("upsert")
        for point in points:
            self.points[str(point.id)] = point

    def ids_of(self, paper_id: str) -> set[str]:
        return {
            pid
            for pid, p in self.points.items()
            if (p.payload or {}).get("paper_id") == paper_id
        }


@pytest.fixture
def fake_qdrant() -> FakeQdrant:
    return FakeQdrant()


@pytest.fixture
def ingest_model() -> MagicMock:
    """Embedder double: one zero vector per input text."""
    model = MagicMock()
    model.get_sentence_embedding_dimension.return_value = TEST_VECTOR_DIM
    model.encode.side_effect = lambda texts, **_: np.zeros(
        (len(texts), TEST_VECTOR_DIM), dtype=np.float32
    )
    return model


MODEL_NAME = "test/embedder"


class TestIngest:
    def _write_paper(
        self, tmp_path: Path, name: str = "paper.json", **overrides: object
    ) -> None:
        paper = {
            "title": "Test Paper",
            "authors": ["Author A"],
            "year": 2024,
            "doi": "10.test/123",
            "abstract": "A test abstract about AI in medicine.",
            **overrides,
        }
        (tmp_path / name).write_text(json.dumps(paper))

    def _write_parsed(self, parsed_dir: Path, paper_id: str = "paper") -> None:
        doc = (
            DocBuilder(name=paper_id)
            .title("Test Paper")
            .heading("1 Introduction", page=1)
            .text(words("intro", 120), page=1)
            .heading("2 Methods", page=2)
            .heading("2.1 Datasets", page=3)
            .text(words("data", 120), page=3)
            .heading("3 Results", page=5)
            .text(words("res", 120), page=5)
            .heading("References", page=9)
            .text(words("ref", 120), page=9)
            .build()
        )
        parsed_dir.mkdir(parents=True, exist_ok=True)
        doc.save_as_json(parsed_dir / f"{paper_id}.json")

    def _ingest(
        self,
        papers_dir: Path,
        client: FakeQdrant,
        model: MagicMock,
        *,
        parsed_dir: Path | None = None,
        overwrite: bool = False,
    ) -> int:
        return ingest(
            papers_dir,
            client,  # type: ignore[arg-type]
            model,
            model_name=MODEL_NAME,
            parsed_dir=parsed_dir,
            overwrite=overwrite,
            tokenizer=WhitespaceTokenizer(max_tokens=200),
        )

    def _seed_collection(
        self,
        client: FakeQdrant,
        *,
        size: int = TEST_VECTOR_DIM,
        embedding_model: str | None = MODEL_NAME,
    ) -> None:
        client.collections[COLLECTION_NAME] = VectorParams(
            size=size, distance=Distance.COSINE
        )
        payload = {"paper_id": "old"}
        if embedding_model is not None:
            payload["embedding_model"] = embedding_model
        client.points["seed"] = PointStruct(id="seed", vector=[0.0], payload=payload)

    # --- fallback and provenance -------------------------------------------------

    def test_abstract_only_without_parsed_cache(
        self,
        tmp_path: Path,
        fake_qdrant: FakeQdrant,
        ingest_model: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        self._write_paper(tmp_path)

        with caplog.at_level(logging.WARNING):
            count = self._ingest(
                tmp_path, fake_qdrant, ingest_model, parsed_dir=tmp_path / "none"
            )

        assert count == 1
        (point,) = fake_qdrant.points.values()
        assert point.id == point_id("paper", 0)
        assert point.payload == {
            "title": "Test Paper",
            "authors": ["Author A"],
            "year": 2024,
            "doi": "10.test/123",
            "paper_id": "paper",
            "section": "Abstract",
            "page": 1,
            "chunk_index": 0,
            "chunk_type": "text",
            "text": "A test abstract about AI in medicine.",
            "embedding_model": MODEL_NAME,
        }
        # One warning for the paper, one summary for the whole corpus (R4).
        messages = [r.getMessage() for r in caplog.records]
        assert any("paper: no usable parsed cache" in m for m in messages)
        assert any("No parsed papers found" in m for m in messages)

    def test_parsed_dir_none_means_abstract_only(
        self, tmp_path: Path, fake_qdrant: FakeQdrant, ingest_model: MagicMock
    ) -> None:
        self._write_paper(tmp_path)

        assert self._ingest(tmp_path, fake_qdrant, ingest_model) == 1

    def test_indexes_full_text_with_section_and_page(
        self,
        tmp_path: Path,
        fake_qdrant: FakeQdrant,
        ingest_model: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        papers, parsed = tmp_path / "papers", tmp_path / "parsed"
        papers.mkdir()
        self._write_paper(papers)
        self._write_parsed(parsed)

        with caplog.at_level(logging.WARNING):
            count = self._ingest(papers, fake_qdrant, ingest_model, parsed_dir=parsed)

        payloads = [p.payload or {} for p in fake_qdrant.points.values()]
        assert count == len(payloads) > 1
        by_index = sorted(payloads, key=lambda p: p["chunk_index"])
        assert [p["chunk_index"] for p in by_index] == list(range(count))
        assert by_index[0]["section"] == "Abstract"
        sections = {p["section"] for p in by_index}
        assert {"Introduction", "Methods > Datasets", "Results"} <= sections
        assert not any("ref0" in p["text"] for p in by_index)  # References dropped
        datasets = next(p for p in by_index if p["section"] == "Methods > Datasets")
        assert datasets["page"] == 3
        assert all(p["paper_id"] == "paper" for p in by_index)
        assert all(p["embedding_model"] == MODEL_NAME for p in by_index)
        assert not caplog.records  # full text found: no fallback warnings

    def test_embeds_the_header_plus_body_text(
        self, tmp_path: Path, fake_qdrant: FakeQdrant, ingest_model: MagicMock
    ) -> None:
        self._write_paper(tmp_path)

        self._ingest(tmp_path, fake_qdrant, ingest_model)

        (texts,) = ingest_model.encode.call_args.args
        assert texts == ["Test Paper\nAbstract\nA test abstract about AI in medicine."]

    def test_corrupt_cache_falls_back_to_abstract(
        self, tmp_path: Path, fake_qdrant: FakeQdrant, ingest_model: MagicMock
    ) -> None:
        papers, parsed = tmp_path / "papers", tmp_path / "parsed"
        papers.mkdir()
        parsed.mkdir()
        self._write_paper(papers)
        (parsed / "paper.json").write_text("{not json")

        assert self._ingest(papers, fake_qdrant, ingest_model, parsed_dir=parsed) == 1

    # --- identity, idempotency ---------------------------------------------------

    def test_rerun_is_idempotent(
        self, tmp_path: Path, fake_qdrant: FakeQdrant, ingest_model: MagicMock
    ) -> None:
        papers, parsed = tmp_path / "papers", tmp_path / "parsed"
        papers.mkdir()
        self._write_paper(papers)
        self._write_parsed(parsed)

        first = self._ingest(papers, fake_qdrant, ingest_model, parsed_dir=parsed)
        ids = set(fake_qdrant.points)
        second = self._ingest(papers, fake_qdrant, ingest_model, parsed_dir=parsed)

        assert first == second
        assert set(fake_qdrant.points) == ids
        assert ids == {point_id("paper", i) for i in range(first)}

    def test_shrinking_paper_leaves_no_stale_points(
        self, tmp_path: Path, fake_qdrant: FakeQdrant, ingest_model: MagicMock
    ) -> None:
        papers, parsed = tmp_path / "papers", tmp_path / "parsed"
        papers.mkdir()
        self._write_paper(papers)
        self._write_paper(papers, "other.json", title="Other")
        self._write_parsed(parsed)
        self._ingest(papers, fake_qdrant, ingest_model, parsed_dir=parsed)
        long_ids = fake_qdrant.ids_of("paper")
        other_ids = fake_qdrant.ids_of("other")

        (parsed / "paper.json").unlink()  # "paper" now has only its abstract
        self._ingest(papers, fake_qdrant, ingest_model, parsed_dir=parsed)

        assert len(long_ids) > 2
        assert fake_qdrant.ids_of("paper") == {point_id("paper", 0)}
        assert fake_qdrant.ids_of("other") == other_ids

    def test_deletes_a_papers_points_before_upserting_them(
        self, tmp_path: Path, fake_qdrant: FakeQdrant, ingest_model: MagicMock
    ) -> None:
        self._write_paper(tmp_path)

        self._ingest(tmp_path, fake_qdrant, ingest_model)

        assert fake_qdrant.calls == ["create_collection", "delete_points", "upsert"]

    def test_point_ids_are_stable_uuid5(self) -> None:
        assert point_id("paper", 3) == point_id("paper", 3)
        assert point_id("paper", 3) != point_id("paper", 4)
        assert point_id("paper", 3) != point_id("other", 3)
        assert uuid.UUID(point_id("paper", 3)).version == 5

    # --- collection management ---------------------------------------------------

    def test_collection_size_comes_from_the_model(
        self, tmp_path: Path, fake_qdrant: FakeQdrant, ingest_model: MagicMock
    ) -> None:
        self._write_paper(tmp_path)
        ingest_model.get_sentence_embedding_dimension.return_value = 768
        ingest_model.encode.side_effect = lambda texts, **_: np.zeros(
            (len(texts), 768), dtype=np.float32
        )

        self._ingest(tmp_path, fake_qdrant, ingest_model)

        assert fake_qdrant.collections[COLLECTION_NAME].size == 768

    def test_model_without_dimension_raises(
        self, tmp_path: Path, fake_qdrant: FakeQdrant, ingest_model: MagicMock
    ) -> None:
        self._write_paper(tmp_path)
        ingest_model.get_sentence_embedding_dimension.return_value = None

        with pytest.raises(ValueError, match="vector dimension"):
            self._ingest(tmp_path, fake_qdrant, ingest_model)

        assert fake_qdrant.calls == []

    def test_skips_paper_with_empty_abstract(
        self, tmp_path: Path, fake_qdrant: FakeQdrant, ingest_model: MagicMock
    ) -> None:
        self._write_paper(tmp_path, abstract="")

        assert self._ingest(tmp_path, fake_qdrant, ingest_model) == 0
        assert not fake_qdrant.points

    def test_skips_invalid_json(
        self, tmp_path: Path, fake_qdrant: FakeQdrant, ingest_model: MagicMock
    ) -> None:
        (tmp_path / "bad.json").write_text("not valid json{")

        assert self._ingest(tmp_path, fake_qdrant, ingest_model) == 0

    def test_overwrite_drops_existing_collection(
        self, tmp_path: Path, fake_qdrant: FakeQdrant, ingest_model: MagicMock
    ) -> None:
        self._write_paper(tmp_path)
        self._seed_collection(fake_qdrant)

        self._ingest(tmp_path, fake_qdrant, ingest_model, overwrite=True)

        assert fake_qdrant.calls[:2] == ["delete_collection", "create_collection"]
        assert "seed" not in fake_qdrant.points

    def test_matching_collection_is_kept(
        self, tmp_path: Path, fake_qdrant: FakeQdrant, ingest_model: MagicMock
    ) -> None:
        self._write_paper(tmp_path)
        self._seed_collection(fake_qdrant)

        self._ingest(tmp_path, fake_qdrant, ingest_model)

        assert "delete_collection" not in fake_qdrant.calls
        assert "create_collection" not in fake_qdrant.calls
        assert "seed" in fake_qdrant.points

    @pytest.mark.parametrize(
        ("size", "embedding_model", "reason"),
        [
            (768, MODEL_NAME, "vector size"),
            (TEST_VECTOR_DIM, "other/embedder", "embedding model"),
            # A pre-Phase-8 collection: same size, but no embedding_model on its points.
            (TEST_VECTOR_DIM, None, "embedding model"),
        ],
    )
    def test_mismatching_collection_is_recreated(
        self,
        tmp_path: Path,
        fake_qdrant: FakeQdrant,
        ingest_model: MagicMock,
        caplog: pytest.LogCaptureFixture,
        size: int,
        embedding_model: str | None,
        reason: str,
    ) -> None:
        self._write_paper(tmp_path)
        self._seed_collection(fake_qdrant, size=size, embedding_model=embedding_model)

        with caplog.at_level(logging.WARNING):
            self._ingest(tmp_path, fake_qdrant, ingest_model)

        assert fake_qdrant.calls[:2] == ["delete_collection", "create_collection"]
        assert "seed" not in fake_qdrant.points
        assert fake_qdrant.collections[COLLECTION_NAME].size == TEST_VECTOR_DIM
        assert any(
            "Recreating collection" in r.getMessage() and reason in r.getMessage()
            for r in caplog.records
        )

    def test_empty_matching_collection_is_kept(
        self, tmp_path: Path, fake_qdrant: FakeQdrant, ingest_model: MagicMock
    ) -> None:
        self._write_paper(tmp_path)
        self._seed_collection(fake_qdrant)
        fake_qdrant.points.clear()

        self._ingest(tmp_path, fake_qdrant, ingest_model)

        assert "delete_collection" not in fake_qdrant.calls


# ---------------------------------------------------------------------------
# TestSearch
# ---------------------------------------------------------------------------


class TestSearch:
    def _make_point(
        self, payload: dict, score: float = 0.85, point_id: int = 0
    ) -> MagicMock:
        point = MagicMock()
        point.payload = payload
        point.score = score
        point.id = point_id
        return point

    async def test_returns_citations_and_confidence(
        self, async_qdrant: AsyncMock
    ) -> None:
        async_qdrant.query_points.return_value.points = [
            self._make_point(
                {
                    "title": "VLM Paper",
                    "authors": ["B"],
                    "year": 2024,
                    "doi": "10.x/1",
                    "text": "VLMs are powerful.",
                },
                score=0.91,
            )
        ]
        vector = np.zeros(TEST_VECTOR_DIM, dtype=np.float32)

        citations, confidence = await search(vector, async_qdrant, top_k=5)

        assert len(citations) == 1
        assert citations[0].title == "VLM Paper"
        assert citations[0].year == 2024
        assert confidence == 0.91

    async def test_reads_chunk_provenance_fields(self, async_qdrant: AsyncMock) -> None:
        async_qdrant.query_points.return_value.points = [
            self._make_point(
                {
                    "title": "Full-text Paper",
                    "authors": ["A"],
                    "year": 2024,
                    "doi": None,
                    "text": "We use two datasets.",
                    "paper_id": "full_text_paper",
                    "section": "Methods > Datasets",
                    "page": 4,
                    "chunk_index": 7,
                }
            )
        ]

        citations, _ = await search(
            np.zeros(TEST_VECTOR_DIM, dtype=np.float32), async_qdrant
        )

        assert citations[0].paper_id == "full_text_paper"
        assert citations[0].section == "Methods > Datasets"
        assert citations[0].page == 4
        assert citations[0].chunk_index == 7

    async def test_legacy_payload_without_provenance_fields(
        self, async_qdrant: AsyncMock
    ) -> None:
        async_qdrant.query_points.return_value.points = [
            self._make_point(
                {
                    "title": "Old Paper",
                    "authors": ["A"],
                    "year": 2023,
                    "doi": "10.x/2",
                    "text": "Abstract text.",
                }
            )
        ]

        citations, _ = await search(
            np.zeros(TEST_VECTOR_DIM, dtype=np.float32), async_qdrant
        )

        assert len(citations) == 1
        assert citations[0].excerpt == "Abstract text."
        assert citations[0].paper_id is None
        assert citations[0].section is None
        assert citations[0].page is None
        assert citations[0].chunk_index is None

    async def test_malformed_provenance_fields_become_none_not_skipped(
        self, async_qdrant: AsyncMock
    ) -> None:
        async_qdrant.query_points.return_value.points = [
            self._make_point(
                {
                    "title": "Odd Paper",
                    "authors": [],
                    "year": 2024,
                    "text": "Body.",
                    "paper_id": 123,
                    "section": ["not", "a", "string"],
                    "page": "four",
                    "chunk_index": True,
                }
            )
        ]

        citations, _ = await search(
            np.zeros(TEST_VECTOR_DIM, dtype=np.float32), async_qdrant
        )

        assert len(citations) == 1
        assert citations[0].paper_id is None
        assert citations[0].section is None
        assert citations[0].page is None
        assert citations[0].chunk_index is None

    async def test_default_threshold_matches_query_request(
        self, async_qdrant: AsyncMock
    ) -> None:
        await search(np.zeros(TEST_VECTOR_DIM, dtype=np.float32), async_qdrant)

        kwargs = async_qdrant.query_points.call_args.kwargs
        assert kwargs["score_threshold"] == DEFAULT_SCORE_THRESHOLD
        assert QueryRequest(question="q").score_threshold == DEFAULT_SCORE_THRESHOLD

    async def test_empty_results_return_zero_confidence(
        self, async_qdrant: AsyncMock
    ) -> None:
        async_qdrant.query_points.return_value.points = []
        vector = np.zeros(TEST_VECTOR_DIM, dtype=np.float32)

        citations, confidence = await search(vector, async_qdrant)

        assert citations == []
        assert confidence == 0.0

    async def test_skips_malformed_payload(self, async_qdrant: AsyncMock) -> None:
        async_qdrant.query_points.return_value.points = [
            self._make_point(
                {"title": "Bad", "authors": [], "year": "not-an-int", "text": "X"}
            ),
        ]
        vector = np.zeros(TEST_VECTOR_DIM, dtype=np.float32)

        citations, _ = await search(vector, async_qdrant)
        assert citations == []


# ---------------------------------------------------------------------------
# TestQueryRoute
# ---------------------------------------------------------------------------


class TestQueryRoute:
    def _make_point(self, payload: dict, score: float = 0.88) -> MagicMock:
        point = MagicMock()
        point.payload = payload
        point.score = score
        point.id = 1
        return point

    async def test_returns_citations_when_results_found(
        self, http_client: AsyncClient, async_qdrant: AsyncMock
    ) -> None:
        async_qdrant.query_points.return_value.points = [
            self._make_point(
                {
                    "title": "Radiology AI",
                    "authors": ["C"],
                    "year": 2023,
                    "doi": None,
                    "text": "AI in radiology.",
                },
                score=0.88,
            )
        ]

        response = await http_client.post(
            "/query", json={"question": "How does AI perform in radiology?"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["grounded"] is True
        assert data["debug"]["route"] == "retrieval"
        assert len(data["citations"]) == 1
        assert isinstance(data["answer"], str) and data["answer"]
        assert data["confidence"] == 0.88

    async def test_returns_ungrounded_when_no_results(
        self, http_client: AsyncClient
    ) -> None:
        response = await http_client.post(
            "/query", json={"question": "What is quantum computing?"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["grounded"] is False
        assert data["citations"] == []
        assert data["confidence"] == 0.0

    async def test_unsafe_query_still_refused(self, http_client: AsyncClient) -> None:
        response = await http_client.post(
            "/query", json={"question": "Should I take aspirin for my symptoms?"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["grounded"] is False
        assert data["debug"]["route"] == "unsafe_medical_advice"


# --- Embedding model consistency (API startup check) ---------------------------------


class _ScrollClient:
    def __init__(self, points: list[object] | Exception) -> None:
        self._points = points

    async def scroll(self, *args: object, **kwargs: object):
        if isinstance(self._points, Exception):
            raise self._points
        return self._points, None


class _Point:
    def __init__(self, payload: dict[str, object] | None) -> None:
        self.payload = payload


async def test_embedding_mismatch_names_both_models() -> None:
    from app.retrieval.consistency import embedding_model_mismatch

    client = _ScrollClient([_Point({"embedding_model": "all-MiniLM-L6-v2"})])

    message = await embedding_model_mismatch(client, "BAAI/bge-small-en-v1.5")  # type: ignore[arg-type]

    assert message is not None
    assert "all-MiniLM-L6-v2" in message
    assert "BAAI/bge-small-en-v1.5" in message


async def test_embedding_match_empty_collection_and_unreachable_qdrant_are_silent() -> (
    None
):
    from app.retrieval.consistency import embedding_model_mismatch

    for points in (
        [_Point({"embedding_model": "m"})],
        [],
        RuntimeError("connection refused"),
    ):
        assert await embedding_model_mismatch(_ScrollClient(points), "m") is None  # type: ignore[arg-type]


async def test_point_without_embedding_model_counts_as_mismatch() -> None:
    from app.retrieval.consistency import embedding_model_mismatch

    assert await embedding_model_mismatch(_ScrollClient([_Point({})]), "m") is not None  # type: ignore[arg-type]
