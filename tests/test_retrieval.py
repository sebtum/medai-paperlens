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
    Document,
    FilterSelector,
    FusionQuery,
    Modifier,
    PointStruct,
    SparseVectorParams,
    VectorParams,
)

from app.evaluation.metrics import RankedHit
from app.llm.factory import get_llm_provider
from app.main import app
from app.models.query import DEFAULT_SCORE_THRESHOLD, QueryRequest
from app.retrieval.client import get_async_client
from app.retrieval.embedding import (
    BGE_QUERY_PREFIX,
    SentenceTransformerProvider,
    embed,
    get_embedding_provider,
    get_model_name,
    get_query_prefix,
)
from app.retrieval.ingestion import COLLECTION_NAME, ingest, point_id
from app.retrieval.rerank import get_reranker
from app.retrieval.search import search, search_hits, search_hits_hybrid

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
    app.dependency_overrides[get_reranker] = lambda: None

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


class TestQueryPrefix:
    """ADR-0025: the prefix is put in front of queries only, never of documents."""

    async def test_provider_prepends_the_prefix_to_queries(
        self, mock_model: MagicMock
    ) -> None:
        provider = SentenceTransformerProvider(mock_model, query_prefix="Q: ")
        await provider.embed("what is a CNN?")
        mock_model.encode.assert_called_once_with("Q: what is a CNN?")

    async def test_provider_without_prefix_embeds_the_text_as_is(
        self, mock_model: MagicMock
    ) -> None:
        await SentenceTransformerProvider(mock_model).embed("what is a CNN?")
        mock_model.encode.assert_called_once_with("what is a CNN?")

    def test_ingestion_embeds_documents_without_the_prefix(
        self, tmp_path: Path, fake_qdrant: FakeQdrant, ingest_model: MagicMock
    ) -> None:
        # ingestion takes the raw model, not the provider, so no prefix can reach it
        paper = {"title": "T", "authors": ["A"], "year": 2024, "abstract": "Imaging."}
        (tmp_path / "paper.json").write_text(json.dumps(paper))
        ingest(
            tmp_path,
            fake_qdrant,  # type: ignore[arg-type]
            ingest_model,
            model_name=MODEL_NAME,
            tokenizer=WhitespaceTokenizer(max_tokens=200),
        )
        texts = [t for call in ingest_model.encode.call_args_list for t in call.args[0]]
        assert texts
        assert not any(BGE_QUERY_PREFIX in t for t in texts)

    def test_unset_variable_and_bge_model_gives_the_bge_instruction(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("EMBEDDING_QUERY_PREFIX", raising=False)
        assert get_query_prefix("BAAI/bge-small-en-v1.5") == BGE_QUERY_PREFIX
        assert get_query_prefix("BAAI/bge-base-en-v1.5") == BGE_QUERY_PREFIX

    def test_default_model_name_is_read_from_the_environment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("EMBEDDING_QUERY_PREFIX", raising=False)
        monkeypatch.setenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        assert get_query_prefix() == ""
        monkeypatch.delenv("EMBEDDING_MODEL")
        assert get_query_prefix() == BGE_QUERY_PREFIX

    def test_empty_variable_switches_the_prefix_off(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("EMBEDDING_QUERY_PREFIX", "")
        assert get_query_prefix("BAAI/bge-small-en-v1.5") == ""

    def test_set_variable_is_used_as_is(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EMBEDDING_QUERY_PREFIX", "query: ")
        assert get_query_prefix("BAAI/bge-small-en-v1.5") == "query: "
        assert get_query_prefix("some/other-model") == "query: "

    def test_non_bge_model_gets_no_prefix_by_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("EMBEDDING_QUERY_PREFIX", raising=False)
        assert get_query_prefix("sentence-transformers/all-MiniLM-L6-v2") == ""


# ---------------------------------------------------------------------------
# TestIngest
# ---------------------------------------------------------------------------


class FakeQdrant:
    """In-memory stand-in for the parts of `QdrantClient` that ingestion uses."""

    def __init__(self) -> None:
        self.collections: dict[str, Any] = {}
        self.sparse: dict[str, dict[str, SparseVectorParams] | None] = {}
        self.points: dict[str, PointStruct] = {}
        self.calls: list[str] = []

    def get_collections(self) -> CollectionsResponse:
        return CollectionsResponse(
            collections=[CollectionDescription(name=n) for n in self.collections]
        )

    def create_collection(
        self,
        collection_name: str,
        vectors_config: Any,
        sparse_vectors_config: dict[str, SparseVectorParams] | None = None,
    ) -> None:
        self.calls.append("create_collection")
        self.collections[collection_name] = vectors_config
        self.sparse[collection_name] = sparse_vectors_config

    def delete_collection(self, collection_name: str) -> None:
        self.calls.append("delete_collection")
        self.collections.pop(collection_name)
        self.sparse.pop(collection_name, None)
        self.points.clear()

    def get_collection(self, collection_name: str) -> SimpleNamespace:
        vectors = self.collections[collection_name]
        params = SimpleNamespace(
            vectors=vectors, sparse_vectors=self.sparse.get(collection_name)
        )
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


class RecordingQdrant(FakeQdrant):
    """FakeQdrant that records which collection each call addressed."""

    def __init__(self) -> None:
        super().__init__()
        self.touched: list[tuple[str, str]] = []

    def create_collection(
        self,
        collection_name: str,
        vectors_config: Any,
        sparse_vectors_config: dict[str, SparseVectorParams] | None = None,
    ) -> None:
        self.touched.append(("create_collection", collection_name))
        super().create_collection(
            collection_name, vectors_config, sparse_vectors_config
        )

    def delete_collection(self, collection_name: str) -> None:
        self.touched.append(("delete_collection", collection_name))
        # FakeQdrant keeps one flat point store; keep it so only the call is recorded.
        points = self.points
        super().delete_collection(collection_name)
        self.points = points

    def get_collection(self, collection_name: str) -> SimpleNamespace:
        self.touched.append(("get_collection", collection_name))
        return super().get_collection(collection_name)

    def scroll(
        self, collection_name: str, limit: int, **kwargs: Any
    ) -> tuple[list[SimpleNamespace], None]:
        self.touched.append(("scroll", collection_name))
        return super().scroll(collection_name, limit, **kwargs)

    def delete(self, collection_name: str, points_selector: FilterSelector) -> None:
        self.touched.append(("delete", collection_name))
        super().delete(collection_name, points_selector)

    def upsert(self, collection_name: str, points: list[PointStruct]) -> None:
        self.touched.append(("upsert", collection_name))
        super().upsert(collection_name, points)


class TestIngestCollectionName:
    EVAL = "eval_test-embedder"

    def _write_paper(self, tmp_path: Path) -> None:
        paper = {
            "title": "Test Paper",
            "authors": ["Author A"],
            "year": 2024,
            "doi": "10.test/123",
            "abstract": "A test abstract about AI in medicine.",
        }
        (tmp_path / "paper.json").write_text(json.dumps(paper))

    def _ingest(
        self,
        tmp_path: Path,
        client: RecordingQdrant,
        model: MagicMock,
        **kwargs: Any,
    ) -> int:
        return ingest(
            tmp_path,
            client,  # type: ignore[arg-type]
            model,
            model_name=MODEL_NAME,
            tokenizer=WhitespaceTokenizer(max_tokens=200),
            **kwargs,
        )

    def test_default_collection_is_papers(
        self, tmp_path: Path, ingest_model: MagicMock
    ) -> None:
        self._write_paper(tmp_path)
        client = RecordingQdrant()

        self._ingest(tmp_path, client, ingest_model)

        assert set(client.collections) == {COLLECTION_NAME}
        assert {name for _, name in client.touched} == {COLLECTION_NAME}

    def test_non_default_collection_leaves_papers_untouched(
        self, tmp_path: Path, ingest_model: MagicMock
    ) -> None:
        self._write_paper(tmp_path)
        client = RecordingQdrant()
        client.collections[COLLECTION_NAME] = VectorParams(
            size=TEST_VECTOR_DIM, distance=Distance.COSINE
        )
        client.points["app"] = PointStruct(
            id="app",
            vector=[0.0],
            payload={"paper_id": "paper", "embedding_model": "app/embedder"},
        )

        count = self._ingest(
            tmp_path, client, ingest_model, collection_name=self.EVAL, overwrite=True
        )

        assert count == 1
        assert set(client.collections) == {COLLECTION_NAME, self.EVAL}
        # Every call addressed the eval collection; `papers` was never read or written.
        assert client.touched
        assert {name for _, name in client.touched} == {self.EVAL}
        assert client.collections[COLLECTION_NAME].size == TEST_VECTOR_DIM

    def test_stale_check_and_recreation_apply_to_the_named_collection(
        self, tmp_path: Path, ingest_model: MagicMock
    ) -> None:
        self._write_paper(tmp_path)
        client = RecordingQdrant()
        for name in (COLLECTION_NAME, self.EVAL):
            client.collections[name] = VectorParams(
                size=TEST_VECTOR_DIM, distance=Distance.COSINE
            )
        client.points["old"] = PointStruct(
            id="old",
            vector=[0.0],
            payload={"paper_id": "old", "embedding_model": "other/embedder"},
        )

        self._ingest(tmp_path, client, ingest_model, collection_name=self.EVAL)

        assert ("get_collection", self.EVAL) in client.touched
        assert ("delete_collection", self.EVAL) in client.touched
        assert ("create_collection", self.EVAL) in client.touched
        assert COLLECTION_NAME in client.collections
        assert all(name == self.EVAL for _, name in client.touched)


class TestIngestHybrid:
    """`ingest(..., hybrid=True)` (Phase 10, eval only): dense plus BM25 layout."""

    EVAL = "eval_hybrid"

    def _write_paper(self, tmp_path: Path) -> None:
        paper = {
            "title": "Test Paper",
            "authors": ["Author A"],
            "year": 2024,
            "doi": "10.test/123",
            "abstract": "A test abstract about AI in medicine.",
        }
        (tmp_path / "paper.json").write_text(json.dumps(paper))

    def _ingest(
        self,
        tmp_path: Path,
        client: RecordingQdrant,
        model: MagicMock,
        **kwargs: Any,
    ) -> int:
        kwargs.setdefault("collection_name", self.EVAL)
        return ingest(
            tmp_path,
            client,  # type: ignore[arg-type]
            model,
            model_name=MODEL_NAME,
            tokenizer=WhitespaceTokenizer(max_tokens=200),
            **kwargs,
        )

    def test_collection_has_a_named_dense_vector_and_a_sparse_bm25_vector(
        self, tmp_path: Path, ingest_model: MagicMock
    ) -> None:
        self._write_paper(tmp_path)
        client = RecordingQdrant()

        self._ingest(tmp_path, client, ingest_model, hybrid=True)

        dense = client.collections[self.EVAL]
        assert set(dense) == {"dense"}
        assert dense["dense"].size == TEST_VECTOR_DIM
        assert dense["dense"].distance == Distance.COSINE
        sparse = client.sparse[self.EVAL]
        assert sparse is not None and set(sparse) == {"bm25"}
        assert sparse["bm25"].modifier == Modifier.IDF

    def test_points_carry_the_dense_vector_and_a_bm25_document_of_embed_text(
        self, tmp_path: Path, ingest_model: MagicMock
    ) -> None:
        self._write_paper(tmp_path)
        client = RecordingQdrant()

        count = self._ingest(tmp_path, client, ingest_model, hybrid=True)

        embedded = [
            text for call in ingest_model.encode.call_args_list for text in call[0][0]
        ]
        assert count == len(client.points) == len(embedded) == 1
        (point,) = client.points.values()
        assert isinstance(point.vector, dict)
        assert set(point.vector) == {"dense", "bm25"}
        assert point.vector["dense"] == [0.0] * TEST_VECTOR_DIM
        document = point.vector["bm25"]
        assert isinstance(document, Document)
        assert document.model == "qdrant/bm25"
        assert document.text == embedded[0]  # the same text the dense side embedded
        assert document.options is None  # default options, also at query time

    def test_payloads_and_ids_equal_the_plain_layout(
        self, tmp_path: Path, ingest_model: MagicMock
    ) -> None:
        self._write_paper(tmp_path)
        plain, hybrid = RecordingQdrant(), RecordingQdrant()

        self._ingest(tmp_path, plain, ingest_model, collection_name="eval_plain")
        self._ingest(tmp_path, hybrid, ingest_model, hybrid=True)

        assert set(plain.points) == set(hybrid.points)
        for pid, point in plain.points.items():
            assert hybrid.points[pid].payload == point.payload
        assert isinstance(next(iter(plain.points.values())).vector, list)

    def test_rechunking_never_touches_the_app_collection(
        self, tmp_path: Path, ingest_model: MagicMock
    ) -> None:
        self._write_paper(tmp_path)
        client = RecordingQdrant()

        for kwargs in ({"chunk_max_tokens": 64}, {"merge_peers": False}):
            with pytest.raises(ValueError, match="re-chunking is eval only"):
                self._ingest(
                    tmp_path,
                    client,
                    ingest_model,
                    collection_name=COLLECTION_NAME,
                    **kwargs,
                )
        assert client.touched == []

    def test_rechunking_options_reach_an_eval_collection(
        self, tmp_path: Path, ingest_model: MagicMock
    ) -> None:
        self._write_paper(tmp_path)
        client = RecordingQdrant()

        count = self._ingest(
            tmp_path,
            client,
            ingest_model,
            collection_name="eval_chunk256_merge",
            chunk_max_tokens=64,
            merge_peers=False,
        )

        assert count == 1
        assert {name for _, name in client.touched} == {"eval_chunk256_merge"}

    def test_papers_is_never_written_in_the_hybrid_layout(
        self, tmp_path: Path, ingest_model: MagicMock
    ) -> None:
        self._write_paper(tmp_path)
        client = RecordingQdrant()

        with pytest.raises(ValueError, match="eval only"):
            self._ingest(
                tmp_path,
                client,
                ingest_model,
                hybrid=True,
                collection_name=COLLECTION_NAME,
            )
        assert client.touched == []

    def test_only_the_eval_collection_is_touched(
        self, tmp_path: Path, ingest_model: MagicMock
    ) -> None:
        self._write_paper(tmp_path)
        client = RecordingQdrant()
        client.collections[COLLECTION_NAME] = VectorParams(
            size=TEST_VECTOR_DIM, distance=Distance.COSINE
        )

        self._ingest(tmp_path, client, ingest_model, hybrid=True, overwrite=True)

        assert {name for _, name in client.touched} == {self.EVAL}
        assert isinstance(client.collections[COLLECTION_NAME], VectorParams)

    def test_a_matching_hybrid_collection_is_reused_not_recreated(
        self, tmp_path: Path, ingest_model: MagicMock
    ) -> None:
        self._write_paper(tmp_path)
        client = RecordingQdrant()
        self._ingest(tmp_path, client, ingest_model, hybrid=True)
        client.touched.clear()

        self._ingest(tmp_path, client, ingest_model, hybrid=True)

        assert ("delete_collection", self.EVAL) not in client.touched
        assert ("create_collection", self.EVAL) not in client.touched
        assert len(client.points) == 1

    def test_another_embedding_model_recreates_the_hybrid_collection(
        self, tmp_path: Path, ingest_model: MagicMock
    ) -> None:
        self._write_paper(tmp_path)
        client = RecordingQdrant()
        self._ingest(tmp_path, client, ingest_model, hybrid=True)
        for point in client.points.values():
            assert point.payload is not None
            point.payload["embedding_model"] = "other/embedder"
        client.touched.clear()

        self._ingest(tmp_path, client, ingest_model, hybrid=True)

        assert ("delete_collection", self.EVAL) in client.touched
        assert ("create_collection", self.EVAL) in client.touched
        assert client.sparse[self.EVAL] is not None

    def test_a_plain_layout_collection_is_recreated_as_hybrid(
        self, tmp_path: Path, ingest_model: MagicMock
    ) -> None:
        self._write_paper(tmp_path)
        client = RecordingQdrant()
        client.collections[self.EVAL] = VectorParams(
            size=TEST_VECTOR_DIM, distance=Distance.COSINE
        )

        self._ingest(tmp_path, client, ingest_model, hybrid=True)

        assert ("delete_collection", self.EVAL) in client.touched
        assert set(client.collections[self.EVAL]) == {"dense"}

    def test_a_hybrid_collection_without_the_sparse_vector_is_recreated(
        self, tmp_path: Path, ingest_model: MagicMock
    ) -> None:
        self._write_paper(tmp_path)
        client = RecordingQdrant()
        client.collections[self.EVAL] = {
            "dense": VectorParams(size=TEST_VECTOR_DIM, distance=Distance.COSINE)
        }

        self._ingest(tmp_path, client, ingest_model, hybrid=True)

        assert ("delete_collection", self.EVAL) in client.touched
        assert client.sparse[self.EVAL] is not None

    def test_a_plain_ingest_recreates_a_hybrid_collection(
        self, tmp_path: Path, ingest_model: MagicMock
    ) -> None:
        self._write_paper(tmp_path)
        client = RecordingQdrant()
        self._ingest(tmp_path, client, ingest_model, hybrid=True)
        client.touched.clear()

        self._ingest(tmp_path, client, ingest_model)

        assert ("delete_collection", self.EVAL) in client.touched
        assert isinstance(client.collections[self.EVAL], VectorParams)
        assert isinstance(next(iter(client.points.values())).vector, list)


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


class TestSearchCollectionName:
    async def test_search_defaults_to_papers_and_accepts_a_collection(
        self, async_qdrant: AsyncMock
    ) -> None:
        vector = np.zeros(TEST_VECTOR_DIM, dtype=np.float32)

        await search(vector, async_qdrant)
        kwargs = async_qdrant.query_points.call_args.kwargs
        assert kwargs["collection_name"] == COLLECTION_NAME

        await search(vector, async_qdrant, collection_name="eval_x")
        kwargs = async_qdrant.query_points.call_args.kwargs
        assert kwargs["collection_name"] == "eval_x"


class TestSearchHits:
    def _point(self, payload: dict, score: float, point_id: int = 0) -> MagicMock:
        point = MagicMock()
        point.payload = payload
        point.score = score
        point.id = point_id
        return point

    def _payload(self, **overrides: object) -> dict:
        return {
            "title": "T",
            "authors": [],
            "year": 2024,
            "text": "body",
            "paper_id": "p1",
            "section": "Methods > Datasets",
            "chunk_index": 3,
            "chunk_type": "table",
            **overrides,
        }

    async def test_returns_scores_and_chunk_type_in_rank_order(
        self, async_qdrant: AsyncMock
    ) -> None:
        async_qdrant.query_points.return_value.points = [
            self._point(self._payload(), 0.91),
            self._point(
                self._payload(paper_id="p2", chunk_index=0, chunk_type="text"), 0.42, 1
            ),
        ]

        hits = await search_hits(
            np.zeros(TEST_VECTOR_DIM, dtype=np.float32),
            async_qdrant,
            top_k=20,
            score_threshold=0.0,
        )

        assert hits == [
            RankedHit("p1", "Methods > Datasets", 3, "table", 0.91),
            RankedHit("p2", "Methods > Datasets", 0, "text", 0.42),
        ]

    async def test_passes_query_parameters_through(
        self, async_qdrant: AsyncMock
    ) -> None:
        vector = np.zeros(TEST_VECTOR_DIM, dtype=np.float32)

        await search_hits(
            vector,
            async_qdrant,
            top_k=20,
            score_threshold=0.0,
            collection_name="eval_x",
        )

        kwargs = async_qdrant.query_points.call_args.kwargs
        assert kwargs["collection_name"] == "eval_x"
        assert kwargs["limit"] == 20
        assert kwargs["score_threshold"] == 0.0

    async def test_default_collection_is_papers(self, async_qdrant: AsyncMock) -> None:
        await search_hits(
            np.zeros(TEST_VECTOR_DIM, dtype=np.float32),
            async_qdrant,
            top_k=5,
            score_threshold=0.6,
        )

        kwargs = async_qdrant.query_points.call_args.kwargs
        assert kwargs["collection_name"] == COLLECTION_NAME

    async def test_missing_chunk_type_counts_as_text(
        self, async_qdrant: AsyncMock
    ) -> None:
        payload = self._payload()
        del payload["chunk_type"]
        async_qdrant.query_points.return_value.points = [self._point(payload, 0.7)]

        (hit,) = await search_hits(
            np.zeros(TEST_VECTOR_DIM, dtype=np.float32),
            async_qdrant,
            top_k=5,
            score_threshold=0.0,
        )

        assert hit.chunk_type == "text"

    async def test_skips_points_without_chunk_identity(
        self, async_qdrant: AsyncMock
    ) -> None:
        async_qdrant.query_points.return_value.points = [
            self._point(self._payload(paper_id=123), 0.9, 0),
            self._point(self._payload(section=None), 0.8, 1),
            self._point(self._payload(chunk_index=True), 0.7, 2),
            self._point({"title": "legacy", "text": "x"}, 0.6, 3),
            self._point(self._payload(chunk_index=5), 0.5, 4),
        ]

        hits = await search_hits(
            np.zeros(TEST_VECTOR_DIM, dtype=np.float32),
            async_qdrant,
            top_k=5,
            score_threshold=0.0,
        )

        assert [h.chunk_index for h in hits] == [5]

    async def test_empty_results(self, async_qdrant: AsyncMock) -> None:
        hits = await search_hits(
            np.zeros(TEST_VECTOR_DIM, dtype=np.float32),
            async_qdrant,
            top_k=5,
            score_threshold=0.0,
        )

        assert hits == []


# ---------------------------------------------------------------------------
# TestQueryRoute
# ---------------------------------------------------------------------------


class TestSearchHitsHybrid:
    def _point(self, payload: dict, score: float, point_id: int = 0) -> MagicMock:
        point = MagicMock()
        point.payload = payload
        point.score = score
        point.id = point_id
        return point

    def _payload(self, **overrides: object) -> dict:
        return {
            "paper_id": "p1",
            "section": "Methods",
            "chunk_index": 3,
            "chunk_type": "table",
            **overrides,
        }

    async def _search(self, client: AsyncMock, **kwargs: Any) -> list[RankedHit]:
        params: dict[str, Any] = {
            "top_k": 20,
            "dense_threshold": 0.6,
            "collection_name": "eval_hybrid",
        }
        params.update(kwargs)
        return await search_hits_hybrid(
            np.ones(TEST_VECTOR_DIM, dtype=np.float32),
            "bare query text",
            client,
            **params,
        )

    async def test_one_query_with_a_dense_and_a_bm25_prefetch_fused_by_rrf(
        self, async_qdrant: AsyncMock
    ) -> None:
        await self._search(async_qdrant)

        async_qdrant.query_points.assert_awaited_once()
        kwargs = async_qdrant.query_points.call_args.kwargs
        assert kwargs["collection_name"] == "eval_hybrid"
        assert kwargs["limit"] == 20
        assert kwargs["with_payload"] is True
        assert kwargs["query"] == FusionQuery(fusion="rrf")
        dense, bm25 = kwargs["prefetch"]
        assert dense.using == "dense"
        assert dense.limit == 20
        assert dense.score_threshold == 0.6
        assert dense.query == [1.0] * TEST_VECTOR_DIM
        assert bm25.using == "bm25"
        assert bm25.limit == 20
        assert bm25.score_threshold is None  # no cosine cut on the keyword list
        assert bm25.query == Document(text="bare query text", model="qdrant/bm25")
        assert bm25.query.options is None

    async def test_prefetch_limit_defaults_to_top_k_and_can_differ(
        self, async_qdrant: AsyncMock
    ) -> None:
        await self._search(async_qdrant, top_k=5)
        first = async_qdrant.query_points.call_args.kwargs
        assert [p.limit for p in first["prefetch"]] == [5, 5]

        await self._search(async_qdrant, top_k=5, prefetch_limit=20)
        second = async_qdrant.query_points.call_args.kwargs
        assert second["limit"] == 5
        assert [p.limit for p in second["prefetch"]] == [20, 20]

    async def test_has_no_default_collection(self, async_qdrant: AsyncMock) -> None:
        with pytest.raises(TypeError):
            await search_hits_hybrid(  # type: ignore[call-arg]
                np.zeros(TEST_VECTOR_DIM, dtype=np.float32),
                "q",
                async_qdrant,
                top_k=20,
                dense_threshold=0.6,
            )

    async def test_returns_ranked_hits_with_chunk_type_and_the_fused_score(
        self, async_qdrant: AsyncMock
    ) -> None:
        async_qdrant.query_points.return_value.points = [
            self._point(self._payload(), 0.0328),
            self._point(self._payload(paper_id="p2", chunk_type=None), 0.0161, 1),
        ]

        hits = await self._search(async_qdrant)

        assert hits == [
            RankedHit("p1", "Methods", 3, "table", 0.0328),
            RankedHit("p2", "Methods", 3, "text", 0.0161),
        ]

    async def test_a_point_without_chunk_identity_is_skipped(
        self, async_qdrant: AsyncMock
    ) -> None:
        async_qdrant.query_points.return_value.points = [
            self._point(self._payload(section=None), 0.03),
            self._point(self._payload(), 0.02, 1),
        ]

        hits = await self._search(async_qdrant)

        assert [h.paper_id for h in hits] == ["p1"]


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
