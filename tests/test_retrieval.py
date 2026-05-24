import json
from collections.abc import AsyncGenerator
from unittest.mock import MagicMock

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient
from qdrant_client.models import CollectionDescription, CollectionsResponse

from app.main import app
from app.retrieval.client import get_client
from app.retrieval.embedding import embed, get_model
from app.retrieval.ingestion import COLLECTION_NAME, VECTOR_DIM, ingest
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
def mock_model() -> MagicMock:
    model = MagicMock()
    model.encode.return_value = np.zeros(VECTOR_DIM, dtype=np.float32)
    return model


@pytest.fixture
async def http_client(
    mock_qdrant: MagicMock, mock_model: MagicMock
) -> AsyncGenerator[AsyncClient]:
    app.dependency_overrides[get_client] = lambda: mock_qdrant
    app.dependency_overrides[get_model] = lambda: mock_model
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# TestEmbed
# ---------------------------------------------------------------------------


class TestEmbed:
    def test_returns_ndarray_from_model_encode(self, mock_model: MagicMock) -> None:
        result = embed("test query", mock_model)
        assert isinstance(result, np.ndarray)
        assert result.shape == (VECTOR_DIM,)
        mock_model.encode.assert_called_once_with("test query")


# ---------------------------------------------------------------------------
# TestIngest
# ---------------------------------------------------------------------------


class TestIngest:
    def _write_paper(
        self, tmp_path, name: str = "paper.json", **overrides: object
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

    def test_creates_collection_and_upserts(
        self, tmp_path, mock_qdrant: MagicMock
    ) -> None:
        self._write_paper(tmp_path)
        mock_model = MagicMock()
        mock_model.encode.return_value = np.zeros((1, VECTOR_DIM), dtype=np.float32)

        count = ingest(tmp_path, mock_qdrant, mock_model)

        assert count == 1
        mock_qdrant.create_collection.assert_called_once()
        mock_qdrant.upsert.assert_called_once()

    def test_skips_paper_with_empty_abstract(
        self, tmp_path, mock_qdrant: MagicMock
    ) -> None:
        self._write_paper(tmp_path, abstract="")

        count = ingest(tmp_path, mock_qdrant, MagicMock())

        assert count == 0
        mock_qdrant.upsert.assert_not_called()

    def test_skips_invalid_json(self, tmp_path, mock_qdrant: MagicMock) -> None:
        (tmp_path / "bad.json").write_text("not valid json{")

        count = ingest(tmp_path, mock_qdrant, MagicMock())
        assert count == 0

    def test_overwrite_drops_existing_collection(
        self, tmp_path, mock_qdrant: MagicMock
    ) -> None:
        self._write_paper(tmp_path)
        mock_qdrant.get_collections.return_value = CollectionsResponse(
            collections=[CollectionDescription(name=COLLECTION_NAME)]
        )
        mock_model = MagicMock()
        mock_model.encode.return_value = np.zeros((1, VECTOR_DIM), dtype=np.float32)

        ingest(tmp_path, mock_qdrant, mock_model, overwrite=True)

        mock_qdrant.delete_collection.assert_called_once_with(COLLECTION_NAME)
        mock_qdrant.create_collection.assert_called_once()

    def test_no_overwrite_keeps_existing_collection(
        self, tmp_path, mock_qdrant: MagicMock
    ) -> None:
        self._write_paper(tmp_path)
        mock_qdrant.get_collections.return_value = CollectionsResponse(
            collections=[CollectionDescription(name=COLLECTION_NAME)]
        )
        mock_model = MagicMock()
        mock_model.encode.return_value = np.zeros((1, VECTOR_DIM), dtype=np.float32)

        ingest(tmp_path, mock_qdrant, mock_model, overwrite=False)

        mock_qdrant.delete_collection.assert_not_called()
        mock_qdrant.create_collection.assert_not_called()


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

    def test_returns_citations_and_confidence(self, mock_qdrant: MagicMock) -> None:
        mock_qdrant.query_points.return_value.points = [
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
        vector = np.zeros(VECTOR_DIM, dtype=np.float32)

        citations, confidence = search(vector, mock_qdrant, top_k=5)

        assert len(citations) == 1
        assert citations[0].title == "VLM Paper"
        assert citations[0].year == 2024
        assert confidence == 0.91

    def test_empty_results_return_zero_confidence(
        self, mock_qdrant: MagicMock
    ) -> None:
        mock_qdrant.query_points.return_value.points = []
        vector = np.zeros(VECTOR_DIM, dtype=np.float32)

        citations, confidence = search(vector, mock_qdrant)

        assert citations == []
        assert confidence == 0.0

    def test_skips_malformed_payload(self, mock_qdrant: MagicMock) -> None:
        mock_qdrant.query_points.return_value.points = [
            self._make_point(
                {"title": "Bad", "authors": [], "year": "not-an-int", "text": "X"}
            ),
        ]
        vector = np.zeros(VECTOR_DIM, dtype=np.float32)

        citations, _ = search(vector, mock_qdrant)
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
        self, http_client: AsyncClient, mock_qdrant: MagicMock
    ) -> None:
        mock_qdrant.query_points.return_value.points = [
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
        assert data["answer"] == "AI in radiology."
        assert data["confidence"] == 0.88

    async def test_returns_ungrounded_when_no_results(
        self, http_client: AsyncClient, mock_qdrant: MagicMock
    ) -> None:
        mock_qdrant.query_points.return_value.points = []

        response = await http_client.post(
            "/query", json={"question": "What is quantum computing?"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["grounded"] is False
        assert data["citations"] == []
        assert data["confidence"] == 0.0

    async def test_unsafe_query_still_refused(
        self, http_client: AsyncClient
    ) -> None:
        response = await http_client.post(
            "/query", json={"question": "Should I take aspirin for my symptoms?"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["grounded"] is False
        assert data["debug"]["route"] == "unsafe_medical_advice"
