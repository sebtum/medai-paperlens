from unittest.mock import AsyncMock, MagicMock

import httpx
import numpy as np
import pytest
from asgi_lifespan import LifespanManager
from conftest import TEST_VECTOR_DIM, FakeLlm
from httpx import ASGITransport, AsyncClient

from app.llm.factory import get_llm_provider
from app.main import app
from app.retrieval.client import get_async_client
from app.retrieval.embedding import get_embedding_provider


@pytest.fixture
async def mocked_app():
    """App with all external dependencies stubbed out and lifespan running."""
    mock_client = AsyncMock()
    mock_client.query_points.return_value.points = []

    mock_provider = AsyncMock()
    mock_provider.embed.return_value = np.zeros(TEST_VECTOR_DIM, dtype=np.float32)

    fake_llm = FakeLlm(answer="Mocked answer.")

    app.dependency_overrides[get_async_client] = lambda: mock_client
    app.dependency_overrides[get_embedding_provider] = lambda: mock_provider
    app.dependency_overrides[get_llm_provider] = lambda: fake_llm

    async with LifespanManager(app) as manager:
        yield manager.app

    app.dependency_overrides.clear()


async def test_query_valid_returns_retrieval_route(mocked_app) -> None:
    async with AsyncClient(
        transport=ASGITransport(app=mocked_app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/query",
            json={"question": "What is the accuracy of AI in detecting pneumonia?"},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["debug"]["route"] == "retrieval"
    assert isinstance(body["citations"], list)
    assert isinstance(body["confidence"], float | int)


async def test_query_citations_include_provenance_and_legacy_payloads() -> None:
    def point(payload: dict, score: float) -> MagicMock:
        p = MagicMock()
        p.payload = payload
        p.score = score
        p.id = 0
        return p

    mock_client = AsyncMock()
    mock_client.query_points.return_value.points = [
        point(
            {
                "title": "Chunked",
                "authors": [],
                "year": 2024,
                "text": "Body.",
                "paper_id": "chunked",
                "section": "Results",
                "page": 5,
                "chunk_index": 3,
            },
            0.9,
        ),
        point({"title": "Legacy", "authors": [], "year": 2023, "text": "Abs."}, 0.8),
    ]
    mock_provider = AsyncMock()
    mock_provider.embed.return_value = np.zeros(TEST_VECTOR_DIM, dtype=np.float32)
    app.dependency_overrides[get_async_client] = lambda: mock_client
    app.dependency_overrides[get_embedding_provider] = lambda: mock_provider
    app.dependency_overrides[get_llm_provider] = lambda: FakeLlm(answer="Answer.")

    try:
        async with (
            LifespanManager(app) as manager,
            AsyncClient(
                transport=ASGITransport(app=manager.app), base_url="http://test"
            ) as client,
        ):
            response = await client.post("/query", json={"question": "How big?"})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    chunked, legacy = response.json()["citations"]
    assert (chunked["paper_id"], chunked["section"]) == ("chunked", "Results")
    assert (chunked["page"], chunked["chunk_index"]) == (5, 3)
    assert legacy["title"] == "Legacy"
    assert legacy["paper_id"] is None
    assert legacy["page"] is None


async def test_query_empty_question_returns_422(managed_app) -> None:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=managed_app), base_url="http://test"
    ) as client:
        response = await client.post("/query", json={"question": ""})
    assert response.status_code == 422


async def test_query_whitespace_question_returns_422(managed_app) -> None:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=managed_app), base_url="http://test"
    ) as client:
        response = await client.post("/query", json={"question": "   "})
    assert response.status_code == 422


async def test_query_missing_field_returns_422(managed_app) -> None:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=managed_app), base_url="http://test"
    ) as client:
        response = await client.post("/query", json={})
    assert response.status_code == 422


async def test_query_unsafe_personal_advice_returns_refusal(managed_app) -> None:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=managed_app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/query",
            json={"question": "Should I take aspirin for my symptoms?"},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["grounded"] is False
    assert body["debug"]["route"] == "unsafe_medical_advice"
    assert "medical advice" in body["answer"].lower()


async def test_query_research_question_about_diagnosis_is_allowed(mocked_app) -> None:
    """A research question about AI diagnosing conditions is not personal advice."""
    async with AsyncClient(
        transport=ASGITransport(app=mocked_app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/query",
            json={"question": "How do AI models perform in cancer diagnosis studies?"},
        )
    assert response.status_code == 200
    assert response.json()["debug"]["route"] == "retrieval"
