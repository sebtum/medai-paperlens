from unittest.mock import AsyncMock

import httpx
import numpy as np
import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient

from app.llm.ollama import get_ollama_client
from app.main import app
from app.retrieval.client import get_async_client
from app.retrieval.embedding import get_embedding_provider
from app.retrieval.ingestion import VECTOR_DIM


@pytest.fixture
async def mocked_app():
    """App with all external dependencies stubbed out and lifespan running."""
    mock_client = AsyncMock()
    mock_client.query_points.return_value.points = []

    mock_provider = AsyncMock()
    mock_provider.embed.return_value = np.zeros(VECTOR_DIM, dtype=np.float32)

    mock_ollama = AsyncMock()
    mock_ollama.generate.return_value = "Mocked answer."

    app.dependency_overrides[get_async_client] = lambda: mock_client
    app.dependency_overrides[get_embedding_provider] = lambda: mock_provider
    app.dependency_overrides[get_ollama_client] = lambda: mock_ollama

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
