from unittest.mock import MagicMock

import httpx
import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.retrieval.client import get_client
from app.retrieval.embedding import get_model
from app.retrieval.ingestion import VECTOR_DIM


@pytest.fixture
def mocked_app():
    """Provide dependency-overridden app with empty Qdrant results."""
    mock_client = MagicMock()
    mock_client.query_points.return_value.points = []
    mock_model = MagicMock()
    mock_model.encode.return_value = np.zeros(VECTOR_DIM, dtype=np.float32)
    app.dependency_overrides[get_client] = lambda: mock_client
    app.dependency_overrides[get_model] = lambda: mock_model
    yield app
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


async def test_query_empty_question_returns_422() -> None:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post("/query", json={"question": ""})
    assert response.status_code == 422


async def test_query_whitespace_question_returns_422() -> None:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post("/query", json={"question": "   "})
    assert response.status_code == 422


async def test_query_missing_field_returns_422() -> None:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post("/query", json={})
    assert response.status_code == 422


async def test_query_unsafe_personal_advice_returns_refusal() -> None:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
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
