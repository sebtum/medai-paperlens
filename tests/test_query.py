import httpx
import pytest

from app.main import app


@pytest.mark.asyncio
async def test_query_valid_returns_stub() -> None:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/query",
            json={"question": "What is the accuracy of AI in detecting pneumonia?"},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["grounded"] is False
    assert body["debug"]["route"] == "stub"
    assert isinstance(body["citations"], list)
    assert isinstance(body["confidence"], float | int)


@pytest.mark.asyncio
async def test_query_empty_question_returns_422() -> None:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post("/query", json={"question": ""})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_query_whitespace_question_returns_422() -> None:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post("/query", json={"question": "   "})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_query_missing_field_returns_422() -> None:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post("/query", json={})
    assert response.status_code == 422


@pytest.mark.asyncio
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


@pytest.mark.asyncio
async def test_query_research_question_about_diagnosis_is_allowed() -> None:
    """A research question about AI diagnosing conditions is not personal advice."""
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/query",
            json={"question": "How do AI models perform in cancer diagnosis studies?"},
        )
    assert response.status_code == 200
    assert response.json()["debug"]["route"] == "stub"
