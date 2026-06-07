from unittest.mock import AsyncMock

import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient

from app.llm.ollama import get_ollama_client
from app.main import app


def _make_mock_ollama(warm: bool) -> AsyncMock:
    mock = AsyncMock()
    mock.is_model_warm.return_value = warm
    mock._model = "qwen3.5:4b"
    return mock


@pytest.fixture
async def warm_app():
    app.dependency_overrides[get_ollama_client] = lambda: _make_mock_ollama(warm=True)
    async with LifespanManager(app) as manager:
        yield manager.app
    app.dependency_overrides.clear()


@pytest.fixture
async def cold_app():
    app.dependency_overrides[get_ollama_client] = lambda: _make_mock_ollama(warm=False)
    async with LifespanManager(app) as manager:
        yield manager.app
    app.dependency_overrides.clear()


async def test_model_status_warm(warm_app) -> None:
    async with AsyncClient(
        transport=ASGITransport(app=warm_app), base_url="http://test"
    ) as client:
        response = await client.get("/model/status")
        assert response.status_code == 200
        body = response.json()
        assert body["warm"] is True
        assert body["model"] == "qwen3.5:4b"
        assert body["estimated_warmup_seconds"] is None


async def test_model_status_cold(cold_app) -> None:
    async with AsyncClient(
        transport=ASGITransport(app=cold_app), base_url="http://test"
    ) as client:
        response = await client.get("/model/status")
        assert response.status_code == 200
        body = response.json()
        assert body["warm"] is False
        assert body["model"] == "qwen3.5:4b"
        assert isinstance(body["estimated_warmup_seconds"], int)
        assert body["estimated_warmup_seconds"] > 0
