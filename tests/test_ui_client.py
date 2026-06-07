from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from ui.app import (
    ModelStatus,
    QueryResult,
    _build_spinner_text,
    call_query_api,
    check_backend_health,
    check_model_status,
)

_VALID_RESULT: QueryResult = {
    "answer": "Vision transformers show strong performance on medical imaging tasks.",
    "citations": [],
    "confidence": 0.85,
    "grounded": True,
    "debug": {"route": "retrieval"},
}

_REFUSAL_RESULT: QueryResult = {
    "answer": "Literature summaries only — no medical advice.",
    "citations": [],
    "confidence": 0.0,
    "grounded": False,
    "debug": {"route": "unsafe_medical_advice"},
}

_COLD_STATUS: ModelStatus = {
    "warm": False,
    "model": "qwen3.5:4b",
    "estimated_warmup_seconds": 120,
}

_WARM_STATUS: ModelStatus = {
    "warm": True,
    "model": "qwen3.5:4b",
    "estimated_warmup_seconds": None,
}


def _make_mock_client(response_json: dict, status_code: int = 200) -> AsyncMock:
    mock_response = MagicMock()
    mock_response.json.return_value = response_json
    mock_response.status_code = status_code
    mock_response.raise_for_status = MagicMock()
    if status_code >= 400:
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "error", request=MagicMock(), response=mock_response
        )

    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.post.return_value = mock_response
    mock_client.get.return_value = mock_response
    return mock_client


# --- call_query_api ---

async def test_call_query_api_returns_result_on_200() -> None:
    mock = _make_mock_client(_VALID_RESULT)
    with patch("ui.app.httpx.AsyncClient", return_value=mock):
        result, error = await call_query_api("What is AI accuracy for pneumonia?")
    assert result is not None
    assert error is None
    assert result["debug"]["route"] == "retrieval"
    assert result["confidence"] == pytest.approx(0.85)


async def test_call_query_api_returns_none_on_connect_error() -> None:
    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.post.side_effect = httpx.ConnectError("refused")
    with patch("ui.app.httpx.AsyncClient", return_value=mock_client):
        result, error = await call_query_api("test question")
    assert result is None
    assert error == "connect_error"


async def test_call_query_api_returns_none_on_timeout() -> None:
    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.post.side_effect = httpx.TimeoutException("timed out")
    with patch("ui.app.httpx.AsyncClient", return_value=mock_client):
        result, error = await call_query_api("test question")
    assert result is None
    assert error == "timeout"


async def test_call_query_api_returns_none_on_http_error() -> None:
    mock = _make_mock_client({}, status_code=500)
    with patch("ui.app.httpx.AsyncClient", return_value=mock):
        result, error = await call_query_api("test question")
    assert result is None
    assert error == "http_error"


async def test_call_query_api_refusal_returns_result_not_none() -> None:
    mock = _make_mock_client(_REFUSAL_RESULT)
    with patch("ui.app.httpx.AsyncClient", return_value=mock):
        result, error = await call_query_api("Should I take aspirin for my symptoms?")
    assert result is not None
    assert error is None
    assert result["debug"]["route"] == "unsafe_medical_advice"
    assert result["grounded"] is False


# --- check_backend_health ---

async def test_check_backend_health_returns_true_on_200() -> None:
    mock = _make_mock_client({"status": "ok"}, 200)
    with patch("ui.app.httpx.AsyncClient", return_value=mock):
        assert await check_backend_health() is True


async def test_check_backend_health_returns_false_on_connect_error() -> None:
    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.get.side_effect = httpx.ConnectError("refused")
    with patch("ui.app.httpx.AsyncClient", return_value=mock_client):
        assert await check_backend_health() is False


# --- check_model_status ---

async def test_check_model_status_returns_cold_status() -> None:
    mock = _make_mock_client(_COLD_STATUS)
    with patch("ui.app.httpx.AsyncClient", return_value=mock):
        status = await check_model_status()
    assert status is not None
    assert status["warm"] is False
    assert status["estimated_warmup_seconds"] == 120


async def test_check_model_status_returns_warm_status() -> None:
    mock = _make_mock_client(_WARM_STATUS)
    with patch("ui.app.httpx.AsyncClient", return_value=mock):
        status = await check_model_status()
    assert status is not None
    assert status["warm"] is True
    assert status["estimated_warmup_seconds"] is None


async def test_check_model_status_returns_none_on_connect_error() -> None:
    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.get.side_effect = httpx.ConnectError("refused")
    with patch("ui.app.httpx.AsyncClient", return_value=mock_client):
        assert await check_model_status() is None


async def test_check_model_status_returns_none_on_http_error() -> None:
    mock = _make_mock_client({}, status_code=500)
    with patch("ui.app.httpx.AsyncClient", return_value=mock):
        assert await check_model_status() is None


# --- _build_spinner_text ---

def test_build_spinner_text_warm_model() -> None:
    assert _build_spinner_text(_WARM_STATUS) == "Querying…"


def test_build_spinner_text_none_status() -> None:
    assert _build_spinner_text(None) == "Querying…"


def test_build_spinner_text_cold_with_estimate() -> None:
    text = _build_spinner_text(_COLD_STATUS)
    assert "qwen3.5:4b" in text
    assert "2 min" in text


def test_build_spinner_text_cold_no_estimate() -> None:
    status: ModelStatus = {
        "warm": False,
        "model": "qwen3.5:4b",
        "estimated_warmup_seconds": None,
    }
    text = _build_spinner_text(status)
    assert "qwen3.5:4b" in text
    assert "min" not in text
