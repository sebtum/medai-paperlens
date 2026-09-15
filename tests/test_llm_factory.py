from unittest.mock import AsyncMock, MagicMock

import pytest

from app.llm.factory import make_llm_provider
from app.llm.gemini import GeminiClient
from app.llm.ollama import OllamaClient


def test_default_provider_is_ollama(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    provider = make_llm_provider()
    assert isinstance(provider, OllamaClient)


def test_gemini_provider_without_api_key_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        make_llm_provider()


def test_gemini_provider_with_api_key_returns_gemini_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    provider = make_llm_provider()
    assert isinstance(provider, GeminiClient)


def test_unknown_provider_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
        make_llm_provider()


async def test_think_false_lands_at_payload_top_level_not_options() -> None:
    """Regression guard: `think` must be a top-level Ollama payload field, not
    nested inside `options` - that was the latency bug the think plumbing fix
    addresses (see plan_llm_model_decision.md).
    """
    client = OllamaClient("http://localhost:11434", "qwen3:0.6b", think=False)
    mock_http = AsyncMock()
    mock_response = MagicMock()
    mock_response.json.return_value = {"response": "ok"}
    mock_http.post.return_value = mock_response
    client._http = mock_http

    await client.generate("hello")

    _, kwargs = mock_http.post.call_args
    payload = kwargs["json"]
    assert payload["think"] is False
    assert "think" not in payload["options"]
