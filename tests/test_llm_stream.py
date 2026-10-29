import json

import httpx
import pytest

from app.llm.gemini import GeminiClient
from app.llm.ollama import OllamaClient


def _ndjson(*chunks: dict) -> bytes:
    return ("\n".join(json.dumps(c) for c in chunks) + "\n").encode()


def _client(handler) -> OllamaClient:
    client = OllamaClient("http://ollama", "qwen3:0.6b", think=False)
    client._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return client


async def test_ollama_stream_generate_yields_response_chunks():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            content=_ndjson(
                {"response": "Hel", "done": False},
                {"response": "lo", "done": False},
                {"response": "", "done": True},
            ),
        )

    client = _client(handler)
    tokens = [t async for t in client.stream("hi", temperature=0.1)]

    assert tokens == ["Hel", "lo"]
    assert seen["url"] == "http://ollama/api/generate"
    assert seen["payload"]["stream"] is True
    assert seen["payload"]["prompt"] == "hi"
    assert seen["payload"]["options"] == {"temperature": 0.1}
    assert seen["payload"]["think"] is False


async def test_ollama_stream_chat_yields_message_content():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return httpx.Response(
            200,
            content=_ndjson(
                {"message": {"role": "assistant", "content": "A"}},
                {"message": {"role": "assistant", "content": "B"}},
                {"message": {"role": "assistant", "content": ""}, "done": True},
            ),
        )

    client = _client(handler)
    tokens = [t async for t in client.stream([{"role": "user", "content": "q"}])]

    assert tokens == ["A", "B"]
    assert seen["url"] == "http://ollama/api/chat"


async def test_ollama_stream_never_yields_thinking():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=_ndjson(
                {"response": "", "thinking": "hmm"},
                {"response": "answer"},
                {"message": {"content": "", "thinking": "x"}},
            ),
        )

    client = _client(handler)
    assert [t async for t in client.stream("hi")] == ["answer"]


async def test_ollama_stream_http_error_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, content=b"boom")

    client = _client(handler)
    with pytest.raises(httpx.HTTPStatusError):
        _ = [t async for t in client.stream("hi")]


async def test_ollama_stream_requires_context_manager():
    client = OllamaClient("http://ollama", "m")
    with pytest.raises(RuntimeError):
        _ = [t async for t in client.stream("hi")]


class _FakeChunk:
    def __init__(self, text: str | None) -> None:
        self.text = text


class _FakeGeminiModels:
    def __init__(self, texts: list[str | None]) -> None:
        self._texts = texts
        self.calls: list[dict] = []

    async def generate_content_stream(self, **kwargs):
        self.calls.append(kwargs)

        async def gen():
            for t in self._texts:
                yield _FakeChunk(t)

        return gen()


class _FakeGenaiClient:
    def __init__(self, models: _FakeGeminiModels) -> None:
        self.aio = type("Aio", (), {"models": models})()


async def test_gemini_stream_yields_chunk_texts_and_skips_empty():
    models = _FakeGeminiModels(["Hel", None, "", "lo"])
    client = GeminiClient("gemini-test", api_key="k")
    client._client = _FakeGenaiClient(models)

    tokens = [t async for t in client.stream([{"role": "user", "content": "q"}])]

    assert tokens == ["Hel", "lo"]
    assert models.calls == [{"model": "gemini-test", "contents": "q"}]


async def test_gemini_stream_requires_context_manager():
    client = GeminiClient("gemini-test", api_key="k")
    with pytest.raises(RuntimeError):
        _ = [t async for t in client.stream("hi")]


def test_fake_llm_satisfies_llm_provider():
    from conftest import FakeLlm

    from app.llm.base import LlmProvider

    assert isinstance(FakeLlm(), LlmProvider)


async def test_no_langsmith_client_is_built_when_tracing_unset(monkeypatch):
    import langsmith

    def _boom(*args, **kwargs):
        raise AssertionError("langsmith.Client constructed while tracing is off")

    monkeypatch.setattr(langsmith.Client, "__init__", _boom)

    def handler(request: httpx.Request) -> httpx.Response:
        if json.loads(request.content)["stream"]:
            return httpx.Response(200, content=_ndjson({"response": "ok"}))
        return httpx.Response(200, json={"response": "ok"})

    client = _client(handler)
    assert await client.generate("hi") == "ok"
    assert [t async for t in client.stream("hi")] == ["ok"]

    gemini = GeminiClient("gemini-test", api_key="k")
    gemini._client = _FakeGenaiClient(_FakeGeminiModels(["a", "b"]))
    assert [t async for t in gemini.stream("hi")] == ["a", "b"]
