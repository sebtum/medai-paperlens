import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import numpy as np
from asgi_lifespan import LifespanManager
from conftest import FakeLlm
from httpx import ASGITransport, AsyncClient, Response

from app.llm.factory import get_llm_provider
from app.main import app
from app.retrieval.client import get_async_client
from app.retrieval.embedding import get_embedding_provider
from app.retrieval.ingestion import VECTOR_DIM
from app.workflow.generate import _ERROR_ANSWER

_QUESTION = "How does AI perform in radiology?"


def _point(score: float = 0.88) -> MagicMock:
    point = MagicMock()
    point.payload = {
        "title": "Radiology AI",
        "authors": ["C"],
        "year": 2023,
        "doi": None,
        "text": "AI in radiology.",
    }
    point.score = score
    point.id = 1
    return point


def _parse_sse(body: str) -> list[tuple[str, dict]]:
    events = []
    for block in body.split("\n\n"):
        event, data = None, []
        for line in block.splitlines():
            if line.startswith("event:"):
                event = line.removeprefix("event:").strip()
            elif line.startswith("data:"):
                data.append(line.removeprefix("data:").strip())
        if event is not None:
            events.append((event, json.loads("\n".join(data))))
    return events


async def _post(
    path: str,
    fake_llm: FakeLlm,
    *,
    points: list | None = None,
    qdrant_error: Exception | None = None,
    body: dict | None = None,
) -> Response:
    qdrant = AsyncMock()
    qdrant.query_points.return_value.points = points or []
    if qdrant_error is not None:
        qdrant.query_points.side_effect = qdrant_error
    embedder = AsyncMock()
    embedder.embed.return_value = np.zeros(VECTOR_DIM, dtype=np.float32)

    app.dependency_overrides[get_async_client] = lambda: qdrant
    app.dependency_overrides[get_embedding_provider] = lambda: embedder
    app.dependency_overrides[get_llm_provider] = lambda: fake_llm
    try:
        async with (
            LifespanManager(app) as manager,
            AsyncClient(
                transport=ASGITransport(app=manager.app), base_url="http://test"
            ) as client,
        ):
            return await client.post(path, json=body or {"question": _QUESTION})
    finally:
        app.dependency_overrides.clear()


async def test_stream_headers() -> None:
    response = await _post("/query/stream", FakeLlm(), points=[_point()])
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["x-accel-buffering"] == "no"


async def test_stream_event_order() -> None:
    fake = FakeLlm(tokens=["AI ", "works."])
    response = await _post("/query/stream", fake, points=[_point()])
    events = _parse_sse(response.text)
    names = [e for e, _ in events]

    assert names == [
        "step_start",
        "step_end",  # classify
        "step_start",
        "step_end",  # rewrite
        "step_start",
        "step_end",
        "citations",  # retrieve
        "step_start",
        "token",
        "token",
        "step_end",  # generate
        "step_start",
        "step_end",  # validate
        "final",
    ]
    nodes = [d["node"] for e, d in events if e == "step_start"]
    assert nodes == ["classify", "rewrite", "retrieve", "generate", "validate"]
    assert all(d["elapsed_ms"] >= 0 for e, d in events if e == "step_end")
    assert [d["text"] for e, d in events if e == "token"] == ["AI ", "works."]

    citations = next(d for e, d in events if e == "citations")
    assert len(citations["citations"]) == 1
    assert citations["confidence"] == 0.88


async def test_stream_unsafe_query_has_no_tokens() -> None:
    fake = FakeLlm()
    response = await _post(
        "/query/stream",
        fake,
        body={"question": "Should I take my medication for my condition?"},
    )
    events = _parse_sse(response.text)

    assert "token" not in [e for e, _ in events]
    assert fake.calls == []
    name, final = events[-1]
    assert name == "final"
    assert final["grounded"] is False
    assert final["debug"]["route"] == "unsafe_medical_advice"


async def test_stream_llm_error_sends_error_then_final() -> None:
    fake = FakeLlm(error=Exception("Ollama unavailable"))
    response = await _post("/query/stream", fake, points=[_point()])
    events = _parse_sse(response.text)
    names = [e for e, _ in events]

    assert "error" in names
    assert names.index("error") < names.index("final")
    assert names[-1] == "final"
    assert events[-1][1]["answer"] == _ERROR_ANSWER


async def test_stream_unexpected_exception_sends_error_then_final() -> None:
    response = await _post(
        "/query/stream", FakeLlm(), qdrant_error=RuntimeError("qdrant exploded")
    )
    events = _parse_sse(response.text)

    names = [e for e, _ in events]
    assert names[-2:] == ["error", "final"]
    assert "citations" not in names
    final = events[-1][1]
    assert final["answer"] == _ERROR_ANSWER
    assert final["grounded"] is False


async def test_stream_final_equals_query_response() -> None:
    llm_args = {"answer": "AI works."}
    stream = await _post("/query/stream", FakeLlm(**llm_args), points=[_point()])
    plain = await _post("/query", FakeLlm(**llm_args), points=[_point()])

    final = _parse_sse(stream.text)[-1][1]
    assert final == plain.json()


async def test_stream_empty_question_returns_422() -> None:
    response = await _post("/query/stream", FakeLlm(), body={"question": "   "})
    assert response.status_code == 422


class _BlockingLlm(FakeLlm):
    """Streams one token, then blocks until cancelled; ``cancelled`` marks cleanup."""

    def __init__(self) -> None:
        super().__init__(answer="unused")
        self.cancelled = asyncio.Event()

    async def stream(self, prompt, **kwargs):
        self.calls.append(prompt)
        try:
            yield "first "
            await asyncio.Event().wait()
        finally:
            self.cancelled.set()


async def test_client_disconnect_cancels_graph_and_llm() -> None:
    blocking = _BlockingLlm()
    qdrant = AsyncMock()
    qdrant.query_points.return_value.points = [_point()]
    embedder = AsyncMock()
    embedder.embed.return_value = np.zeros(VECTOR_DIM, dtype=np.float32)
    app.dependency_overrides[get_async_client] = lambda: qdrant
    app.dependency_overrides[get_embedding_provider] = lambda: embedder
    app.dependency_overrides[get_llm_provider] = lambda: blocking

    body = json.dumps({"question": _QUESTION}).encode()
    token_sent = asyncio.Event()
    body_delivered = False

    async def receive() -> dict:
        nonlocal body_delivered
        if not body_delivered:
            body_delivered = True
            return {"type": "http.request", "body": body, "more_body": False}
        await token_sent.wait()
        return {"type": "http.disconnect"}

    async def send(message: dict) -> None:
        if b"event: token" in message.get("body", b""):
            token_sent.set()

    # No ``spec_version``: Starlette takes its listen_for_disconnect path, the
    # same one uvicorn uses for streaming responses.
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/query/stream",
        "raw_path": b"/query/stream",
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"host", b"test"),
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode()),
        ],
        "client": ("127.0.0.1", 1234),
        "server": ("test", 80),
    }
    try:
        async with LifespanManager(app) as manager:
            await asyncio.wait_for(manager.app(scope, receive, send), timeout=5)
            await asyncio.wait_for(blocking.cancelled.wait(), timeout=2)
    finally:
        app.dependency_overrides.clear()

    assert token_sent.is_set()
    assert blocking.cancelled.is_set()
