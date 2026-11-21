import asyncio
import logging
import time
from collections.abc import AsyncIterator
from typing import Annotated, Any

from fastapi import APIRouter, Depends
from fastapi.sse import EventSourceResponse, ServerSentEvent
from qdrant_client import AsyncQdrantClient

from app.llm.base import LlmProvider
from app.llm.factory import get_llm_provider
from app.models.query import (
    CitationsEvent,
    ErrorEvent,
    QueryRequest,
    QueryResponse,
    StepEndEvent,
    StepStartEvent,
    TokenEvent,
)
from app.retrieval.client import get_async_client
from app.retrieval.embedding import EmbeddingProvider, get_embedding_provider
from app.retrieval.rerank import Reranker, get_reranker
from app.workflow.generate import _ERROR_ANSWER
from app.workflow.graph import build_workflow
from app.workflow.state import WorkflowState

logger = logging.getLogger(__name__)

router = APIRouter()


def _initial_state(request: QueryRequest) -> WorkflowState:
    return {
        "question": request.question,
        "rewritten": request.question,
        "is_unsafe": False,
        "citations": [],
        "confidence": 0.0,
        "answer": "",
        "grounded": False,
        "route": "",
        "score_threshold": request.score_threshold,
        "rerank_top_score": None,
        "rerank_fallback": False,
    }


def _to_response(state: WorkflowState, reranker: Reranker | None) -> QueryResponse:
    return QueryResponse(
        answer=state["answer"],
        citations=state["citations"],
        confidence=state["confidence"],
        grounded=state["grounded"],
        debug={
            "route": state["route"],
            "rewritten": state["rewritten"],
            # No model ran when the step fell back, so none is named.
            "rerank_model": (
                reranker.model_name
                if reranker is not None and not state["rerank_fallback"]
                else None
            ),
            "rerank_top_score": state["rerank_top_score"],
            "rerank_fallback": state["rerank_fallback"],
        },
    )


@router.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    client: Annotated[AsyncQdrantClient, Depends(get_async_client)],
    provider: Annotated[EmbeddingProvider, Depends(get_embedding_provider)],
    llm: Annotated[LlmProvider, Depends(get_llm_provider)],
    reranker: Annotated[Reranker | None, Depends(get_reranker)],
) -> QueryResponse:
    workflow = build_workflow(client, provider, llm, reranker)
    state = await workflow.ainvoke(_initial_state(request))
    return _to_response(state, reranker)


@router.post("/query/stream", response_class=EventSourceResponse)
async def query_stream(
    request: QueryRequest,
    client: Annotated[AsyncQdrantClient, Depends(get_async_client)],
    provider: Annotated[EmbeddingProvider, Depends(get_embedding_provider)],
    llm: Annotated[LlmProvider, Depends(get_llm_provider)],
    reranker: Annotated[Reranker | None, Depends(get_reranker)],
) -> AsyncIterator[ServerSentEvent]:
    """Run the same workflow as ``/query`` and stream progress as SSE (ADR-0017)."""
    workflow = build_workflow(client, provider, llm, reranker)
    state = _initial_state(request)
    started: dict[str, float] = {}
    try:
        async for mode, chunk in _iterate_in_own_task(
            workflow.astream(state, stream_mode=["tasks", "custom", "values"])
        ):
            if mode == "values":
                state = chunk
            elif mode == "tasks":
                event = _task_event(chunk, started)
                if event is not None:
                    yield event
                if (
                    chunk.get("name") == "rerank"
                    and "result" in chunk
                    and chunk.get("error") is None
                ):
                    # The final five come from rerank. Rerank leaves `confidence`
                    # alone, so the cosine score set by retrieve is read from the
                    # state (its "values" update came before rerank started).
                    result = chunk["result"] or {}
                    yield ServerSentEvent(
                        event="citations",
                        data=CitationsEvent(
                            citations=result.get("citations", []),
                            confidence=state["confidence"],
                        ),
                    )
            elif mode == "custom":
                event = _custom_event(chunk)
                if event is not None:
                    yield event
    except Exception:
        logger.exception("query_stream: workflow failed")
        yield ServerSentEvent(event="error", data=ErrorEvent(message=_ERROR_ANSWER))
        state = {**state, "answer": _ERROR_ANSWER, "grounded": False}
    yield ServerSentEvent(event="final", data=_to_response(state, reranker))


_pump_tasks: set[asyncio.Task[None]] = set()


async def _iterate_in_own_task(stream: AsyncIterator[Any]) -> AsyncIterator[Any]:
    """Iterate ``stream`` in a separate task so a client disconnect stops it.

    FastAPI cancels the SSE response with an anyio cancel scope, which keeps
    re-cancelling the consumer while LangGraph awaits its own cleanup. The node
    task, and with it the LLM request, then keeps running. A separate task is
    cancelled once, outside that scope, and cleans up normally (ADR-0017).
    """
    queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()

    async def pump() -> None:
        try:
            async for item in stream:
                await queue.put(("item", item))
        except Exception as exc:
            await queue.put(("error", exc))
        else:
            await queue.put(("done", None))

    task = asyncio.create_task(pump())
    _pump_tasks.add(task)  # keep a reference so the task is not garbage collected
    task.add_done_callback(_pump_tasks.discard)
    try:
        while True:
            kind, value = await queue.get()
            if kind == "item":
                yield value
            elif kind == "error":
                raise value
            else:
                return
    finally:
        task.cancel()


def _task_event(
    chunk: dict[str, Any], started: dict[str, float]
) -> ServerSentEvent | None:
    task_id, node = chunk["id"], chunk["name"]
    if "input" in chunk:
        started[task_id] = time.monotonic()
        return ServerSentEvent(event="step_start", data=StepStartEvent(node=node))
    if "result" in chunk and chunk.get("error") is None:
        elapsed = time.monotonic() - started.pop(task_id, time.monotonic())
        return ServerSentEvent(
            event="step_end",
            data=StepEndEvent(node=node, elapsed_ms=round(elapsed * 1000)),
        )
    return None


def _custom_event(chunk: Any) -> ServerSentEvent | None:
    if not isinstance(chunk, dict):
        return None
    if chunk.get("type") == "token":
        return ServerSentEvent(event="token", data=TokenEvent(text=chunk["text"]))
    if chunk.get("type") == "error":
        return ServerSentEvent(event="error", data=ErrorEvent(message=chunk["message"]))
    return None
