from unittest.mock import AsyncMock

import numpy as np
from conftest import FakeLlm
from langgraph.graph import END, START, StateGraph
from qdrant_client import AsyncQdrantClient

from app.models.query import Citation
from app.retrieval.ingestion import VECTOR_DIM
from app.workflow.classify import classify_node
from app.workflow.generate import _ERROR_ANSWER, make_generate_node
from app.workflow.graph import build_workflow
from app.workflow.state import WorkflowState


def _safe_state(**overrides) -> WorkflowState:
    base: WorkflowState = {
        "question": "How do AI models detect pneumonia?",
        "rewritten": "",
        "is_unsafe": False,
        "citations": [],
        "confidence": 0.0,
        "answer": "",
        "grounded": False,
        "route": "",
        "score_threshold": 0.4,
    }
    base.update(overrides)  # type: ignore[typeddict-item]
    return base


def _citation(excerpt: str = "Test excerpt.") -> Citation:
    return Citation(
        title="Test Paper", authors=["Author A"], year=2024, doi=None, excerpt=excerpt
    )


async def test_classify_safe_query() -> None:
    result = classify_node(_safe_state())
    assert result["is_unsafe"] is False
    assert result["route"] == "retrieval"


async def test_classify_unsafe_query() -> None:
    result = classify_node(
        _safe_state(question="My symptoms are bad, should I take aspirin?")
    )
    assert result["is_unsafe"] is True
    assert result["route"] == "unsafe_medical_advice"
    assert result["grounded"] is False


async def test_generate_with_citations() -> None:
    fake_llm = FakeLlm(answer="AI achieves 90% accuracy in pneumonia detection.")

    state = _safe_state(
        question="How accurate is AI in pneumonia detection?",
        rewritten="AI pneumonia detection accuracy",
        citations=[_citation("AI models achieve high accuracy in X-ray analysis.")],
        confidence=0.85,
        route="retrieval",
    )

    generate_node = make_generate_node(fake_llm)
    result = await generate_node(state)

    assert result["answer"] == "AI achieves 90% accuracy in pneumonia detection."
    assert len(fake_llm.calls) == 1


async def test_generate_fallback_on_ollama_error() -> None:
    fake_llm = FakeLlm(error=Exception("Ollama unavailable"))

    state = _safe_state(
        citations=[_citation()],
        confidence=0.7,
        route="retrieval",
    )

    generate_node = make_generate_node(fake_llm)
    result = await generate_node(state)

    assert result["answer"] == _ERROR_ANSWER


def _generate_graph(fake_llm: FakeLlm):
    graph = StateGraph(WorkflowState)
    graph.add_node("generate", make_generate_node(fake_llm))
    graph.add_edge(START, "generate")
    graph.add_edge("generate", END)
    return graph.compile()


async def test_generate_answer_is_stripped_join_of_tokens() -> None:
    fake_llm = FakeLlm(tokens=["  Hello", " ", "world.\n"])
    result = await make_generate_node(fake_llm)(_safe_state(citations=[_citation()]))
    assert result["answer"] == "Hello world."


async def test_generate_empty_tokens_returns_error_answer() -> None:
    fake_llm = FakeLlm(tokens=[" ", "\n"])
    result = await make_generate_node(fake_llm)(_safe_state(citations=[_citation()]))
    assert result["answer"] == _ERROR_ANSWER


async def test_generate_streams_tokens_in_order_via_custom_mode() -> None:
    graph = _generate_graph(FakeLlm(tokens=["A", "B", "C"]))
    chunks = [
        c
        async for c in graph.astream(
            _safe_state(citations=[_citation()]), stream_mode=["custom"]
        )
    ]
    assert chunks == [("custom", {"type": "token", "text": t}) for t in ("A", "B", "C")]


async def test_generate_llm_error_emits_error_event() -> None:
    graph = _generate_graph(FakeLlm(error=Exception("boom")))
    chunks = [
        c
        async for c in graph.astream(
            _safe_state(citations=[_citation()]), stream_mode=["custom", "values"]
        )
    ]
    assert ("custom", {"type": "error", "message": _ERROR_ANSWER}) in chunks
    assert chunks[-1][1]["answer"] == _ERROR_ANSWER


async def test_generate_under_ainvoke_has_no_writer_side_effects() -> None:
    graph = _generate_graph(FakeLlm(answer="Done."))
    state = await graph.ainvoke(_safe_state(citations=[_citation()]))
    assert state["answer"] == "Done."


async def test_full_workflow_e2e_safe() -> None:
    mock_client = AsyncMock(spec=AsyncQdrantClient)
    mock_client.query_points.return_value.points = []

    mock_provider = AsyncMock()
    mock_provider.embed.return_value = np.zeros(VECTOR_DIM, dtype=np.float32)

    fake_llm = FakeLlm(answer="No matching papers found for this query.")

    workflow = build_workflow(mock_client, mock_provider, fake_llm)

    state = await workflow.ainvoke(
        _safe_state(
            question="What AI models are used in radiology?",
            rewritten="What AI models are used in radiology?",
        )
    )

    assert state["is_unsafe"] is False
    assert state["route"] == "retrieval"
    assert isinstance(state["citations"], list)
    assert isinstance(state["grounded"], bool)
    assert fake_llm.calls


async def test_full_workflow_e2e_unsafe() -> None:
    mock_client = AsyncMock(spec=AsyncQdrantClient)
    mock_provider = AsyncMock()
    fake_llm = FakeLlm()

    workflow = build_workflow(mock_client, mock_provider, fake_llm)

    state = await workflow.ainvoke(
        _safe_state(
            question="Should I take my medication for my condition?",
            rewritten="Should I take my medication for my condition?",
        )
    )

    assert state["is_unsafe"] is True
    assert state["route"] == "unsafe_medical_advice"
    assert state["grounded"] is False
    assert fake_llm.calls == []
    mock_client.query_points.assert_not_called()
