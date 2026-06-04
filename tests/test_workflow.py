from unittest.mock import AsyncMock

import numpy as np
from qdrant_client import AsyncQdrantClient

from app.models.query import Citation
from app.retrieval.ingestion import VECTOR_DIM
from app.workflow.classify import classify_node
from app.workflow.generate import make_generate_node, _ERROR_ANSWER
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
    mock_llm = AsyncMock()
    mock_llm.generate.return_value = "AI achieves 90% accuracy in pneumonia detection."

    state = _safe_state(
        question="How accurate is AI in pneumonia detection?",
        rewritten="AI pneumonia detection accuracy",
        citations=[_citation("AI models achieve high accuracy in X-ray analysis.")],
        confidence=0.85,
        route="retrieval",
    )

    generate_node = make_generate_node(mock_llm)
    result = await generate_node(state)

    assert result["answer"] == "AI achieves 90% accuracy in pneumonia detection."
    mock_llm.generate.assert_called_once()


async def test_generate_fallback_on_ollama_error() -> None:
    mock_llm = AsyncMock()
    mock_llm.generate.side_effect = Exception("Ollama unavailable")

    state = _safe_state(
        citations=[_citation()],
        confidence=0.7,
        route="retrieval",
    )

    generate_node = make_generate_node(mock_llm)
    result = await generate_node(state)

    assert result["answer"] == _ERROR_ANSWER


async def test_full_workflow_e2e_safe() -> None:
    mock_client = AsyncMock(spec=AsyncQdrantClient)
    mock_client.query_points.return_value.points = []

    mock_provider = AsyncMock()
    mock_provider.embed.return_value = np.zeros(VECTOR_DIM, dtype=np.float32)

    mock_llm = AsyncMock()
    mock_llm.generate.return_value = "No matching papers found for this query."

    workflow = build_workflow(mock_client, mock_provider, mock_llm)

    state = await workflow.ainvoke(_safe_state(
        question="What AI models are used in radiology?",
        rewritten="What AI models are used in radiology?",
    ))

    assert state["is_unsafe"] is False
    assert state["route"] == "retrieval"
    assert isinstance(state["citations"], list)
    assert isinstance(state["grounded"], bool)
    mock_llm.generate.assert_called()


async def test_full_workflow_e2e_unsafe() -> None:
    mock_client = AsyncMock(spec=AsyncQdrantClient)
    mock_provider = AsyncMock()
    mock_llm = AsyncMock()

    workflow = build_workflow(mock_client, mock_provider, mock_llm)

    state = await workflow.ainvoke(_safe_state(
        question="Should I take my medication for my condition?",
        rewritten="Should I take my medication for my condition?",
    ))

    assert state["is_unsafe"] is True
    assert state["route"] == "unsafe_medical_advice"
    assert state["grounded"] is False
    mock_llm.generate.assert_not_called()
    mock_client.query_points.assert_not_called()
