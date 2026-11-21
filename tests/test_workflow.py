from unittest.mock import AsyncMock

import numpy as np
import pytest
from conftest import TEST_VECTOR_DIM, FakeLlm, FakeReranker
from langgraph.graph import END, START, StateGraph
from qdrant_client import AsyncQdrantClient

from app.models.query import Citation
from app.workflow.classify import classify_node
from app.workflow.generate import _ERROR_ANSWER, make_generate_node
from app.workflow.graph import build_workflow
from app.workflow.rerank import make_rerank_node
from app.workflow.retrieve import RETRIEVE_DEPTH, make_retrieve_node
from app.workflow.rewrite import clean_rewrite, make_rewrite_node
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
        "rerank_top_score": None,
        "rerank_fallback": False,
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


def _chunk_citation(index: int, excerpt: str = "Body.", **fields) -> Citation:
    return Citation(
        title=f"Paper {index}",
        authors=["A"],
        year=2024,
        excerpt=excerpt,
        **fields,
    )


async def test_generate_evidence_labels_title_section_and_page() -> None:
    fake_llm = FakeLlm()
    citations = [
        _chunk_citation(1, "Alpha.", section="Methods > Datasets", page=4),
        _chunk_citation(2, "Beta.", section="Results"),  # no page
        _chunk_citation(3, "Gamma.", page=7),  # no section
        _chunk_citation(4, "Delta."),  # abstract-only citation
    ]

    await make_generate_node(fake_llm)(_safe_state(citations=citations))

    (prompt,) = fake_llm.calls
    assert isinstance(prompt, str)
    assert "[1] Paper 1 — Methods > Datasets (p. 4)\nAlpha." in prompt
    assert "[2] Paper 2 — Results\nBeta." in prompt
    assert "[3] Paper 3 (p. 7)\nGamma." in prompt
    assert "[4] Paper 4\nDelta." in prompt


async def test_generate_guard_drops_whole_lowest_ranked_items(caplog) -> None:
    fake_llm = FakeLlm()
    excerpts = [f"E{i} " + "x" * 296 for i in range(1, 6)]  # ~100 tokens per item
    citations = [_chunk_citation(i + 1, e) for i, e in enumerate(excerpts)]

    with caplog.at_level("WARNING"):
        await make_generate_node(fake_llm, evidence_budget_tokens=300)(
            _safe_state(citations=citations)
        )

    (prompt,) = fake_llm.calls
    assert isinstance(prompt, str)
    kept = [i for i, e in enumerate(excerpts, 1) if e in prompt]
    assert kept == [1, 2]  # a prefix: the top-ranked items survive
    assert all(f"[{i}]" not in prompt for i in (3, 4, 5))
    assert any("dropped 3 of 5" in r.getMessage() for r in caplog.records)


async def test_generate_guard_never_cuts_an_item() -> None:
    fake_llm = FakeLlm()
    excerpt = "start " + "y" * 600 + " end"

    await make_generate_node(fake_llm, evidence_budget_tokens=10)(
        _safe_state(citations=[_chunk_citation(1, excerpt)])
    )

    # Over budget on its own, but the top item is kept whole rather than cut or dropped.
    (prompt,) = fake_llm.calls
    assert isinstance(prompt, str)
    assert excerpt in prompt


async def test_generate_guard_is_silent_when_evidence_fits(caplog) -> None:
    fake_llm = FakeLlm()
    citations = [_chunk_citation(i, "short") for i in range(1, 6)]

    with caplog.at_level("WARNING"):
        await make_generate_node(fake_llm)(_safe_state(citations=citations))

    (prompt,) = fake_llm.calls
    assert isinstance(prompt, str)
    assert all(f"[{i}] Paper {i}" in prompt for i in range(1, 6))
    assert not caplog.records


async def test_full_workflow_e2e_safe() -> None:
    mock_client = AsyncMock(spec=AsyncQdrantClient)
    mock_client.query_points.return_value.points = []

    mock_provider = AsyncMock()
    mock_provider.embed.return_value = np.zeros(TEST_VECTOR_DIM, dtype=np.float32)

    fake_llm = FakeLlm(answer="No matching papers found for this query.")

    workflow = build_workflow(mock_client, mock_provider, fake_llm, None)

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

    workflow = build_workflow(mock_client, mock_provider, fake_llm, None)

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


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("What is X?", "What is X?"),
        ("  What is X? \n", "What is X?"),
        ("Rewritten question: What is X?", "What is X?"),
        ("rewritten question:What is X?", "What is X?"),
        ("REWRITTEN QUESTION :  What is X?", "What is X?"),
        ("Rewritten question: Rewritten question: What is X?", "What is X?"),
        ('"What is X?"', "What is X?"),
        ("'What is X?'", "What is X?"),
        (
            "\N{LEFT DOUBLE QUOTATION MARK}What is X?\N{RIGHT DOUBLE QUOTATION MARK}",
            "What is X?",
        ),
        ('Rewritten question: "What is X?"', "What is X?"),
        ('"Rewritten question: What is X?"', "What is X?"),
        ("Rewritten question:", ""),
        ('""', ""),
        ("   ", ""),
        # Only a leading label is removed; a label inside the text stays.
        (
            "Explain the Rewritten question: idea",
            "Explain the Rewritten question: idea",
        ),
    ],
)
def test_clean_rewrite(raw: str, expected: str) -> None:
    assert clean_rewrite(raw) == expected


async def test_rewrite_node_uses_the_cleaned_text() -> None:
    node = make_rewrite_node(
        FakeLlm(answer="Rewritten question: AI pneumonia accuracy")
    )
    result = await node(_safe_state())
    assert result == {"rewritten": "AI pneumonia accuracy"}


async def test_rewrite_node_falls_back_when_only_the_label_is_returned() -> None:
    node = make_rewrite_node(FakeLlm(answer="Rewritten question:"))
    result = await node(_safe_state(question="Original question?"))
    assert result == {"rewritten": "Original question?"}


# --- retrieve depth and rerank step (Phase 10) ---------------------------------------


def _ranked_citation(n: int) -> Citation:
    return Citation(
        title=f"Paper {n}",
        authors=[],
        year=2024,
        excerpt=f"Excerpt {n}.",
        section=f"Section {n}",
    )


def _ranked_state(count: int, **overrides) -> WorkflowState:
    fields = {
        "rewritten": "rewritten question",
        "citations": [_ranked_citation(n) for n in range(1, count + 1)],
        **overrides,
    }
    return _safe_state(**fields)


async def test_retrieve_fetches_the_retrieve_depth() -> None:
    mock_client = AsyncMock(spec=AsyncQdrantClient)
    mock_client.query_points.return_value.points = []
    mock_provider = AsyncMock()
    mock_provider.embed.return_value = np.zeros(TEST_VECTOR_DIM, dtype=np.float32)

    await make_retrieve_node(mock_client, mock_provider)(
        _safe_state(rewritten="q", score_threshold=0.6)
    )

    assert RETRIEVE_DEPTH == 20
    assert mock_client.query_points.call_args.kwargs["limit"] == RETRIEVE_DEPTH


async def test_rerank_node_orders_by_score_and_cuts_to_keep() -> None:
    scores = [0.1, 0.9, 0.3, 0.8, 0.2, 0.7, 0.4]
    reranker = FakeReranker(scores=scores)

    result = await make_rerank_node(reranker, keep=3)(_ranked_state(7))

    assert [c.title for c in result["citations"]] == ["Paper 2", "Paper 4", "Paper 6"]
    assert result["rerank_top_score"] == 0.9
    assert result["rerank_fallback"] is False


async def test_rerank_node_sends_rewritten_query_and_shared_passage_text() -> None:
    reranker = FakeReranker(scores=[0.5, 0.4])

    await make_rerank_node(reranker)(_ranked_state(2))

    ((query, passages),) = reranker.calls
    assert query == "rewritten question"
    assert passages == [
        "Paper 1\nSection 1\nExcerpt 1.",
        "Paper 2\nSection 2\nExcerpt 2.",
    ]


async def test_rerank_node_falls_back_to_question_when_rewrite_is_empty() -> None:
    reranker = FakeReranker(scores=[0.5])

    await make_rerank_node(reranker)(_ranked_state(1, rewritten=""))

    assert reranker.calls[0][0] == "How do AI models detect pneumonia?"


async def test_rerank_node_ties_keep_the_vector_order() -> None:
    reranker = FakeReranker(scores=[0.5, 0.5, 0.5, 0.9])

    result = await make_rerank_node(reranker, keep=3)(_ranked_state(4))

    assert [c.title for c in result["citations"]] == ["Paper 4", "Paper 1", "Paper 2"]


async def test_rerank_node_default_keeps_five() -> None:
    result = await make_rerank_node(FakeReranker(scores=[0.5] * 20))(_ranked_state(20))

    assert len(result["citations"]) == 5


async def test_rerank_node_off_cuts_to_five_in_vector_order() -> None:
    result = await make_rerank_node(None)(_ranked_state(20))

    assert [c.title for c in result["citations"]] == [f"Paper {n}" for n in range(1, 6)]
    assert result["rerank_top_score"] is None
    assert result["rerank_fallback"] is False


async def test_rerank_node_error_keeps_vector_order_and_flags_fallback(
    caplog,
) -> None:
    reranker = FakeReranker(error=RuntimeError("model crashed"))

    with caplog.at_level("ERROR"):
        result = await make_rerank_node(reranker)(_ranked_state(8))

    assert [c.title for c in result["citations"]] == [f"Paper {n}" for n in range(1, 6)]
    assert result["rerank_top_score"] is None
    assert result["rerank_fallback"] is True
    assert "reranking failed" in caplog.text


async def test_rerank_node_wrong_score_count_is_a_fallback() -> None:
    reranker = FakeReranker(scores=[0.9])  # fake returns fewer scores than passages

    result = await make_rerank_node(reranker)(_ranked_state(3))

    assert [c.title for c in result["citations"]] == ["Paper 1", "Paper 2", "Paper 3"]
    assert result["rerank_fallback"] is True


async def test_rerank_node_no_citations_skips_the_reranker() -> None:
    reranker = FakeReranker(scores=[])

    result = await make_rerank_node(reranker)(_ranked_state(0))

    assert result["citations"] == []
    assert result["rerank_top_score"] is None
    assert result["rerank_fallback"] is False
    assert reranker.calls == []
