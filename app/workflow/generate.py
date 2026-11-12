import logging
import time

from langgraph.config import get_stream_writer
from langgraph.types import StreamWriter

from app.llm.base import LlmProvider
from app.workflow.state import WorkflowState

logger = logging.getLogger(__name__)

_PROMPT = (
    "You are a medical AI research assistant. "
    "Based on the following evidence from research papers, answer the question. "
    "Do not give personal medical advice. Cite specific papers where relevant.\n\n"
    "Question: {question}\n\nEvidence:\n{evidence}\n\nAnswer:"
)

_ERROR_ANSWER = "An error occurred while generating the answer. Please try again later."


def _stream_writer() -> StreamWriter:
    """LangGraph custom-event writer; a no-op when called outside a graph run."""
    try:
        return get_stream_writer()
    except RuntimeError, KeyError:
        return lambda _chunk: None


def make_generate_node(llm: LlmProvider):
    async def generate_node(state: WorkflowState) -> dict:
        if not state["citations"]:
            logger.info("generate_node: no citations, skipping LLM call")
            return {"answer": "No relevant papers found."}

        evidence_text = "\n\n".join(
            f"[{i + 1}] {c.excerpt}" for i, c in enumerate(state["citations"])
        )
        logger.info("generate_node: start (%d citations)", len(state["citations"]))
        t0 = time.monotonic()
        write = _stream_writer()
        tokens: list[str] = []
        try:
            async for token in llm.stream(
                _PROMPT.format(question=state["question"], evidence=evidence_text)
            ):
                tokens.append(token)
                write({"type": "token", "text": token})
        except Exception:
            logger.exception(
                "LLM-Generierung fehlgeschlagen (%.1fs)", time.monotonic() - t0
            )
            write({"type": "error", "message": _ERROR_ANSWER})
            return {"answer": _ERROR_ANSWER}
        answer = "".join(tokens).strip()

        elapsed = time.monotonic() - t0
        if not answer:
            logger.warning(
                "generate_node: LLM returned empty string after %.1fs", elapsed
            )
            return {"answer": _ERROR_ANSWER}

        logger.info("generate_node: done in %.1fs (%d chars)", elapsed, len(answer))
        return {"answer": answer}

    return generate_node
