import logging
import time

from langgraph.config import get_stream_writer
from langgraph.types import StreamWriter

from app.llm.base import LlmProvider
from app.models.query import Citation
from app.workflow.state import WorkflowState

logger = logging.getLogger(__name__)

_PROMPT = (
    "You are a medical AI research assistant. "
    "Based on the following evidence from research papers, answer the question. "
    "Do not give personal medical advice. Cite specific papers where relevant.\n\n"
    "Question: {question}\n\nEvidence:\n{evidence}\n\nAnswer:"
)

_ERROR_ANSWER = "An error occurred while generating the answer. Please try again later."

# Evidence is sized against the model's context window (OLLAMA_NUM_CTX, default 8192):
# the budget leaves room for the prompt, the question and the answer. Five chunks of at
# most 512 tokens stay well under it, so the guard is only a safety net.
DEFAULT_EVIDENCE_BUDGET_TOKENS = 4000
_CHARS_PER_TOKEN = 3  # conservative: scientific text with numbers and symbols
_SEPARATOR = "\n\n"


def _evidence_item(index: int, citation: Citation) -> str:
    """`[n] Title — Section (p. 3)`, then the excerpt; missing fields are left out."""
    label = f"[{index}] {citation.title}"
    if citation.section:
        label += f" — {citation.section}"
    if citation.page is not None:
        label += f" (p. {citation.page})"
    return f"{label}\n{citation.excerpt}"


def _estimated_tokens(text: str) -> int:
    return -(-len(text) // _CHARS_PER_TOKEN)


def _fit_evidence(items: list[str], budget_tokens: int) -> list[str]:
    """Keep the highest-ranked items that fit; whole items are dropped, never cut.

    The top item is always kept, so a question with citations is never sent without
    evidence.
    """
    kept: list[str] = []
    used = 0
    for item in items:
        cost = _estimated_tokens(item + _SEPARATOR)
        if kept and used + cost > budget_tokens:
            break
        kept.append(item)
        used += cost
    if len(kept) < len(items):
        logger.warning(
            "generate_node: evidence over budget (%d tokens), dropped %d of %d items",
            budget_tokens,
            len(items) - len(kept),
            len(items),
        )
    return kept


def _stream_writer() -> StreamWriter:
    """LangGraph custom-event writer; a no-op when called outside a graph run."""
    try:
        return get_stream_writer()
    except RuntimeError, KeyError:
        return lambda _chunk: None


def make_generate_node(
    llm: LlmProvider, *, evidence_budget_tokens: int = DEFAULT_EVIDENCE_BUDGET_TOKENS
):
    async def generate_node(state: WorkflowState) -> dict:
        if not state["citations"]:
            logger.info("generate_node: no citations, skipping LLM call")
            return {"answer": "No relevant papers found."}

        evidence = _fit_evidence(
            [_evidence_item(i + 1, c) for i, c in enumerate(state["citations"])],
            evidence_budget_tokens,
        )
        evidence_text = _SEPARATOR.join(evidence)
        logger.info(
            "generate_node: start (%d of %d citations)",
            len(evidence),
            len(state["citations"]),
        )
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
