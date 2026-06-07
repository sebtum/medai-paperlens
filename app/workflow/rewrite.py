import logging
import time

from app.llm.base import LlmProvider
from app.workflow.state import WorkflowState

logger = logging.getLogger(__name__)

_PROMPT = (
    "You are a medical literature search assistant. "
    "Rewrite the question to improve retrieval over academic paper abstracts. "
    "Make it concise and research-focused. "
    "Return only the rewritten question, nothing else. /no_think\n\n"
    "Question: {question}\nRewritten question:"
)


def make_rewrite_node(llm: LlmProvider):
    async def rewrite_node(state: WorkflowState) -> dict:
        logger.info("rewrite_node: start")
        t0 = time.monotonic()
        try:
            rewritten = (await llm.generate(
                _PROMPT.format(question=state["question"])
            )).strip()
        except Exception:
            logger.exception("Query rewrite failed, using original question")
            rewritten = ""
        elapsed = time.monotonic() - t0
        logger.info(
            "rewrite_node: done in %.1fs → %r", elapsed, rewritten or state["question"]
        )
        return {"rewritten": rewritten or state["question"]}

    return rewrite_node
