import logging

from app.llm.base import LlmProvider
from app.workflow.state import WorkflowState

logger = logging.getLogger(__name__)

_PROMPT = (
    "You are a medical literature search assistant. "
    "Rewrite the question to improve retrieval over academic paper abstracts. "
    "Make it concise and research-focused. "
    "Return only the rewritten question, nothing else.\n\n"
    "Question: {question}\nRewritten question:"
)


def make_rewrite_node(llm: LlmProvider):
    async def rewrite_node(state: WorkflowState) -> dict:
        try:
            rewritten = (await llm.generate(
                _PROMPT.format(question=state["question"])
            )).strip()
        except Exception:
            logger.exception("Query rewrite failed, using original question")
            rewritten = ""
        return {"rewritten": rewritten or state["question"]}

    return rewrite_node
