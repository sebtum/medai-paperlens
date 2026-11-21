import logging
import re
import time

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

REWRITE_CLEANUP_VERSION = "1"
"""Version of `clean_rewrite`. Part of the eval rewrite-cache key: change the cleanup
rules and you must bump this, so rewrites stored under the old rules are not reused.
"""

_LEADING_LABEL_RE = re.compile(r"^rewritten\s+question\s*:", re.IGNORECASE)
_QUOTE_PAIRS = {'"': '"', "'": "'", "“": "”"}


def clean_rewrite(text: str) -> str:
    """Strip what the small rewrite model echoes around the question.

    Removes surrounding whitespace, a leading "Rewritten question:" label (any case,
    repeated) and one pair of surrounding quotes, until nothing more changes. May return
    an empty string; the caller decides on the fallback.
    """
    cleaned = text.strip()
    while True:
        previous = cleaned
        cleaned = _LEADING_LABEL_RE.sub("", cleaned).strip()
        if len(cleaned) >= 2 and _QUOTE_PAIRS.get(cleaned[0]) == cleaned[-1]:
            cleaned = cleaned[1:-1].strip()
        if cleaned == previous:
            return cleaned


def make_rewrite_node(llm: LlmProvider):
    async def rewrite_node(state: WorkflowState) -> dict:
        logger.info("rewrite_node: start")
        t0 = time.monotonic()
        try:
            rewritten = clean_rewrite(
                await llm.generate(_PROMPT.format(question=state["question"]))
            )
        except Exception:
            logger.exception("Query rewrite failed, using original question")
            rewritten = ""
        elapsed = time.monotonic() - t0
        logger.info(
            "rewrite_node: done in %.1fs → %r", elapsed, rewritten or state["question"]
        )
        return {"rewritten": rewritten or state["question"]}

    return rewrite_node
