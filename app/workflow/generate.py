import logging

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


def make_generate_node(llm: LlmProvider):
    async def generate_node(state: WorkflowState) -> dict:
        if not state["citations"]:
            return {"answer": "No relevant papers found."}

        evidence_text = "\n\n".join(
            f"[{i + 1}] {c.excerpt}" for i, c in enumerate(state["citations"])
        )

        try:
            answer = (await llm.generate(
                _PROMPT.format(question=state["question"], evidence=evidence_text)
            )).strip()
        except Exception:
            logger.exception("LLM-Generierung fehlgeschlagen")
            return {"answer": _ERROR_ANSWER}

        if not answer:
            logger.warning(
                "LLM returned empty string — possible refusal or safety-filter trigger."
            )
            return {"answer": _ERROR_ANSWER}

        return {"answer": answer}

    return generate_node
