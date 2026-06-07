from typing import TypedDict

from app.models.query import Citation


class WorkflowState(TypedDict):
    question: str
    rewritten: str
    is_unsafe: bool
    citations: list[Citation]
    confidence: float
    answer: str
    grounded: bool
    route: str
    score_threshold: float
