from fastapi import APIRouter

from app.core.exceptions import UnsafeQueryError
from app.models.query import QueryRequest, QueryResponse

router = APIRouter()

# Phrases that indicate personal medical advice requests, not research questions.
# Research questions about AI in medicine ("How accurate is AI at diagnosing cancer?")
# are safe; requests for personal guidance ("my symptoms", "should I take") are not.
_UNSAFE_PATTERNS: frozenset[str] = frozenset({
    "diagnose me",
    "my symptoms",
    "should i take",
    "can i take",
    "prescribe me",
    "my prescription",
    "medical advice",
    "treat me",
    "my condition",
    "my illness",
    "my disease",
    "am i sick",
    "i have been diagnosed",
})


def _is_unsafe(question: str) -> bool:
    lower = question.lower()
    return any(pattern in lower for pattern in _UNSAFE_PATTERNS)


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    if _is_unsafe(request.question):
        raise UnsafeQueryError(request.question)
    return QueryResponse(
        answer="Retrieval pipeline not yet implemented.",
        citations=[],
        confidence=0.0,
        grounded=False,
        debug={"route": "stub"},
    )
