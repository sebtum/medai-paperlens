from typing import Annotated

from fastapi import APIRouter, Depends
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from app.core.exceptions import UnsafeQueryError
from app.models.query import QueryRequest, QueryResponse
from app.retrieval.client import get_client
from app.retrieval.embedding import embed_async, get_model
from app.retrieval.search import search

router = APIRouter()

# Phrase-pattern safety filter — interim placeholder until Phase 4 replaces this
# with a proper classifier (Llama Guard / NeMo Guardrails + Ollama).
# Catches personal-advice patterns; research questions about AI in medicine are allowed.
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
async def query(
    request: QueryRequest,
    client: Annotated[QdrantClient, Depends(get_client)],
    model: Annotated[SentenceTransformer, Depends(get_model)],
) -> QueryResponse:
    if _is_unsafe(request.question):
        raise UnsafeQueryError(request.question)

    vector = await embed_async(request.question, model)
    citations, confidence = search(vector, client, top_k=5)
    grounded = len(citations) > 0

    return QueryResponse(
        answer=citations[0].excerpt if grounded else "No relevant papers found.",
        citations=citations,
        confidence=confidence,
        grounded=grounded,
        debug={"route": "retrieval"},
    )
