from typing import Annotated

from fastapi import APIRouter, Depends
from qdrant_client import AsyncQdrantClient

from app.llm.base import LlmProvider
from app.llm.factory import get_llm_provider
from app.models.query import QueryRequest, QueryResponse
from app.retrieval.client import get_async_client
from app.retrieval.embedding import EmbeddingProvider, get_embedding_provider
from app.workflow.graph import build_workflow
from app.workflow.state import WorkflowState

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    client: Annotated[AsyncQdrantClient, Depends(get_async_client)],
    provider: Annotated[EmbeddingProvider, Depends(get_embedding_provider)],
    llm: Annotated[LlmProvider, Depends(get_llm_provider)],
) -> QueryResponse:
    workflow = build_workflow(client, provider, llm)
    initial: WorkflowState = {
        "question": request.question,
        "rewritten": request.question,
        "is_unsafe": False,
        "citations": [],
        "confidence": 0.0,
        "answer": "",
        "grounded": False,
        "route": "",
        "score_threshold": request.score_threshold,
    }
    state = await workflow.ainvoke(initial)
    return QueryResponse(
        answer=state["answer"],
        citations=state["citations"],
        confidence=state["confidence"],
        grounded=state["grounded"],
        debug={"route": state["route"], "rewritten": state["rewritten"]},
    )
