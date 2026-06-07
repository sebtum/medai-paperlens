import logging

from qdrant_client import AsyncQdrantClient

from app.retrieval.embedding import EmbeddingProvider
from app.retrieval.search import search
from app.workflow.state import WorkflowState

logger = logging.getLogger(__name__)


def make_retrieve_node(client: AsyncQdrantClient, provider: EmbeddingProvider):
    async def retrieve_node(state: WorkflowState) -> dict:
        query = state["rewritten"]
        if not query:
            logger.warning("rewritten query is empty, falling back to original")
            query = state["question"]

        vector = await provider.embed(query)

        try:
            citations, confidence = await search(
                vector, client, top_k=5,
                score_threshold=state["score_threshold"],
            )
        except (ConnectionError, TimeoutError, OSError):
            logger.exception("Qdrant search failed")
            return {"citations": [], "confidence": 0.0}

        return {"citations": citations, "confidence": confidence}

    return retrieve_node
