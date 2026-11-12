"""Startup check that the API embeds queries like the indexed corpus."""

import logging

from qdrant_client import AsyncQdrantClient

logger = logging.getLogger(__name__)

COLLECTION_NAME = "papers"


async def embedding_model_mismatch(
    client: AsyncQdrantClient, model_name: str
) -> str | None:
    """Describe a mismatch between `model_name` and the indexed corpus, else `None`.

    Looks at one point's `embedding_model` payload. Two models with the same vector
    size would otherwise fail silently: every score is low and nothing is retrieved.
    An empty or missing collection, or a Qdrant that can't be reached, gives `None`.
    """
    try:
        points, _ = await client.scroll(
            COLLECTION_NAME,
            limit=1,
            with_payload=["embedding_model"],
            with_vectors=False,
        )
    except Exception as exc:  # startup must not depend on Qdrant
        logger.info("Embedding model check skipped: %s", exc)
        return None
    if not points:
        return None
    stored = (points[0].payload or {}).get("embedding_model")
    if stored == model_name:
        return None
    return (
        f"collection {COLLECTION_NAME!r} was indexed with embedding model {stored!r} "
        f"but EMBEDDING_MODEL is {model_name!r}; re-run ingest or fix EMBEDDING_MODEL"
    )
