import logging

import numpy as np
import numpy.typing as npt
from qdrant_client import AsyncQdrantClient

from app.models.query import Citation
from app.retrieval.ingestion import COLLECTION_NAME

logger = logging.getLogger(__name__)


async def search(
    vector: npt.NDArray[np.float32],
    client: AsyncQdrantClient,
    top_k: int = 5,
    score_threshold: float = 0.4,
) -> tuple[list[Citation], float]:
    results = await client.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        limit=top_k,
        score_threshold=score_threshold,
    )

    citations = []
    for point in results.points:
        payload = point.payload or {}
        try:
            citations.append(
                Citation(
                    title=str(payload.get("title", "")),
                    authors=list(payload.get("authors") or []),
                    year=int(payload.get("year") or 0),
                    doi=payload.get("doi"),
                    excerpt=str(payload.get("text", "")),
                )
            )
        except (TypeError, ValueError) as exc:
            logger.warning("Skipping malformed result id=%s: %s", point.id, exc)

    confidence = float(results.points[0].score) if results.points else 0.0
    logger.info(
        "search: %d/%d results above threshold (best=%.2f)",
        len(citations), top_k, confidence,
    )
    return citations, confidence
