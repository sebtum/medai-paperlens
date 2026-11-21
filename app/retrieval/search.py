import logging

import numpy as np
import numpy.typing as npt
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Document,
    Fusion,
    FusionQuery,
    Prefetch,
    ScoredPoint,
)

from app.evaluation.metrics import RankedHit
from app.models.query import DEFAULT_SCORE_THRESHOLD, Citation
from app.retrieval.ingestion import (
    BM25_MODEL,
    BM25_VECTOR_NAME,
    COLLECTION_NAME,
    DENSE_VECTOR_NAME,
)

logger = logging.getLogger(__name__)


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _optional_int(value: object) -> int | None:
    # bool is an int subclass; a stray True/False is not a page number.
    return value if isinstance(value, int) and not isinstance(value, bool) else None


async def _query_points(
    vector: npt.NDArray[np.float32],
    client: AsyncQdrantClient,
    *,
    top_k: int,
    score_threshold: float,
    collection_name: str,
) -> list[ScoredPoint]:
    results = await client.query_points(
        collection_name=collection_name,
        query=vector,
        limit=top_k,
        score_threshold=score_threshold,
    )
    return list(results.points)


def _ranked_hits(points: list[ScoredPoint]) -> list[RankedHit]:
    """Scored points as `RankedHit`s; points without a chunk identity are skipped."""
    hits: list[RankedHit] = []
    for point in points:
        payload = point.payload or {}
        paper_id = _optional_str(payload.get("paper_id"))
        section = _optional_str(payload.get("section"))
        chunk_index = _optional_int(payload.get("chunk_index"))
        if paper_id is None or section is None or chunk_index is None:
            logger.warning("Skipping result id=%s without chunk identity", point.id)
            continue
        hits.append(
            RankedHit(
                paper_id=paper_id,
                section=section,
                chunk_index=chunk_index,
                chunk_type=_optional_str(payload.get("chunk_type")) or "text",
                score=float(point.score),
            )
        )
    return hits


async def search_hits(
    vector: npt.NDArray[np.float32],
    client: AsyncQdrantClient,
    *,
    top_k: int,
    score_threshold: float,
    collection_name: str = COLLECTION_NAME,
) -> list[RankedHit]:
    """Ranked chunk hits with their scores, for retrieval evaluation (ADR-0023).

    A point whose payload lacks a usable `paper_id`, `section` or `chunk_index` (a
    pre-Phase-8 point) cannot be matched against labels and is skipped with a warning.
    A missing `chunk_type` counts as `"text"`.
    """
    points = await _query_points(
        vector,
        client,
        top_k=top_k,
        score_threshold=score_threshold,
        collection_name=collection_name,
    )
    return _ranked_hits(points)


async def search_hits_hybrid(
    vector: npt.NDArray[np.float32],
    query_text: str,
    client: AsyncQdrantClient,
    *,
    top_k: int,
    dense_threshold: float,
    prefetch_limit: int | None = None,
    collection_name: str,
) -> list[RankedHit]:
    """Hybrid (dense plus BM25) ranked chunk hits for retrieval evaluation (ADR-0028).

    Eval only: `collection_name` must be a hybrid collection, built by
    `ingest(..., hybrid=True)`, with the named dense vector `dense` and the sparse
    vector `bm25`; the app's `papers` collection has neither, so there is deliberately
    no default.

    Two candidate lists are fetched inside Qdrant and merged by reciprocal rank fusion
    (RRF): the dense list (`vector`, cosine at or above `dense_threshold`, at most
    `prefetch_limit` hits, default `top_k`) and the BM25 list (`query_text`, same limit,
    no threshold, so a chunk can enter through keywords alone). The merged list is
    cut at `top_k`.

    `vector` is what the dense side embedded, including the BGE instruction prefix.
    `query_text` is the same rewritten query WITHOUT that prefix: the instruction is
    a dense-model convention, and its words ("represent", "sentence", "searching",
    "relevant", "passages") would only add noise terms on the keyword side. The BM25
    options are the `qdrant/bm25` defaults, as at index time; none is passed.

    `RankedHit.score` is the RRF score, which is rank-based: it is neither a cosine
    value nor a BM25 value and cannot be cut at 0.6 (the run's score kind is `rrf`).
    """
    limit = prefetch_limit if prefetch_limit is not None else top_k
    results = await client.query_points(
        collection_name=collection_name,
        prefetch=[
            Prefetch(
                query=vector.tolist(),
                using=DENSE_VECTOR_NAME,
                limit=limit,
                score_threshold=dense_threshold,
            ),
            Prefetch(
                query=Document(text=query_text, model=BM25_MODEL),
                using=BM25_VECTOR_NAME,
                limit=limit,
            ),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=top_k,
        with_payload=True,
    )
    return _ranked_hits(list(results.points))


async def search(
    vector: npt.NDArray[np.float32],
    client: AsyncQdrantClient,
    top_k: int = 5,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    *,
    collection_name: str = COLLECTION_NAME,
) -> tuple[list[Citation], float]:
    points = await _query_points(
        vector,
        client,
        top_k=top_k,
        score_threshold=score_threshold,
        collection_name=collection_name,
    )

    citations = []
    for point in points:
        payload = point.payload or {}
        try:
            citations.append(
                Citation(
                    title=str(payload.get("title", "")),
                    authors=list(payload.get("authors") or []),
                    year=int(payload.get("year") or 0),
                    doi=payload.get("doi"),
                    excerpt=str(payload.get("text", "")),
                    paper_id=_optional_str(payload.get("paper_id")),
                    section=_optional_str(payload.get("section")),
                    page=_optional_int(payload.get("page")),
                    chunk_index=_optional_int(payload.get("chunk_index")),
                )
            )
        except (TypeError, ValueError) as exc:
            logger.warning("Skipping malformed result id=%s: %s", point.id, exc)

    confidence = float(points[0].score) if points else 0.0
    logger.info(
        "search: %d/%d results above threshold (best=%.2f)",
        len(citations),
        top_k,
        confidence,
    )
    return citations, confidence
