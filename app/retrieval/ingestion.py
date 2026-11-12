"""Ingest the paper corpus into Qdrant: one point per chunk (ADR-0021, ADR-0022).

Each manifest in `papers_dir` is chunked together with its parsed Docling document from
`parsed_dir` (written by `scripts/parse_papers.py`). A paper without a usable cache file
falls back to its abstract only, so a fresh clone still works.
"""

import json
import logging
import uuid
from pathlib import Path
from typing import Any

from docling_core.transforms.chunker.tokenizer.base import BaseTokenizer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointStruct,
    VectorParams,
)
from sentence_transformers import SentenceTransformer

from app.retrieval.chunking import chunk_paper, make_embedder_tokenizer
from app.retrieval.parsed import load_parsed

logger = logging.getLogger(__name__)

COLLECTION_NAME = "papers"
BATCH_SIZE = 32

# Fixed namespace for point IDs: the same (paper_id, chunk_index) always maps to the
# same ID, so re-ingesting overwrites rather than duplicates.
PAPERLENS_NS = uuid.uuid5(uuid.NAMESPACE_DNS, "paperlens.medai")


def point_id(paper_id: str, chunk_index: int) -> str:
    return str(uuid.uuid5(PAPERLENS_NS, f"{paper_id}:{chunk_index}"))


def _load_manifests(papers_dir: Path) -> list[tuple[str, dict[str, Any]]]:
    """Valid manifests as `(paper_id, manifest)`; `paper_id` is the file stem."""
    papers: list[tuple[str, dict[str, Any]]] = []
    for path in sorted(papers_dir.glob("*.json")):
        try:
            with path.open(encoding="utf-8") as f:
                paper = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Skipping %s: %s", path.name, exc)
            continue

        abstract = paper.get("abstract", "").strip()
        if not abstract:
            logger.warning("Skipping %s: missing or empty abstract", path.name)
            continue

        papers.append((path.stem, paper))
    return papers


def _collection_is_stale(
    client: QdrantClient, vector_dim: int, model_name: str
) -> str | None:
    """Why the existing collection can't be reused with this model, or `None`.

    Checks the vector size, then one sampled point's `embedding_model`. A point without
    that field comes from before Phase 8 (MiniLM, integer IDs), even if its size
    happens to match.
    """
    vectors = client.get_collection(COLLECTION_NAME).config.params.vectors
    if not isinstance(vectors, VectorParams):
        return "it uses named vectors"
    if vectors.size != vector_dim:
        return f"vector size {vectors.size} differs from the model's {vector_dim}"

    sample, _ = client.scroll(
        COLLECTION_NAME, limit=1, with_payload=["embedding_model"], with_vectors=False
    )
    if sample:
        stored = (sample[0].payload or {}).get("embedding_model")
        if stored != model_name:
            return f"it was built with embedding model {stored!r}, not {model_name!r}"
    return None


def _prepare_collection(
    client: QdrantClient, vector_dim: int, model_name: str, overwrite: bool
) -> None:
    existing = {c.name for c in client.get_collections().collections}
    if COLLECTION_NAME in existing:
        reason = "--overwrite requested" if overwrite else None
        if reason is None:
            reason = _collection_is_stale(client, vector_dim, model_name)
            if reason is not None:
                logger.warning("Recreating collection %r: %s", COLLECTION_NAME, reason)
        if reason is not None:
            client.delete_collection(COLLECTION_NAME)
            existing.discard(COLLECTION_NAME)

    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
        )


def _delete_paper_points(client: QdrantClient, paper_id: str) -> None:
    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=FilterSelector(
            filter=Filter(
                must=[FieldCondition(key="paper_id", match=MatchValue(value=paper_id))]
            )
        ),
    )


def ingest(
    papers_dir: Path,
    client: QdrantClient,
    model: SentenceTransformer,
    *,
    model_name: str,
    parsed_dir: Path | None = None,
    overwrite: bool = False,
    tokenizer: BaseTokenizer | None = None,
) -> int:
    """Chunk, embed and upsert every paper; return the number of chunks indexed.

    `model_name` is stored on every point, and a collection built with another model
    (or another vector size) is recreated. `parsed_dir=None` means no parsed cache, so
    every paper is indexed from its abstract. `tokenizer` defaults to the embedder's own
    (`make_embedder_tokenizer`); it is a parameter so tests need no model download.
    """
    papers = _load_manifests(papers_dir)

    vector_dim = model.get_sentence_embedding_dimension()
    if vector_dim is None:
        raise ValueError("embedding model does not report its vector dimension")

    _prepare_collection(client, vector_dim, model_name, overwrite)

    if not papers:
        return 0
    if tokenizer is None:
        tokenizer = make_embedder_tokenizer(model)

    total = 0
    parsed_count = 0
    for paper_id, paper in papers:
        doc = load_parsed(parsed_dir, paper_id) if parsed_dir is not None else None
        if doc is None:
            logger.warning(
                "%s: no usable parsed cache, indexing the abstract only", paper_id
            )
        else:
            parsed_count += 1
        chunks = chunk_paper(paper, doc, tokenizer, paper_id=paper_id)

        # Delete first so a paper that now has fewer chunks leaves nothing stale.
        _delete_paper_points(client, paper_id)
        for start in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[start : start + BATCH_SIZE]
            embeddings = model.encode(
                [chunk.embed_text for chunk in batch], show_progress_bar=False
            )
            points = [
                PointStruct(
                    id=point_id(paper_id, chunk.chunk_index),
                    vector=embedding.tolist(),  # type: ignore[union-attr]
                    payload={
                        "title": paper.get("title", ""),
                        "authors": paper.get("authors", []),
                        "year": paper.get("year", 0),
                        "doi": paper.get("doi"),
                        "paper_id": paper_id,
                        "section": chunk.section,
                        "page": chunk.page,
                        "chunk_index": chunk.chunk_index,
                        "chunk_type": chunk.chunk_type,
                        "text": chunk.text,
                        "embedding_model": model_name,
                    },
                )
                for chunk, embedding in zip(batch, embeddings, strict=True)
            ]
            client.upsert(collection_name=COLLECTION_NAME, points=points)
            total += len(points)

    if parsed_count == 0:
        logger.warning(
            "No parsed papers found in %s: the whole corpus is indexed from "
            "abstracts only. Run scripts/download_papers.py and "
            "scripts/parse_papers.py for full text.",
            parsed_dir,
        )
    return total
