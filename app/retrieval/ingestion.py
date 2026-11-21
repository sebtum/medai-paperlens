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
    Document,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    Modifier,
    PointStruct,
    SparseVectorParams,
    VectorParams,
)
from sentence_transformers import SentenceTransformer

from app.retrieval.chunking import chunk_paper, make_embedder_tokenizer
from app.retrieval.parsed import load_parsed

logger = logging.getLogger(__name__)

COLLECTION_NAME = "papers"
BATCH_SIZE = 32

# Hybrid layout (Phase 10, ADR-0028; eval only): a NAMED dense vector plus a sparse BM25
# vector that Qdrant computes itself from a `Document`. The app's `papers` collection
# keeps its single unnamed dense vector.
DENSE_VECTOR_NAME = "dense"
BM25_VECTOR_NAME = "bm25"
BM25_MODEL = "qdrant/bm25"

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
    client: QdrantClient,
    vector_dim: int,
    model_name: str,
    collection_name: str,
    hybrid: bool = False,
) -> str | None:
    """Why the existing collection can't be reused with this model, or `None`.

    Checks the layout (unnamed dense vector, or for `hybrid` the named `dense` vector
    plus the sparse `bm25` vector), then the vector size, then one sampled point's
    `embedding_model`. A point without that field comes from before Phase 8 (MiniLM,
    integer IDs), even if its size happens to match.
    """
    params = client.get_collection(collection_name).config.params
    vectors = params.vectors
    if hybrid:
        if not isinstance(vectors, dict) or DENSE_VECTOR_NAME not in vectors:
            return f"it has no named dense vector {DENSE_VECTOR_NAME!r}"
        if BM25_VECTOR_NAME not in (params.sparse_vectors or {}):
            return f"it has no sparse vector {BM25_VECTOR_NAME!r}"
        dense = vectors[DENSE_VECTOR_NAME]
    else:
        if not isinstance(vectors, VectorParams):
            return "it uses named vectors"
        dense = vectors
    if dense.size != vector_dim:
        return f"vector size {dense.size} differs from the model's {vector_dim}"

    sample, _ = client.scroll(
        collection_name, limit=1, with_payload=["embedding_model"], with_vectors=False
    )
    if sample:
        stored = (sample[0].payload or {}).get("embedding_model")
        if stored != model_name:
            return f"it was built with embedding model {stored!r}, not {model_name!r}"
    return None


def _prepare_collection(
    client: QdrantClient,
    vector_dim: int,
    model_name: str,
    overwrite: bool,
    collection_name: str,
    hybrid: bool = False,
) -> None:
    existing = {c.name for c in client.get_collections().collections}
    if collection_name in existing:
        reason = "--overwrite requested" if overwrite else None
        if reason is None:
            reason = _collection_is_stale(
                client, vector_dim, model_name, collection_name, hybrid
            )
            if reason is not None:
                logger.warning("Recreating collection %r: %s", collection_name, reason)
        if reason is not None:
            client.delete_collection(collection_name)
            existing.discard(collection_name)

    if collection_name not in existing:
        if hybrid:
            client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    DENSE_VECTOR_NAME: VectorParams(
                        size=vector_dim, distance=Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    BM25_VECTOR_NAME: SparseVectorParams(modifier=Modifier.IDF)
                },
            )
        else:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
            )


def _point_vector(
    dense: list[float], embed_text: str, hybrid: bool
) -> list[float] | dict[str, Any]:
    """The point's vector: the dense list, or the named dense plus BM25 document."""
    if not hybrid:
        return dense
    return {
        DENSE_VECTOR_NAME: dense,
        BM25_VECTOR_NAME: Document(text=embed_text, model=BM25_MODEL),
    }


def _delete_paper_points(
    client: QdrantClient, paper_id: str, collection_name: str
) -> None:
    client.delete(
        collection_name=collection_name,
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
    collection_name: str = COLLECTION_NAME,
    hybrid: bool = False,
    chunk_max_tokens: int | None = None,
    merge_peers: bool = True,
) -> int:
    """Chunk, embed and upsert every paper; return the number of chunks indexed.

    `model_name` is stored on every point, and a collection built with another model
    (or another vector size) is recreated. `parsed_dir=None` means no parsed cache, so
    every paper is indexed from its abstract. `tokenizer` defaults to the embedder's own
    (`make_embedder_tokenizer`); it is a parameter so tests need no model download.
    `collection_name` selects the target collection (default `papers`); other
    collections, such as the per-embedder eval ones, are never touched.

    `hybrid=True` (eval only, ADR-0028) writes the hybrid layout instead: a named dense
    vector `dense` plus a sparse vector `bm25` (modifier IDF) that Qdrant computes from
    a `Document` over the chunk's `embed_text` with the `qdrant/bm25` model and its
    default options (the same options must be used at query time). Payloads and point
    IDs are identical to the plain layout. It refuses the app collection `papers`.

    `chunk_max_tokens` and `merge_peers` (Phase 10, T11) change how papers are cut, for
    the chunk-size experiment. They are eval only and refuse the app collection too.
    """
    if hybrid and collection_name == COLLECTION_NAME:
        raise ValueError(
            f"hybrid ingestion is eval only; refusing to rebuild {COLLECTION_NAME!r}"
        )
    if (chunk_max_tokens is not None or not merge_peers) and (
        collection_name == COLLECTION_NAME
    ):
        raise ValueError(
            "re-chunking is eval only; refusing to rebuild "
            f"{COLLECTION_NAME!r} with other chunk settings"
        )
    papers = _load_manifests(papers_dir)

    vector_dim = model.get_sentence_embedding_dimension()
    if vector_dim is None:
        raise ValueError("embedding model does not report its vector dimension")

    _prepare_collection(
        client, vector_dim, model_name, overwrite, collection_name, hybrid
    )

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
        chunks = chunk_paper(
            paper,
            doc,
            tokenizer,
            paper_id=paper_id,
            max_tokens=chunk_max_tokens,
            merge_peers=merge_peers,
        )

        # Delete first so a paper that now has fewer chunks leaves nothing stale.
        _delete_paper_points(client, paper_id, collection_name)
        for start in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[start : start + BATCH_SIZE]
            embeddings = model.encode(
                [chunk.embed_text for chunk in batch], show_progress_bar=False
            )
            points = [
                PointStruct(
                    id=point_id(paper_id, chunk.chunk_index),
                    vector=_point_vector(
                        embedding.tolist(),  # type: ignore[union-attr]
                        chunk.embed_text,
                        hybrid,
                    ),
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
            client.upsert(collection_name=collection_name, points=points)
            total += len(points)

    if parsed_count == 0:
        logger.warning(
            "No parsed papers found in %s: the whole corpus is indexed from "
            "abstracts only. Run scripts/download_papers.py and "
            "scripts/parse_papers.py for full text.",
            parsed_dir,
        )
    return total
