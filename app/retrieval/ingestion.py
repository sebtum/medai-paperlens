import json
import logging
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

COLLECTION_NAME = "papers"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
VECTOR_DIM = 384
BATCH_SIZE = 32


def ingest(
    papers_dir: Path,
    client: QdrantClient,
    model: SentenceTransformer,
    *,
    overwrite: bool = False,
) -> int:
    papers = []
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

        papers.append(paper)

    existing = {c.name for c in client.get_collections().collections}
    if overwrite and COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
        existing.discard(COLLECTION_NAME)

    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        )

    total = 0
    for batch_start in range(0, len(papers), BATCH_SIZE):
        batch = papers[batch_start : batch_start + BATCH_SIZE]
        texts = [p["abstract"] for p in batch]
        embeddings = model.encode(texts, show_progress_bar=False)

        points = [
            PointStruct(
                id=batch_start + j,
                vector=embedding.tolist(),  # type: ignore[union-attr]
                payload={
                    "title": paper.get("title", ""),
                    "authors": paper.get("authors", []),
                    "year": paper.get("year", 0),
                    "doi": paper.get("doi"),
                    "text": paper["abstract"],
                },
            )
            for j, (paper, embedding) in enumerate(zip(batch, embeddings, strict=True))
        ]

        client.upsert(collection_name=COLLECTION_NAME, points=points)
        total += len(points)

    return total
