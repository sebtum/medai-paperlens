import argparse
import logging
import os
import sys
from pathlib import Path

import dotenv

from app.retrieval.client import get_client
from app.retrieval.embedding import get_model, get_model_name
from app.retrieval.ingestion import COLLECTION_NAME, ingest

# Read `.env` like the API does, so both use the same EMBEDDING_MODEL and QDRANT_URL.
dotenv.load_dotenv()
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest papers into Qdrant.")
    parser.add_argument(
        "--papers-dir",
        default=os.getenv(
            "PAPERS_DIR",
            str(Path(__file__).parent.parent / "data" / "papers"),
        ),
        help="Directory containing paper JSON files (env: PAPERS_DIR).",
    )
    parser.add_argument(
        "--parsed-dir",
        default=os.getenv(
            "PARSED_DIR",
            str(Path(__file__).parent.parent / "data" / "parsed"),
        ),
        help=(
            "Directory with parsed papers from scripts/parse_papers.py; papers "
            "without one are indexed from their abstract (env: PARSED_DIR)."
        ),
    )
    parser.add_argument(
        "--collection",
        default=COLLECTION_NAME,
        help=(
            "Qdrant collection to ingest into (default: %(default)s, the app "
            "collection). Use e.g. eval_<model-slug> for evaluation embedders."
        ),
    )
    parser.add_argument(
        "--hybrid",
        action="store_true",
        help=(
            "Eval only (ADR-0028): write a named dense vector plus a sparse BM25 "
            "vector computed by Qdrant. Needs a non-default --collection, e.g. "
            "eval_hybrid; the app collection is never written this way."
        ),
    )
    parser.add_argument(
        "--chunk-tokens",
        type=int,
        help=(
            "Eval only (Phase 10, T11): cap chunks at this many tokens (embed text "
            "including the title line). Needs a non-default --collection."
        ),
    )
    parser.add_argument(
        "--no-merge-peers",
        action="store_true",
        help=(
            "Eval only (Phase 10, T11): keep Docling's pieces apart instead of "
            "merging neighbours under one heading. Needs a non-default --collection."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Drop and recreate the collection before ingesting.",
    )
    args = parser.parse_args()

    papers_dir = Path(args.papers_dir)
    if not papers_dir.is_dir():
        logger.error("Papers directory not found: %s", papers_dir)
        sys.exit(1)

    if args.hybrid and args.collection == COLLECTION_NAME:
        logger.error(
            "--hybrid is eval only: pass --collection eval_hybrid, not %r.",
            COLLECTION_NAME,
        )
        sys.exit(1)

    if (args.chunk_tokens is not None or args.no_merge_peers) and (
        args.collection == COLLECTION_NAME
    ):
        logger.error(
            "--chunk-tokens and --no-merge-peers are eval only: pass --collection "
            "eval_..., not %r.",
            COLLECTION_NAME,
        )
        sys.exit(1)

    if args.overwrite:
        logger.warning(
            "--overwrite will DELETE the existing Qdrant collection %r "
            "and re-index from scratch. "
            "Do not use in production unless you intend a full re-index.",
            args.collection,
        )

    try:
        count = ingest(
            papers_dir,
            get_client(),
            get_model(),
            model_name=get_model_name(),
            parsed_dir=Path(args.parsed_dir),
            overwrite=args.overwrite,
            collection_name=args.collection,
            hybrid=args.hybrid,
            chunk_max_tokens=args.chunk_tokens,
            merge_peers=not args.no_merge_peers,
        )
        logger.info("Ingested %d chunks.", count)
    except Exception as exc:
        logger.error("Ingestion failed: %s", exc)
        sys.exit(1)
