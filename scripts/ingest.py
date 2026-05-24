import argparse
import logging
import os
import sys
from pathlib import Path

from app.retrieval.client import get_client
from app.retrieval.embedding import get_model
from app.retrieval.ingestion import ingest

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
        "--overwrite",
        action="store_true",
        help="Drop and recreate the collection before ingesting.",
    )
    args = parser.parse_args()

    papers_dir = Path(args.papers_dir)
    if not papers_dir.is_dir():
        logger.error("Papers directory not found: %s", papers_dir)
        sys.exit(1)

    if args.overwrite:
        logger.warning(
            "--overwrite will DELETE the existing Qdrant collection "
            "and re-index from scratch. "
            "Do not use in production unless you intend a full re-index."
        )

    try:
        count = ingest(papers_dir, get_client(), get_model(), overwrite=args.overwrite)
        logger.info("Ingested %d papers.", count)
    except Exception as exc:
        logger.error("Ingestion failed: %s", exc)
        sys.exit(1)
