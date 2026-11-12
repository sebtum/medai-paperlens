"""Download the corpus PDFs listed in the paper manifests (``data/papers/*.json``).

PDFs land in ``data/pdfs/<paper_id>.pdf`` (gitignored). Existing files are
skipped unless ``--force`` is given. Exits 1 if any download fails.
"""

import argparse
import json
import logging
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
PDF_MAGIC = b"%PDF-"
REQUEST_DELAY_SECONDS = 3.0  # arXiv asks for at most one request every 3 s
USER_AGENT = "medai-paperlens-corpus-downloader/0.1"


def _download_one(client: httpx.Client, url: str, target: Path) -> None:
    """Fetch ``url`` and write it atomically to ``target`` after a magic-byte check."""
    response = client.get(url, follow_redirects=True)
    response.raise_for_status()
    content = response.content
    if not content.startswith(PDF_MAGIC):
        raise ValueError(f"response is not a PDF (missing {PDF_MAGIC!r} header)")
    tmp = target.with_suffix(".pdf.part")
    tmp.write_bytes(content)
    tmp.replace(target)


def download_papers(
    papers_dir: Path,
    pdf_dir: Path,
    client: httpx.Client,
    *,
    force: bool = False,
    delay: float = REQUEST_DELAY_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    """Download every manifest's ``pdf_url``; return the number of failures."""
    pdf_dir.mkdir(parents=True, exist_ok=True)
    failures = 0
    requested = False
    for manifest_path in sorted(papers_dir.glob("*.json")):
        paper_id = manifest_path.stem
        target = pdf_dir / f"{paper_id}.pdf"
        if target.exists() and not force:
            logger.info("%s: already downloaded, skipping", paper_id)
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            url = manifest["pdf_url"]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            logger.error("%s: manifest has no usable pdf_url (%s)", paper_id, exc)
            failures += 1
            continue
        if requested:
            sleep(delay)
        requested = True
        try:
            _download_one(client, url, target)
        except (httpx.HTTPError, ValueError, OSError) as exc:
            logger.error("%s: download failed (%s)", paper_id, exc)
            failures += 1
            continue
        logger.info("%s: downloaded %s", paper_id, url)
    return failures


def main() -> None:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    parser = argparse.ArgumentParser(description="Download corpus PDFs.")
    parser.add_argument(
        "--papers-dir",
        default=os.getenv("PAPERS_DIR", str(ROOT / "data" / "papers")),
        help="Directory containing paper manifests (env: PAPERS_DIR).",
    )
    parser.add_argument(
        "--pdf-dir",
        default=os.getenv("PDF_DIR", str(ROOT / "data" / "pdfs")),
        help="Directory to write PDFs to (env: PDF_DIR).",
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-download files that already exist."
    )
    args = parser.parse_args()

    papers_dir = Path(args.papers_dir)
    if not papers_dir.is_dir():
        logger.error("Papers directory not found: %s", papers_dir)
        sys.exit(1)

    with httpx.Client(
        headers={"User-Agent": USER_AGENT}, timeout=httpx.Timeout(60.0)
    ) as client:
        failures = download_papers(
            papers_dir, Path(args.pdf_dir), client, force=args.force
        )
    if failures:
        logger.error("%d download(s) failed.", failures)
        sys.exit(1)


if __name__ == "__main__":
    main()
