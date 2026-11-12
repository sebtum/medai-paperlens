"""Parse the corpus PDFs (``data/pdfs/``) into Docling documents (``data/parsed/``).

Host-only step (ADR-0020): needs the optional ``pdf`` extra, which is never installed
in CI or the Docker image. Each PDF becomes ``data/parsed/<paper_id>.json``, Docling's
native ``DoclingDocument`` JSON (``save_as_json``); ingestion reads it back with
``app.retrieval.parsed.load_parsed``. Existing files are skipped unless ``--force``.
Exits 1 if the parser is missing or any paper fails.

Parsing is slow (about 25 minutes for all 10 papers) and peaks near 2.7 GB of memory
on the largest paper, so run it with memory free.
"""

import argparse
import logging
import os
import sys
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent

INSTALL_HINT = (
    "The PDF parser is not installed. Install the optional 'pdf' extra into the venv "
    "(do not use `uv run`/`uv sync` on Windows):\n"
    '  .\\.venv\\Scripts\\pip.exe install "docling-slim[format-pdf,models-local,'
    'convert-core]" opencv-python-headless\n'
    "or, equivalently, install the extra declared in pyproject.toml."
)

# A converter turns one PDF into a document that can be saved with ``save_as_json``.
Converter = Callable[[Path], Any]


def build_converter() -> Converter:
    """Create the Docling converter. Imported lazily: the extra may be absent.

    OCR is off (the corpus PDFs have a text layer), table structure is on, and no
    page or picture images are generated, which keeps the cache small.
    """
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    options = PdfPipelineOptions(
        do_ocr=False,
        do_table_structure=True,
        generate_page_images=False,
        generate_picture_images=False,
    )
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )

    def convert(pdf_path: Path) -> Any:
        return converter.convert(pdf_path).document

    return convert


def parse_papers(
    pdf_dir: Path,
    parsed_dir: Path,
    convert: Converter,
    *,
    only: Iterable[str] = (),
    force: bool = False,
) -> int:
    """Parse each PDF into ``parsed_dir``; return the number of failures.

    ``only`` restricts the run to those paper ids; an id without a PDF counts as a
    failure. The document's ``name`` is set to the paper id, which is what
    ``load_parsed`` checks. Files are written atomically.
    """
    parsed_dir.mkdir(parents=True, exist_ok=True)
    wanted = list(dict.fromkeys(only))
    available = {path.stem: path for path in sorted(pdf_dir.glob("*.pdf"))}

    failures = 0
    for paper_id in wanted:
        if paper_id not in available:
            logger.error("%s: no PDF found in %s", paper_id, pdf_dir)
            failures += 1

    for paper_id, pdf_path in available.items():
        if wanted and paper_id not in wanted:
            continue
        target = parsed_dir / f"{paper_id}.json"
        if target.exists() and not force:
            logger.info("%s: already parsed, skipping", paper_id)
            continue
        tmp = target.with_suffix(".json.part")
        try:
            logger.info("%s: parsing %s", paper_id, pdf_path.name)
            doc = convert(pdf_path)
            doc.name = paper_id
            doc.save_as_json(tmp)
            tmp.replace(target)
        except Exception as exc:  # one bad paper must not stop the others
            logger.error(
                "%s: parsing failed (%s: %s)", paper_id, type(exc).__name__, exc
            )
            tmp.unlink(missing_ok=True)
            failures += 1
            continue
        logger.info("%s: wrote %s", paper_id, target)
    return failures


def main() -> None:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    parser = argparse.ArgumentParser(description="Parse corpus PDFs with Docling.")
    parser.add_argument(
        "--pdf-dir",
        default=os.getenv("PDF_DIR", str(ROOT / "data" / "pdfs")),
        help="Directory containing the PDFs (env: PDF_DIR).",
    )
    parser.add_argument(
        "--parsed-dir",
        default=os.getenv("PARSED_DIR", str(ROOT / "data" / "parsed")),
        help="Directory to write the parsed documents to (env: PARSED_DIR).",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        metavar="PAPER_ID",
        help="Parse only this paper id (repeatable).",
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-parse papers that are already parsed."
    )
    args = parser.parse_args()

    pdf_dir = Path(args.pdf_dir)
    if not pdf_dir.is_dir():
        logger.error(
            "PDF directory not found: %s (run scripts/download_papers.py first)",
            pdf_dir,
        )
        sys.exit(1)

    try:
        convert = build_converter()
    except ImportError:
        logger.error(INSTALL_HINT)
        sys.exit(1)

    failures = parse_papers(
        pdf_dir, Path(args.parsed_dir), convert, only=args.only, force=args.force
    )
    if failures:
        logger.error("%d paper(s) failed to parse.", failures)
        sys.exit(1)


if __name__ == "__main__":
    main()
