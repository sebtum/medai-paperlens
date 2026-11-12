"""Parsed-paper cache: Docling's own `DoclingDocument` JSON (ADR-0020, ADR-0021).

`scripts/parse_papers.py` (host only, optional `pdf` extra) converts each PDF with
Docling and writes `<paper_id>.json` with `DoclingDocument.save_as_json`. Ingestion
reads it back through `load_parsed`; that needs only `docling-core`, never the parser.
We use Docling's native format instead of a schema of our own, because the chunker
(Docling's HybridChunker) works on a `DoclingDocument` anyway and the native file
is lossless. A missing or unusable file is never fatal: the caller falls back to the
manifest abstract.
"""

import logging
from pathlib import Path

from docling_core.types.doc import DoclingDocument

logger = logging.getLogger(__name__)


def load_parsed(parsed_dir: Path, paper_id: str) -> DoclingDocument | None:
    """Load `<parsed_dir>/<paper_id>.json`; `None` if absent or unusable.

    A missing file is the normal "not parsed yet" case and is silent. Unreadable
    files, invalid JSON, a document that does not validate, or a document whose
    `name` differs from `paper_id` (Docling names it after the PDF's stem) return
    `None` and log a warning, so one bad file cannot stop an ingest run. Images
    referenced by local paths are never loaded.
    """
    if Path(paper_id).name != paper_id or paper_id in {"", ".", ".."}:
        logger.warning("Refusing parsed cache for invalid paper_id %r", paper_id)
        return None

    path = parsed_dir / f"{paper_id}.json"
    if not path.is_file():
        return None

    try:
        doc = DoclingDocument.load_from_json(path)
    except (OSError, ValueError) as exc:
        # JSON syntax errors, UnicodeDecodeError and pydantic's ValidationError are
        # all ValueErrors.
        logger.warning("Ignoring unusable parsed cache %s: %s", path, exc)
        return None

    if doc.name != paper_id:
        logger.warning(
            "Ignoring parsed cache %s: document name %r does not match file name",
            path,
            doc.name,
        )
        return None
    return doc
