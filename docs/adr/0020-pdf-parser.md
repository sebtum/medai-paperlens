# ADR-0020: Docling as the PDF parser for full-text ingestion

## Status

Accepted (Phase 8 Execute, T2 spike). Parser chosen by the project owner after the
measurements below.

## Context

Phase 8 indexes the full text of 10 arXiv papers. A parse script (T4) turns each PDF
into a cached document (headings, paragraphs, tables with caption, figure captions,
1-based pages). It runs on the host, outside Docker, as an optional `pdf` extra (R1).
The spike below was run against a parser-agnostic cache design. That design was later
replaced by Docling's native format (see "Cache format"). Two candidates were compared in a throwaway Python 3.14.4 venv
(Windows 11, 15.6 GB RAM, CPU only):

- **Docling** (`docling-slim` 2.150.0, `docling-parse` 7.25.0, torch 2.15.0), OCR off.
- **PyMuPDF4LLM** 2.1 (`pymupdf` 2.1, `pymupdf-layout` 2.1), default settings.

Three papers: `conch` (57 pp, unnumbered headings, many figures), `maira_2` (72 pp,
numbered headings, large tables), `llava_med` (17 pp, numbered headings).

## Measurements

| | Docling | PyMuPDF4LLM |
|---|---|---|
| Parse time, `llava_med` / `conch` / `maira_2` | 66 s / 226 s / 160 s | 8–11 s / 18 s / 24 s |
| Peak memory (process + children) | 1.82 GB (`llava_med`), 2.69 GB (`maira_2`); `conch` not measured | 0.31 / 0.36 / 0.39 GB |
| Install (site-packages) | 1,137 MB, plus about 506 MB model weights in the HF cache | 197 MB |
| Headings found, `conch` / `llava_med` / `maira_2` | 29 / 22 / 93 | 28 / 25 / 89 |
| Heading page correct (text found on the stated page) | 100% on all three | 100% on all three |
| Recall against the PDF's own outline | `llava_med` 10/10, `maira_2` 28/29 | `llava_med` 10/10, `maira_2` 28/29 (same miss) |
| Heading levels | all level 1 (flat) | present, but inconsistent (`maira_2`: nearly all level 6, title level 4) |
| Tables | separate items with markdown; caption attached to 17 of 24 (`maira_2`), 5 of 7 (`llava_med`) | `to_markdown`: inline in page text; `to_json`: typed `table` boxes (8 on `llava_med`) |
| Captions | separate `caption` items (52 / 10 / 49) | `to_markdown`: heuristic text matching; `to_json`: typed `caption` boxes (4 on `llava_med`) |
| Text noise | clean | `**bold**` and `<mark>` markup in markdown output |
| Licence | MIT (`docling-slim`, `-parse`, `-ibm-models`, `-core`) | `pymupdf`: AGPL-3.0 or Artifex commercial; `pymupdf4llm` sources: AGPL-3.0; `pymupdf-layout` not verified |
| pip-audit on the dependency tree | clean | clean |

Where the time goes (`llava_med`, 17 pages):

| Run | Total |
|---|---|
| Docling, everything | 71 s |
| Docling, TableFormer off | 24 s (layout model about 1.4 s/page) |
| PyMuPDF4LLM, layout model on | 13 s |
| PyMuPDF4LLM, layout model off | 8 s (5 headings instead of 25, 5 table rows instead of 57) |

TableFormer is about two thirds of Docling's time. Docling's stages overlap, so the
share comes from comparing whole runs, not from adding stage timers. Both parsers use a
neural layout model, and PyMuPDF4LLM's is lighter (about 0.3 s/page). Without it,
PyMuPDF4LLM's heading detection collapses.

**A2 confirmed:** Docling installs and parses end to end on CPython 3.14.4.

**U6 corrected:** `docling-slim[format-pdf,models-local]` is not enough. Table parsing
failed on `import cv2` (TableFormer) and then on `rtree` (layout post-processing). The
working minimal set is `docling-slim[format-pdf,models-local,convert-core]` plus
`opencv-python-headless`. Model weights (layout-heron 164 MB, docling-models 342 MB)
download from Hugging Face on the first parse.

### Method and limits

Headings were scored against the PDFs' own outlines (`conch` has none), and page
numbers by searching each heading's text on its stated page. Tables were compared by eye
on one table. Not measured:

- heading precision (false headings were seen by eye: Docling `CLASSNAME.` and the
  title as a heading; PyMuPDF4LLM `Table 36: …`);
- table accuracy beyond one sample (no TEDS);
- body-text completeness, reading order, header/footer leakage;
- PyMuPDF4LLM's `to_json` route for caption/table linking.

External benchmarks are mixed and mostly vendor-adjacent. They put Docling the same as
or better than PyMuPDF4LLM on tables and PyMuPDF faster. Most predate the PyMuPDF4LLM
2.x layout model. None is a head-to-head on scientific-paper headings. They are
supporting context only.

## Decision

Use **Docling** in `scripts/parse_papers.py`, with OCR off. The `pdf` extra holds
`docling-slim[format-pdf,models-local,convert-core]` and `opencv-python-headless`.

**Cache format (revised in Execute).** The cache is Docling's own `DoclingDocument`,
written with `save_as_json` to `data/parsed/<paper_id>.json`. There is no schema of our
own and no mapping step. The chunker is Docling's HybridChunker (ADR-0021), which works
on a `DoclingDocument`, and the native file is lossless: labels and chunks can be
derived again without re-parsing. The script converts with
`PdfPipelineOptions(do_ocr=False, do_table_structure=True)` and generates no page or
picture images, which keeps the file small (`llava_med`: 994 KB, 47 s). Page headers and
footers land in Docling's furniture layer, and the chunker leaves out footnotes.

The split between host and image:
- `docling` (the parser, models, about 1.1 GB) is host-only, in the `pdf` extra.
- `docling-core[chunking]` (the document model and HybridChunker) is a main dependency,
  because ingestion in Docker loads the cache and chunks it.
- A future parser has to convert into `DoclingDocument`.

The `docling` CLI (`--to json --no-ocr`) could write the same cache. The script stays
because it pins the pipeline options in reviewed code. It also skips papers that are
already parsed: a full run takes about 25 minutes, and one was already killed by memory
pressure. And it prints an install hint when the extra is missing.

The main reasons are the MIT licence against AGPL-3.0, and verified output on three
papers with tables and captions as separate items. It is not a claim that PyMuPDF4LLM is
technically worse: on headings and pages the two matched, and PyMuPDF4LLM is about 7×
lighter on memory and 3–10× faster.

## Alternatives Considered

### PyMuPDF4LLM

Pros:
- About 0.4 GB peak, 8–24 s per paper, 197 MB install.
- `to_json` gives typed boxes (titles, section headers, captions, tables, page
  furniture) with page numbers.

Cons:
- AGPL-3.0 (counted against it in the plan).
- Header levels are unreliable, and markdown output carries markup noise.
- Table and caption linking through `to_json` is unverified here.

### GROBID, pdfplumber

Not evaluated. A 2023 benchmark of academic PDFs found GROBID best for sections
(F1 0.74); a 2025 benchmark found pdfplumber best on table structure. GROBID runs as a
Java service.

## Consequences

Positive:
- MIT-licensed parser. Its native document is the cache, and Docling's chunker
  consumes it directly.

Negative:
- About 2.7 GB peak memory per large paper and roughly 25 minutes for all 10 papers
  (estimated from the three measured); run it with memory free. A background run was
  already killed once by system-wide memory pressure.
- The `pdf` extra adds about 1.1 GB plus 0.5 GB of weights on the host. It stays out of
  CI and the Docker image (`docling-core` is in both).
- Heading levels are flat, so the section hierarchy comes from numbering and names (U7),
  applied by the chunker (ADR-0021). False headings are left for T12 to inspect: "Caption:"
  and text inside figures on `llava_med`. A title reported as a heading is already
  handled in the chunker.
- Most figure captions are not linked to their picture (1 of 7 on `llava_med`).
- Windows prints a Hugging Face symlink warning (harmless;
  `HF_HUB_DISABLE_SYMLINKS_WARNING=1` silences it).

## Follow-up

- T4: add the extra, write the cache with `save_as_json`, and run pip-audit with the
  extra installed (R8).
- T12: inspect one parsed file per paper (heading precision, pages, leaked headers).
- Revisit PyMuPDF4LLM's `to_json` if memory or runtime becomes a problem and the
  AGPL is acceptable.
