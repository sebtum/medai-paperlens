# Phase 8: Full-text ingestion — Plan

## Decisions (resolving U2, U3, U5, U6, U7)
- **U2 preprints.** `conch` → arXiv 2307.12914 (v2, Jul 2023). It has the same title and
  author list as our manifest; the Nature Medicine version (2024) came later.
  `cxr_vlm_generation` → arXiv 2211.12737 v1 (Nov 2022, "RoentGen"). That is an earlier,
  differently titled preprint of the Nature BME 2024 paper, whose abstract our manifest
  holds. Both get a `source_note` in the manifest: the full text is from the preprint and
  may differ from the published version. The manifest abstract stays as it is (chunk 0).
  arXiv IDs are pinned with a version (`2307.12914v2`).
- **U6 Docling deps.** `docling` = `docling-slim[standard]`. That pulls in torch (already
  installed), torchvision, docling-parse, docling-ibm-models, rapidocr, scipy, accelerate
  and the office-format libraries. The spike first tries the slimmer
  `docling-slim[format-pdf,models-local]` with `do_ocr=False`. The layout model and
  TableFormer weights (a few hundred MB) download into the host HF cache on the first parse.
  Optional extra `pdf` in `pyproject.toml`. CI (`uv sync`) and the Dockerfile
  (`uv sync --frozen --no-dev`) install no extras, so the parser is only ever on the host.
  Because CI never installs the extra, pip-audit for it runs **locally with the extra
  installed** (R8).
- **U3 Chunking start values (revised in Execute: Docling HybridChunker, owner decision).**
  Each chunk is embedded as `"{manifest title}\n{heading path}\n{body}"`, Docling's
  `contextualize` layout; the payload `text` is the body alone. The whole string must be
  ≤ `max_seq_length − special tokens` (510) as counted by the embedder's own tokenizer, so
  nothing gets truncated. HybridChunker splits by document item, then text (semchunk), and
  merges undersized neighbours with the same headings. There is **no body target and no
  overlap** (HybridChunker has neither); chunks fill up to the budget. Phase 9 tunes the
  size and measures overlap.
- **U7 Section label = path, depth ≤ 2** (user decision): `"Methods > Datasets"`. Numbering
  is stripped, whitespace collapsed, and ALL-CAPS headings are title-cased. The hierarchy
  comes from heading numbers (`3.2` sits under `3`; `A.1` under `A`). For unnumbered
  headings: a canonical top-level name (Abstract, Introduction, Background, Related work,
  Method(s), Materials and methods, Experiments, Results, Discussion, Conclusion(s),
  Limitations, Appendix) starts a new top level; any other unnumbered heading becomes a
  child of the current top level. Deeper headings map to their depth-2 ancestor. Abstract
  chunk → `"Abstract"`. The cache keeps the raw heading text, so labels can be derived
  again later. *Execute:* the rules are applied by rewriting Docling's heading levels, so
  HybridChunker builds the path. After the back matter, a dotless appendix letter
  (`A Data`) counts as numbering (found on `llava_med`).
- **Chunking (ADR-0021, revised).** Docling's HybridChunker does the splitting, table
  handling (compact markdown with the caption; a table over the limit is split by rows
  with the header row repeated) and sizing. Page headers and footers are dropped by
  Docling's furniture layer, and footnotes by a serializer setting. Kept custom, with
  reasons in ADR-0021:
  - the manifest abstract as chunk 0 (the parsed abstract is dropped);
  - heading levels;
  - dropping front matter and everything under References, Bibliography,
    Acknowledg(e)ments, Author contributions, Competing interests and Funding (appendices
    after the references are kept);
  - the section label;
  - the token budget;
  - `chunk_type`: picture-only chunks → `caption`; unlinked captions count as `text`.

  Tables and captions stay in reading order. `page` = page of the chunk's first item.
  `chunk_index` runs 0..n.
- **Identity (ADR-0021).** Point ID = `uuid5(PAPERLENS_NS, f"{paper_id}:{chunk_index}")`.
  Ingest deletes a paper's points by a `paper_id` filter and then upserts. Re-runs are
  idempotent and no stale chunks remain when a paper's chunk count shrinks. Payload: `title,
  authors, year, doi, paper_id, section, page, chunk_index, chunk_type, text,
  embedding_model`. If the existing collection's vector size or a sampled point's
  `embedding_model` doesn't match the current model, ingest logs a warning and recreates
  the collection. This covers the old MiniLM volumes, which have the same 384 dim and int
  IDs, so `docker compose up` heals them on its own.
- **`Citation` (ADR-0021)**: new fields, all optional: `paper_id: str | None = None`,
  `section: str | None = None`, `page: int | None = None`, `chunk_index: int | None = None`.
  `excerpt` = chunk body. One citation per chunk.
- **Embedder (ADR-0022).** Default `BAAI/bge-small-en-v1.5` via `EMBEDDING_MODEL`. The
  dimension comes from `model.get_sentence_embedding_dimension()` and the chunk limit
  from `model.max_seq_length`; `VECTOR_DIM` is removed. No query prefix (Phase 9). In
  Compose, `ingest` and `api` read the same `EMBEDDING_MODEL` variable, so the two can't
  drift apart.
- **U5 context (user decision).** New env var `OLLAMA_NUM_CTX` (default 8192), sent as
  `options.num_ctx` on every Ollama call, **including warmup**. A different `num_ctx` would
  make Ollama reload the model on the first query. Guard: `make_generate_node(llm, *,
  evidence_budget_tokens=4000)` estimates the size (chars/3, conservative) and drops whole
  lowest-ranked evidence items with a warning. Never truncates. 5 × 512 tokens stays under
  the budget, so the guard is only a safety net.
- **Parsed cache (revised in Execute)**: Docling's native `DoclingDocument` JSON
  (`save_as_json`), `data/parsed/<paper_id>.json`; `doc.name` = `paper_id` (the PDF
  stem). An invalid or missing cache file → warning, then abstract-only.
  `docling-core[chunking]` is a main dependency (ingest in Docker loads it and chunks).
  `docling`, the parser, stays in the host-only `pdf` extra.

## Task graph
G1: T1 ‖ T3 ‖ T6 ‖ T8 → G2: T2 ‖ T5 ‖ T9 ‖ T10 → G3: T4 ‖ T7 → G4: T11 → G5: T12 → G6: T13

T6 and T8 both touch `tests/test_retrieval.py`: when using subagents, run T6 first.
T6 also touches `ingestion.py` before T7 rewrites it.

## Tasks

### [x] T1: Manifest, download script, ignore files
- Files: `data/papers/*.json`, `scripts/download_papers.py` (new), `.gitignore`, `.dockerignore`, `data/parsed/.gitkeep`, `tests/test_download.py` (new)
- Depends on: —
- Group: G1
- Acceptance criteria:
  - [x] All 10 manifests have a versioned `arxiv_id` and a `pdf_url`; `conch` and `cxr_vlm_generation` have a `source_note` (U2)
  - [x] The script downloads to `data/pdfs/<paper_id>.pdf`, skips files that already exist (`--force` re-downloads), waits 3 s between requests, sends a generic User-Agent, checks for the `%PDF-` magic bytes, writes atomically, and exits 1 if any download fails
  - [x] Tested with `httpx.MockTransport` (skip-existing, bad magic bytes → failure), no network
  - [x] `data/pdfs/` and `data/parsed/*` are gitignored (`data/parsed/.gitkeep` stays tracked); both are listed in `.dockerignore` (R1)
- Note: Done. arXiv IDs/versions verified against the arXiv API (titles match; `cxr_vlm_generation` resolves to RoentGen v1). `.dockerignore` lists `data/pdfs/` and `data/parsed/` (`.gitkeep` is excluded from the image too; the dir is mounted at runtime in T11). 8 tests, no network; script not yet run against live arXiv (that happens in T12).

### [x] T2: Parser spike → ADR-0020
- Files: scratch only (a throwaway Python 3.14 venv and the spike script), `docs/adr/0020-pdf-parser.md`
- Depends on: T1 (PDFs)
- Group: G2
- Acceptance criteria:
  - [x] Docling vs PyMuPDF4LLM on `conch` (unnumbered headings, many figures), `maira_2` (numbered, large tables) and one more (`llava_med`): heading recall against a manual check, page-number correctness, table markdown quality, figure captions as separate items, s/paper, install size
  - [x] Confirms or refutes A2 (Docling on cp314). If refuted, or if neither parser is usable → **stop and re-plan**
  - [x] pip-audit of the chosen parser's dependency tree; the slim Docling extras checked (U6)
  - [x] ADR-0020 records the measured numbers; AGPL counts against PyMuPDF
- Note: Done. Docling chosen by the owner (A2 confirmed on 3.14.4). Departures from the plan:
  (1) the slim extras in U6 are not enough: the working set is
  `docling-slim[format-pdf,models-local,convert-core]` + `opencv-python-headless` (T4 must use it);
  (2) "heading recall against a manual check" was scored against the PDF outlines (`llava_med`,
  `maira_2`; `conch` has none), not hand labels; heading precision and table accuracy were not
  measured (limits listed in ADR-0020);
  (3) PyMuPDF4LLM 2.x has its own neural layout model and a `to_json` route that was not fully
  evaluated; the choice rests mainly on licence (MIT vs AGPL) and verified output, not on
  PyMuPDF4LLM being worse. Docling peaks at 2.7 GB on a 72-page paper, so run T4 with memory free.
  Spike scripts stayed in scratch (not committed; `ruff check .` would lint them).

### [x] T3: Parsed-cache schema + loader
- Files: `app/retrieval/parsed.py` (new), `tests/test_parsed.py` (new), `tests/fixtures/parsed/*.json` (new)
- Depends on: —
- Group: G1
- Acceptance criteria:
  - [x] Pydantic `ParsedBlock` and `ParsedPaper` as in Decisions; `load_parsed(parsed_dir, paper_id) -> ParsedPaper | None`
  - [x] Missing file → `None`; invalid JSON, wrong schema or `paper_id` mismatch → `None` + warning
  - [x] Small hand-written fixtures; mypy clean
- Note: Done. **Reworked in Execute (owner decision: Docling instead of custom).** `ParsedBlock`/`ParsedPaper` and the hand-written fixtures are gone. `load_parsed(parsed_dir, paper_id) -> DoclingDocument | None` uses `DoclingDocument.load_from_json`; local images are never loaded. The behaviour is kept: a path-traversal `paper_id` is refused with a warning; a missing file is silent; unreadable, invalid JSON or a failed validation → `None` + warning; `doc.name != paper_id` → `None` + warning. 14 tests against JSON written by `save_as_json` from documents built in code (`tests/docling_builders.py`).

### [x] T4: Parse script + `pdf` extra
- Files: `scripts/parse_papers.py` (new), `pyproject.toml`, `uv.lock`
- Depends on: T2, T3
- Group: G3
- Acceptance criteria:
  - [x] Optional extra `pdf` holds only the chosen parser; lock with `uv lock` only, install it with `.venv\Scripts\pip.exe` (never `uv run` / `uv sync`)
  - [x] Converts with `PdfPipelineOptions(do_ocr=False, do_table_structure=True, generate_page_images=False, generate_picture_images=False)` and writes `doc.save_as_json(...)`: no mapping (revised: the cache is the native `DoclingDocument`). The extra's `docling` must resolve to the same `docling-core` as the main dependency (one lock).
  - [x] Imports the parser lazily; if it's missing, prints a clear install hint and exits 1
  - [x] Writes `data/parsed/<paper_id>.json`; `--only <id>`; skips existing files unless `--force`
  - [x] `pip-audit` clean with the extra installed (R8)
- Note: Done.
  - `scripts/parse_papers.py`: `parse_papers(pdf_dir, parsed_dir, convert, *, only, force)` takes the converter as a parameter, so the 6 tests in `tests/test_parse_papers.py` run with a fake and need no Docling. `build_converter()` imports Docling lazily; if it is missing, `main()` logs the install hint and exits 1. It sets `doc.name = paper_id` before saving, because `load_parsed` checks that. Writes go through a `.json.part` file and are renamed. One failing paper is logged and counted but does not stop the others (exit 1 at the end). `--only` is repeatable; an id without a PDF counts as a failure. Defaults: `--pdf-dir` / `PDF_DIR` = `data/pdfs`, `--parsed-dir` / `PARSED_DIR` = `data/parsed`.
  - `pdf` extra: `docling-slim[format-pdf,models-local,convert-core]>=2.150.0` and `opencv-python-headless>=4.10.0` (the working set from the T2 spike). `uv lock` added only the new packages (docling-slim, docling-parse, docling-ibm-models, torchvision, accelerate, opencv, rtree, pypdfium2, psutil, filetype, langcodes); `docling-core` stays 2.106.0 and torch 2.14.0, so there is one lock. Installed into the venv with `pip install` of the two pinned requirements (not `uv run` / `uv sync`).
  - Install took over an hour on this connection (the opencv wheel downloaded at about 13 KB/s); `pip check` is clean.
  - Smoke test: `llava_med` parsed in 62 s into a 1.0 MB file. `load_parsed` accepts it, and chunking with bge-small gives 38 chunks (28 text, 10 table), longest 502 tokens, labels as in T5's note. The model weights were already in the HF cache from the spike.
  - pip-audit with the extra installed: no known vulnerabilities. ruff, mypy and bandit clean; 166 tests pass.
  - The other 9 papers are not parsed yet; that is T12 (about 25 minutes, run with memory free).

### [x] T5: Structure-aware chunker
- Files: `app/retrieval/chunking.py` (new), `tests/test_chunking.py` (new)
- Depends on: T3
- Group: G2
- Acceptance criteria:
  - [x] `chunk_paper(manifest, parsed | None, count_tokens, max_tokens) -> list[Chunk]`; the token counter is injected (tests use a whitespace counter, R7)
  - [x] Abstract = chunk 0, `section="Abstract"`; `parsed=None` → abstract-only
  - [x] Section labels follow U7: numbered hierarchy, unnumbered canonical names, depth cap, normalisation
  - [x] References and the other dropped sections are excluded; an appendix after the references is kept
  - [x] Every chunk (header + body) ≤ `max_tokens`; overlap only within a section; no text lost (except dropped sections)
  - [x] A table over the limit is split by rows with the caption and header repeated; captions → `caption` chunks; every chunk has `paper_id, section, page, chunk_index, chunk_type` (R2)
- Note: Done. Departures and choices, also recorded in ADR-0021:
  (1) `chunk_paper(manifest, parsed, count_tokens, max_tokens, *, paper_id, target_tokens=350, overlap_tokens=50, min_tail_tokens=None)`: `paper_id` is a required keyword because the manifest has no id field and an abstract-only paper has no `parsed`. It returns frozen `Chunk(paper_id, chunk_index, chunk_type, section, page, text, embed_text)`; `embed_text` is the exact `"{title}. {section}.
{body}"` string, so T7 embeds it as is. T7 must pass a counter that counts the full string the way the embedder sees it (include special tokens).
  (2) Tables and figure captions of a section are emitted after that section's text chunks, not at their position in the text, so a figure in the middle of a paragraph doesn't cut the text into stubs. `chunk_index` still runs 0..n in that emission order.
  (3) After a dropped section, indexing resumes at the next heading whose depth is ≤ the dropped heading's (so a dropped `5.2 Funding` doesn't swallow `5.3 Limitations`; an unnumbered `References` is only left at a canonical name like `Appendix` or a numbered top-level heading). `Abstract` is a dropped section too.
  (4) Bare `A Heading` is not numbering (it would eat the article in "A Comparison of Models"); `A.`, `A.1`, `IV.` and `3.2` are. Parser heading levels are ignored; the label comes from numbers and names only (U7).
  (5) Text is cut at the body target (350), tables at the hard embedder limit. Tail merge counts only the tail's new text (overlap excluded) and uses the hard limit. `page` is the page of the first sentence in the chunk, including a repeated overlap sentence.
  (6) A header (title + section) that leaves no room raises `ValueError`.
  Not tested against real parser output: `data/parsed/` is still empty (T4 / T12). 59 tests, incl. a seeded randomized limit/no-loss check; 174 in the suite.
- **Reworked in Execute (owner decision): Docling HybridChunker instead of the custom chunker.** Notes (1)–(6) above describe the first version and no longer apply. The new version is in ADR-0021.
  - Signature: `chunk_paper(manifest, doc: DoclingDocument | None, tokenizer: BaseTokenizer, *, paper_id) -> list[Chunk]`. `Chunk` is unchanged. `make_embedder_tokenizer(model)` gives a `HuggingFaceTokenizer` with budget `max_seq_length − special tokens`; T7 passes it, and its tests pass a fake.
  - Dependency: `docling-core[chunking]>=2.106,<3` in the main dependencies. Locked at 2.106.0, with semchunk 3.2.5 and tree-sitter 0.26.1 plus grammars (cp314 / abi3 wheels for win_amd64 and manylinux x86_64). Installed with `pip --no-deps`. pip-audit: nothing new; it flags a stray `gitpython` that is in the local venv but not in the lock.
  - Removed: the overlap, the 350 body target and the deferred floats.
  - Findings while building it:
    - `DocMeta.doc_items` comes back as base `DocItem`, so `chunk_type` uses labels.
    - Footnotes are *not* skipped by Docling's default serializer, so the `FOOTNOTE` label is excluded via params.
  - Real-output smoke test on `llava_med` (parsed in the T2 spike venv, 47 s, 994 KB JSON; chunked with bge-small): 38 chunks (28 text, 10 table, 0 caption), longest 502 tokens including special tokens (≤ 512), median 371. No reference text. Labels match the outline (Introduction … Conclusions, Data, Prompts).
    - The smoke test found the dotless appendix headings `A Data` / `B Prompts`. Both chunkers dropped them under References; that is now fixed and tested.
    - False headings from Docling ("Caption", "Visual input example, Biomedical image from [6]") become labels; left for T12.
    - Only 1 of 7 pictures has a linked caption; the other captions are indexed as `text`.
  - 30 tests (whitespace tokenizer + an offline WordLevel tokenizer, no download); 139 in the suite.

### [x] T6: Default embedder bge-small, dimension from the model
- Files: `app/retrieval/embedding.py`, `app/retrieval/ingestion.py`, `tests/test_retrieval.py`, `.env.example`
- Depends on: —
- Group: G1
- Acceptance criteria:
  - [x] `EMBEDDING_MODEL` default `BAAI/bge-small-en-v1.5`, defined in `embedding.py`; `get_model_name()` helper
  - [x] `VECTOR_DIM` removed; the collection is created with `model.get_sentence_embedding_dimension()`
  - [x] Tests mock the dimension (no model download); `EMBEDDING_MODEL` is in `.env.example`
- Note: Done. `DEFAULT_EMBEDDING_MODEL` and `get_model_name()` live in `embedding.py` (an empty `EMBEDDING_MODEL` also falls back to the default); `ingestion.py` no longer defines a model name. `ingest()` raises `ValueError` if the model reports no dimension. Departure from the file list: removing `VECTOR_DIM` broke the imports in `test_workflow.py`, `test_query.py`, `test_query_stream.py`, so those tests and `test_retrieval.py` now use `TEST_VECTOR_DIM = 384` from `tests/conftest.py` (mock embedder size only). README env example updated too. `model_name`/`embedding_model` payload handling is still T7. 110 tests pass.

### [x] T7: Ingest from parsed cache
- Files: `app/retrieval/ingestion.py`, `scripts/ingest.py`, `tests/test_retrieval.py`
- Depends on: T3, T5, T6
- Group: G3
- Acceptance criteria:
  - [x] `ingest(papers_dir, client, model, *, parsed_dir, model_name, overwrite) -> int` (number of chunks); `--parsed-dir` / `PARSED_DIR`
  - [x] uuid5 IDs, delete by `paper_id` filter then upsert, payload as in Decisions (R3)
  - [x] Collection recreated on a dim or `embedding_model` mismatch; `--overwrite` still drops it
  - [x] Paper without a cache file → abstract only plus a per-paper warning; no cache files at all → one summary warning (R4)
  - [x] Tests: idempotent re-run (same IDs), a shrinking paper leaves no stale points, mismatch → recreate, fallback path; all with mocks
  - [x] Chunks with `chunk_paper(manifest, load_parsed(...), make_embedder_tokenizer(model), paper_id=...)` and embeds `chunk.embed_text`; tests inject a fake tokenizer (the `WhitespaceTokenizer` from `tests/docling_builders.py`) so no model is loaded
- Note: Done. `ingest(papers_dir, client, model, *, model_name, parsed_dir=None, overwrite=False, tokenizer=None) -> int` (chunks indexed). Choices beyond the criteria:
  - `parsed_dir=None` behaves like an empty cache (abstract-only, one summary warning). `tokenizer` defaults to `make_embedder_tokenizer(model)` and is a parameter so tests inject `WhitespaceTokenizer`; it is built only when there is at least one valid manifest.
  - `paper_id` is the manifest file stem. IDs come from `point_id(paper_id, chunk_index)` (uuid5, `PAPERLENS_NS` derived from `uuid5(NAMESPACE_DNS, "paperlens.medai")`). Each paper's points are deleted by a `paper_id` filter and then upserted; the delete comes first (as in R3), so a crash between the two leaves that paper missing until the next run.
  - Mismatch check (`_collection_is_stale`): vector size, then one sampled point's `embedding_model`. A point without the field counts as a mismatch, which heals the old MiniLM volumes (same 384 dim, integer IDs). Named-vector collections are recreated. An existing empty collection is kept. Recreation logs a warning with the reason.
  - Warnings: one per paper without a usable cache (an invalid file also warns inside `load_parsed`), plus one corpus-level summary when no paper has a cache.
  - `scripts/ingest.py`: `--parsed-dir` / `PARSED_DIR` (default `data/parsed`), passes `get_model_name()`, logs chunks instead of papers.
  - Tests: `TestIngest` rewritten around an in-memory `FakeQdrant` (get/create/delete collection, scroll, delete by filter, upsert) so idempotency, stale-point removal and delete-before-upsert are real behaviour, not call assertions. Covers the fallback, provenance payload, full text (sections, pages, References dropped), embed text, corrupt cache, rerun IDs, shrinking paper, size from model, size/model/legacy mismatch recreate, overwrite. 33 tests in `test_retrieval.py`, 157 in the suite. ruff, mypy and bandit clean.
  - Not tested against real Docling output or a live Qdrant (T12). The qdrant filter-delete and `scroll` calls are only exercised through the fake, so T12 is where they meet a real server.

### [x] T8: `Citation` fields + search payload
- Files: `app/models/query.py`, `app/retrieval/search.py`, `tests/test_retrieval.py`, `tests/test_query.py`
- Depends on: —
- Group: G1
- Acceptance criteria:
  - [x] Four optional fields as in Decisions; `DEFAULT_SCORE_THRESHOLD` constant in `app/models/query.py`, used by `QueryRequest` and `search()`
  - [x] `search()` reads the new fields defensively; a legacy payload without them still gives a valid `Citation`
  - [x] Existing `/query` and `/query/stream` tests and the `scripts/llm_bench` fixtures pass unchanged (R5)
- Note: Done. `search()` reads the new fields through `_optional_str` / `_optional_int`: a malformed value (wrong type, `True` as a page) becomes `None` instead of dropping the whole citation. `DEFAULT_SCORE_THRESHOLD` is commented as provisional (T12 resets it). `ui/app.py` still hard-codes 0.4 in its three defaults and the slider; that is T12's slider default, so I left it. `scripts/llm_bench/fixtures.py` builds `Citation` without the new fields and imports unchanged. Extra tests: provenance round-trip, legacy payload, malformed fields, default threshold, and a `/query` test with one chunk and one legacy citation. 115 tests pass.

### [x] T9: Evidence labels, `num_ctx`, budget guard
- Files: `app/workflow/generate.py`, `app/llm/ollama.py`, `.env.example`, `tests/test_workflow.py`, `tests/test_llm_stream.py`
- Depends on: T8
- Group: G2
- Acceptance criteria:
  - [x] Evidence lines look like `[n] {title} — {section} (p. {page})` followed by the excerpt; missing fields are left out
  - [x] `OLLAMA_NUM_CTX` (default 8192) goes into `options` for generate, stream and warmup; explicit kwargs still override it; tested with `MockTransport`
  - [x] Guard drops whole trailing items with a warning when over budget, never cuts text (R6); tested
  - [x] Check whether `scripts/llm_bench` builds `OllamaClient` itself; if so, record it in Note (no change unless it breaks)
- Note: Done.
  - `OllamaClient(..., num_ctx=None)`: when set, `num_ctx` goes into `options` for `generate`, `stream` and `warmup` (warmup now sends `options` too); explicit per-call kwargs override it. `make_ollama_client()` reads `OLLAMA_NUM_CTX` (default `DEFAULT_NUM_CTX = 8192`, empty falls back to it). A non-numeric value raises `ValueError` at startup.
  - `scripts/llm_bench/providers.py:123` builds `OllamaClient(base_url, model_name)` itself, so the benchmark sends no `num_ctx` (the constructor default is `None`). It still works unchanged; benchmark timings are therefore at Ollama's own default context, not at 8192.
  - Evidence item: `[n] {title} — {section} (p. {page})\n{excerpt}`, with section and page left out when missing.
  - Guard: `make_generate_node(llm, *, evidence_budget_tokens=4000)`, chars/3 estimate (including the `\n\n` separator), drops trailing whole items with a warning. Choice not in the plan: the top-ranked item is always kept, even if it alone is over budget, so a question with citations is never sent without evidence. It can't happen with 512-token chunks.
  - Tests: 5 in `test_llm_stream.py` (MockTransport: stream, generate, kwarg override, warmup, no num_ctx, env parsing), 4 in `test_workflow.py` (labels with missing fields, drop prefix + warning, never cut, silent when it fits). 166 in the suite; ruff, mypy, bandit clean.
  - `docker-compose.yml` does not pass `OLLAMA_NUM_CTX` yet (T11).

### [x] T10: UI shows paper + section + page
- Files: `ui/app.py`, `tests/test_ui_client.py`
- Depends on: T8
- Group: G2
- Acceptance criteria:
  - [x] `CitationData` gets `NotRequired` fields; the card label reads `{title} ({year}) — {section}, p. {page}`, and the live citations line shows title + section
  - [x] The confidence indicator is relabelled as a top similarity score that is not calibrated (Risk)
  - [x] Works with old responses that lack the new fields
- Note: Done.
  - `CitationData` has `NotRequired` `paper_id`, `section`, `page`, `chunk_index`. `citation_label(citation, index)` gives `n. {title} ({year}) — {section}, p. {page}`; a missing or null section/page is left out, so legacy citations read `n. title (year)`. `citation_summary(citation)` gives `title — section` for the live line.
  - The live line now says "N passages found" (each citation is a chunk, not a paper). Duplicate titles are not collapsed.
  - Confidence text: "Top similarity score: 85% (not calibrated)"; the progress bar and the API field name `confidence` are unchanged.
  - The slider default 0.4 is unchanged (T12). 6 new tests in `tests/test_ui_client.py` (29 there, 172 in the suite); only the pure helpers are tested, not the `st.expander` rendering. ruff and mypy clean.

### [x] T11: Docker Compose wiring
- Files: `docker-compose.yml`
- Depends on: T7
- Group: G4
- Acceptance criteria:
  - [x] `ingest` mounts `./data/parsed:/app/data/parsed:ro`
  - [x] `ingest` and `api` both get `EMBEDDING_MODEL: ${EMBEDDING_MODEL:-BAAI/bge-small-en-v1.5}`; `api` gets `OLLAMA_NUM_CTX`
  - [x] The built image contains no `data/pdfs`, no `data/parsed` and no parser package (checked with `docker run … python -c "import docling"` → ImportError) (R1). `docling_core` *is* expected in the image (chunking at ingest); check `python -c "import docling_core.transforms.chunker"` succeeds
- Note: Done. The image checks ran on the CPU-torch build (see T12): no `docling`, chunker importable, no `data/pdfs` / `data/parsed`.
  - `ingest` mounts `./data/parsed:/app/data/parsed:ro` (`PARSED_DIR` already defaults to `data/parsed`). `ingest` and `api` get `EMBEDDING_MODEL: ${EMBEDDING_MODEL:-BAAI/bge-small-en-v1.5}`; `api` gets `OLLAMA_NUM_CTX: ${OLLAMA_NUM_CTX:-8192}`. `docker compose config` is valid.
  - Checked statically only: `docling-core[chunking]` is a main dependency, `docling` is only in the `pdf` extra, the Dockerfile installs no extras, and `.dockerignore` excludes `data/pdfs/` and `data/parsed/`.
  - **Run (results):** `import docling` → `ModuleNotFoundError`; `import docling_core.transforms.chunker` → ok; `ls data` → only `papers`, no `pdfs` / `parsed`; image 1.56 GB (was 6.27 GB with CUDA torch). Commands were: `docker run --rm <api image> python -c "import docling"` must raise ImportError; `python -c "import docling_core.transforms.chunker"` must succeed; `ls data` shows no `pdfs` / `parsed`.
  - `.gitkeep` is excluded from the image, so the mount point comes from the bind mount; a missing host `./data/parsed` is created empty by Docker, which gives the abstract-only fallback. `.env.example` has both variables already; its "Docker Compose overrides" comment doesn't list them (cosmetic).

### [x] T12: Real-corpus run, U4 threshold, demo path
- Files: `app/models/query.py` (threshold value), `ui/app.py` (slider default)
- Depends on: T4, T7, T9, T10, T11
- Group: G5
- Acceptance criteria:
  - [x] Download and parse all 10 papers; quick manual check of one parsed file per paper (headings, pages, no headers/footers in the text); record problems in Note
  - [x] Chunk counts per paper and type logged (A3); max chunk tokens ≤ 512
  - [x] U4: top-5 score distribution for about 10 on-topic and 3 off-topic questions; set a provisional threshold and record the numbers in Note
  - [x] Fresh clone with an empty `data/parsed/`: `docker compose up --build` completes, abstract-only, warning logged (R4). With the cache present, citations show section + page in the UI
- Note: Done.
  - **Download/parse:** all 10 PDFs were already in `data/pdfs/`; the other 9 papers parsed without errors (0.4–2.6 MB JSON each, `maira_2` 226 s). The manual per-paper check was done through the chunk sections and page ranges, not by reading each file.
  - **Chunks (595 total, all in Qdrant; max embedded length 512 tokens ≤ 512):** conch 52 (50 text, 2 caption), cxr_vlm_generation 52 (39 text, 13 table), deep_tumor_vqa 40 (24/16), gaze2report 18 (13/5), intersectional_fairness_vlm 40 (33 text, 6 table, 1 caption), llava_med 38 (28/10), maira_2 104 (67 text, 28 table, 9 caption), med_palm_m 89 (69/20), pathalign 57 (31 text, 24 table, 2 caption), radfm 105 (61 text, 43 table, 1 caption). Median 338–402 tokens per paper. Pages run from 1 to the last page of each paper (e.g. maira_2 1–67). The real-Qdrant ingest also exercised the filter-delete and upsert path for the first time (T7 gap closed).
  - **Problems in the labels (left as they are):**
    - `conch`: all Methods subsections come out as `Discussion > …` (13 of them), because the paper's Methods heading isn't recognised.
    - False headings from Docling: `deep_tumor_vqa` "(continued from previous page)", `llava_med` "Visual input example, Biomedical image from [6]", `med_palm_m` "Input" / "Target", `radfm` "Contents".
    - `radfm` has a very long title-like top-level heading (`Towards Generalist Foundation Model … Web-scale 2D&3D Medical Data`) and `maira_2` has 60+ distinct labels from its appendix, so labels are long but still correct.
    - Phase 9 can decide whether to fix these; none breaks retrieval or citations.
  - **U4 threshold (bge-small, raw query, no prefix):** 10 on-topic questions that name a paper gave top-5 scores of 0.74–0.92 (lowest: PathAlign 0.739–0.759; highest: RadFact 0.918). 3 off-topic questions (cookies, World Cup, Kubernetes) gave 0.44–0.53 (top-1 max 0.529). `DEFAULT_SCORE_THRESHOLD` and the slider/defaults in `ui/app.py` went from 0.4 to **0.6**. 0.4 would have let every off-topic question through. My on-topic questions name the papers, which inflates their scores, so real questions will score lower; this is why it sits near the off-topic range rather than the midpoint (0.63). Phase 9 tunes it.
  - **End to end (local API, cache present):** "What is RadFact and how does MAIRA-2 use it?" → grounded, 5 citations, top score 0.80, e.g. `Methods > RadFact: An evaluation suite for (grounded) reports`, p. 5, chunk 12.
  - **Found: stale local `.env` breaks retrieval silently.** The gitignored `.env` still has `EMBEDDING_MODEL=all-MiniLM-L6-v2`. `app/main.py` calls `load_dotenv()`, `scripts/ingest.py` does not, so ingest used bge-small and the API queried with MiniLM. Both are 384-dim, so nothing failed; top scores were ~0.17 and `/query` returned "No relevant papers found". I did not edit `.env`; the check above used an `EMBEDDING_MODEL` override. Fix: update `.env`, and consider making `ingest.py` load `.env` and/or having the API check the collection's `embedding_model` at startup (not done).
  - `python scripts/ingest.py` (as in CLAUDE.md) fails with `ModuleNotFoundError: app` here; it works with `PYTHONPATH=.`.
  - 172 tests, ruff, mypy pass after the threshold change.
  - **Fixed afterwards (follow-up to the findings above):**
    - `.env`: `EMBEDDING_MODEL` set to `BAAI/bge-small-en-v1.5` (local file, not committed).
    - `scripts/ingest.py` now calls `dotenv.load_dotenv()` like the API.
    - New `app/retrieval/consistency.py`: at startup the API compares one point's `embedding_model` with `EMBEDDING_MODEL` and logs a warning on a mismatch (never fails startup; silent if Qdrant is empty or unreachable). 3 tests in `tests/test_retrieval.py`. Not seen firing live, because the models match after the `.env` fix.
    - Chunking: `Online Methods` is a canonical top level (`conch`'s Methods now read `Online Methods > …`, not `Discussion > …`); false headings `(continued from previous page)` and `Contents` are left out of label and embed text. 4 tests in `tests/test_chunking.py`. Left alone on purpose: `Input` / `Target` (`med_palm_m`) and `Caption:` / `Example input:` / `Visual input example…` (`llava_med`) are real labels inside examples; `radfm`'s long first heading is the paper's title variant.
    - Ingest corpus again: still 595 chunks, same types and max 512 tokens. A query with the `.env` default (no override) is grounded again (top score 0.83).
    - `CLAUDE.md` and `README.md` use `python -m scripts.ingest` (`python scripts/ingest.py` fails locally with `ModuleNotFoundError: app`). Compose still runs `scripts/ingest.py` in the image, where it works.
  - **Docker image: CPU-only torch (not in the original plan).** The first image build pulled the CUDA torch and 17 NVIDIA packages plus triton (1.7 GB of the 2.5 GB Linux install) for a CPU-only service, which took hours on this connection. `pyproject.toml` now has `torch>=2.14.0,<2.15` as a direct dependency with `[tool.uv.sources]` sending it to `https://download.pytorch.org/whl/cpu` on Linux only (`[[tool.uv.index]] pytorch-cpu`, explicit). `uv lock`: Linux gets `torch 2.14.1+cpu`, Windows keeps `2.14.0`; the NVIDIA packages and triton are gone from the lock, so the install is about 400 MB. The Dockerfile is unchanged (`uv sync --frozen --no-dev` reads the lock). Local pip-audit (venv): no vulnerabilities; not re-run against the lock itself.
  - **Build attempts:** two builds were stopped by hand (CUDA lock), one failed on a corrupted `pyyaml` download (`deflate decompression error`, network), one was killed for low system memory, a fifth is the one that counts. A uv cache mount in the Dockerfile would make retries cheap; not added.
  - **Docker run (final build, exit 0, image 1.56 GB):**
    - Empty `data/parsed/` (`docker compose run -v <empty dir> ingest`): one warning per paper plus the corpus summary, 10 abstract-only chunks. The previous 595 points were removed by the real `paper_id` filter delete (no stale chunks).
    - Real cache mounted (`docker compose run ingest`): 595 chunks, no warnings, payload `embedding_model` = `BAAI/bge-small-en-v1.5`.
    - `api` container from the new image, queried on `localhost:8000`: grounded, 5 citations with section + page, top score 0.82, no mismatch warning.
    - Not run: the full `docker compose up --build` of all five services and the Streamlit UI in a browser (the UI image was not rebuilt; its changes are covered by the unit tests only). The `api` container was stopped again afterwards.

### [x] T13: Docs
- Files: `README.md`, `CLAUDE.md`, `docs/roadmap.md`, `docs/adr/0021-*.md`, `docs/adr/0022-*.md`
- Depends on: all
- Group: G6
- Acceptance criteria:
  - [x] README: download → parse → ingest steps, the `pdf` extra, `OLLAMA_NUM_CTX` and `EMBEDDING_MODEL`, the new `Citation` fields; confidence not presented as calibrated
  - [x] CLAUDE.md: architecture (parsed.py, chunking.py, scripts), env var list, phase status
  - [x] ADR-0021/0022 updated with anything Execute changed; roadmap row 8 marked done
- Note: Done. Also updated `docs/architecture.md` (retrieval layer; it still named MiniLM and "source URL"). README has a new "Full-text corpus" section and a note that Compose indexes abstracts only until `data/parsed/` is filled. ADR-0022: threshold 0.6 with the measurements, the stale-`.env` finding and its two guards. ADR-0021: an "Execute results" section with the T12 numbers and the label fixes. Roadmap row 8 is ✓ with the three ADRs. CLAUDE.md status: Phase 7 merged (PR #11), Phase 8 executed, next is Verify. Nothing in the docs was run; the commands in the README are the ones used in T4/T12.

## ADRs
- ADR-0020: PDF parser choice (written in T2 from the spike results)
- [ADR-0021](../../adr/0021-structure-aware-chunking.md): chunking with Docling's HybridChunker (revised in Execute), chunk/point identity, `Citation` contract change
- [ADR-0022](../../adr/0022-default-embedder-bge-small.md): default embedder `bge-small-en-v1.5`, dimension taken from the model (extends ADR-0012)
