# Phase 8: Full-text ingestion — Verify

Verified together with the open Phase 7 follow-ups (Compose run, live UI path, Ctrl-C stop),
because both need the same running stack. Phase 7 itself was already verified and merged (PR #11).

## Acceptance criteria
| Task | Criterion | ✔/✘ | Evidence |
|---|---|---|---|
| T1 | Manifests (versioned arxiv_id, pdf_url, source_note), download script, ignore files | ✔ | `tests/test_download.py`; 10 PDFs present in `data/pdfs/` |
| T2 | Parser spike → ADR-0020 | ✔ | ADR-0020 exists, indexed in `docs/adr/README.md` |
| T3 | `load_parsed` → `DoclingDocument \| None`, warnings on bad input | ✔ | `tests/test_parsed.py` |
| T4 | `parse_papers` script + `pdf` extra, docling not in image | ✔ | `tests/test_parse_papers.py`; image: `find_spec('docling')` → `None` |
| T5 | HybridChunker chunking, labels, dropped sections, ≤ 512 tokens | ✔ | `tests/test_chunking.py`; real corpus 595 chunks, max 512 (T12) |
| T6 | bge-small default, dim from model | ✔ | `tests/test_retrieval.py`; collection recreated on rebuild |
| T7 | Ingest from cache, uuid5 IDs, filter delete + upsert, stale-collection recreate | ✔ | tests + live: `Ingested 595 chunks.` in the Compose `ingest` container |
| T8 | `Citation` fields + defensive search | ✔ | tests; live `/query` returns section/page/chunk_index |
| T9 | Evidence labels, `OLLAMA_NUM_CTX`, budget guard | ✔ | `tests/test_llm_stream.py`, `tests/test_workflow.py` |
| T10 | UI shows paper + section + page | ✔ | `tests/test_ui_client.py`; `citation_label` run inside the rebuilt UI container (see Demo path) |
| T11 | Compose wiring; image has chunker but no parser, no `data/pdfs`/`data/parsed` | ✔ | rebuilt image: `docling` absent, `docling_core.transforms.chunker` imports, `ls data` → `papers` only; 1.56 GB |
| T12 | Real corpus, threshold 0.6, demo path | ✔ | see Demo path; off-topic query now returns "No relevant papers found." (0 citations) |
| T13 | Docs | ✔ | ADR-0021/0022, README, CLAUDE.md, roadmap, architecture.md in the diff |

## Quality gates
- [x] pytest — 179 passed (1 unrelated qdrant version warning)
- [x] ruff — all checks passed
- [x] mypy — no issues in 32 source files
- [x] bandit — no issues
- [x] pip-audit — "No known vulnerabilities found" against the lock (`uv export --frozen --no-dev` → `pip-audit -r … --no-deps`)

## Demo path
- [x] `docker compose up --build` works end to end (rebuilt all images, exit 0). qdrant/ollama/api healthy, `ingest` exited 0 with `Ingested 595 chunks.`, UI answers on :8501 (`/_stcore/health` → `ok`), `/model/status` → warm.
- [x] **Streaming, live (Phase 7 follow-up 1):** `POST /query/stream` through the real stack: `step_start/step_end` for classify → rewrite → retrieve, `citations` event, then incremental `token` events, then `final`. Run from inside the UI container with `ui.app.stream_query_api`: first token after 1.8 s, 278 tokens, 14.4 s total, grounded, 5 citations.
- [x] **Citations carry section + page:** e.g. `MAIRA-2 … (2024) — Methods > RadFact: An evaluation suite for (grounded) reports, p. 5`.
- [x] **Ctrl-C stop (Phase 7 follow-up 3):** killed a `curl -N` after 9 s (111 tokens). The count stayed at 112 afterwards; Ollama logged `cancel task`, its `/api/generate` call ended at 8.5 s, and CPU fell to ~0 %.
- [x] Safety: unsafe personal-advice query → `grounded: false`, `route: unsafe_medical_advice`; off-topic query → no citations.
- [x] **Streamlit in a real browser:** tested by the owner, works well. A 30 s+ stream was not separately timed through the browser.

## Docs
- [x] ADRs 0020–0022 exist and are indexed
- [x] README, CLAUDE.md, roadmap row 8, architecture.md updated (T13)
- [x] CLAUDE.md status updated to "Phase 8 verified"

## Follow-ups
1. (Done by the owner.) UI checked in a browser.
2. Phase 9 label issues kept as they are (T12 note): false headings `Input`/`Target` (`med_palm_m`), `Caption:`/`Example input:` (`llava_med`); long appendix labels in `maira_2`.
3. Threshold 0.6 is provisional (own questions name the papers, which inflates scores); Phase 9 tunes it.
4. HF Hub "unauthenticated requests" warning in the api log is harmless; set `HF_TOKEN` only if rate limits appear.
5. Untracked `main.py` and `plan_*.md` in the repo root belong to no phase; keep them out of the PR.
6. Not yet pushed: branch `phase-8-full-text-ingestion` (commit ddfc7df) has no remote and no PR.
7. On sign-off: commit this file, update CLAUDE.md status, push, open the PR.
