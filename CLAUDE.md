# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**IMPORTANT (Windows):** Do NOT use `uv run` to invoke tools. On Windows, `uv run` triggers a venv sync on every call and can corrupt the venv with "Access is denied" errors when removing dist-info files. Always invoke tools directly via the venv:

```powershell
# Run the API server
.\.venv\Scripts\uvicorn.exe app.main:app --reload

# Lint
.\.venv\Scripts\ruff.exe check .

# Format
.\.venv\Scripts\ruff.exe format .

# Type check
.\.venv\Scripts\mypy.exe app/

# Security scan (matches CI gate)
.\.venv\Scripts\bandit.exe -r app/

# Dependency CVE audit
.\.venv\Scripts\pip-audit.exe

# Full-text corpus, host side (Phase 8): download PDFs -> parse with Docling -> data/parsed/
.\.venv\Scripts\python.exe -m scripts.download_papers   # data/pdfs/<paper_id>.pdf (gitignored); --force re-downloads
.\.venv\Scripts\python.exe -m scripts.parse_papers      # needs the `pdf` extra (docling); --only <id>, --force

# Ingest papers into Qdrant (requires Qdrant running; no parsed cache = abstract-only fallback)
.\.venv\Scripts\python.exe -m scripts.ingest   # run as a module so `app` imports
.\.venv\Scripts\python.exe -m scripts.ingest --overwrite   # drop and re-index

# Ingest into another collection (eval only, e.g. a second embedder; `papers` stays untouched)
.\.venv\Scripts\python.exe -m scripts.ingest --collection eval_bge-base-en-v1.5

# Retrieval eval (Phase 9; run as modules, needs Qdrant; golden set is LLM-labelled, not human-verified)
.\.venv\Scripts\python.exe -m scripts.eval_retrieval run --config baseline --no-llm   # baseline | no_rewrite | prefix | bge_base; --langsmith also logs an experiment
.\.venv\Scripts\python.exe -m scripts.eval_retrieval compare baseline prefix          # paired bootstrap on nDCG@5
.\.venv\Scripts\python.exe -m scripts.eval_retrieval curve baseline                   # threshold curve
.\.venv\Scripts\python.exe -m scripts.eval_retrieval sections --paper maira_2         # authoring: sections and chunk ids
.\.venv\Scripts\python.exe -m scripts.eval_retrieval pool baseline prefix             # authoring: unlabelled sections in the top 10
.\.venv\Scripts\python.exe -m scripts.sync_golden                                     # copy golden.jsonl to LangSmith (opt-in, needs a clean committed file)

# Benchmark candidate LLM providers/models for speed (run as a module)
.\.venv\Scripts\python.exe -m scripts.benchmark_llm --models ollama:qwen3:0.6b,ollama:qwen3:1.7b

# Generate a human-scorable groundedness/quality review sheet (run as a module)
.\.venv\Scripts\python.exe -m scripts.eval_llm --models ollama:qwen3:0.6b --format markdown

# Run the Streamlit UI
.\.venv\Scripts\streamlit.exe run ui/app.py

# Run all tests
.\.venv\Scripts\pytest.exe tests/

# Run a single test file
.\.venv\Scripts\pytest.exe tests/test_query.py

# Run a single test by name
.\.venv\Scripts\pytest.exe tests/test_query.py::test_query_unsafe_personal_advice_returns_refusal
```

If the venv ever appears broken (ImportError, package at "unknown location"), fix it with pip `--ignore-installed --no-deps` rather than `uv run`:

```powershell
.\.venv\Scripts\pip.exe install --ignore-installed --no-deps "<package>==<version>"
```

## Architecture

MedAI PaperLens is a local-first AI engineering project for medical AI literature intelligence — not a clinical tool. It accepts research questions and returns citation-grounded summaries over a constrained corpus of public medical AI papers.

### Request flow

```
Streamlit UI → FastAPI Backend → LangGraph Workflow
                                             ├── classify_node   (phrase-pattern safety filter)
                                             ├── rewrite_node    (Ollama query rewriter)
                                             ├── retrieve_node   (Qdrant vector search)
                                             ├── generate_node   (Ollama LLM synthesis)
                                             └── validate_node   (grounding check)
                                           → Citation-grounded response
```

### Current implementation (Phase 9 complete)

**API layer**
- `app/main.py` — FastAPI app factory; manages the active LLM provider's lifespan via `make_llm_provider()`; mounts `health_router`, `model_router`, and `query_router`; registers `UnsafeQueryError` exception handler.
- `app/api/routes/health.py` — `GET /health` endpoint.
- `app/api/routes/model.py` — `GET /model/status`. Reports whether the active LLM provider's model is warm.
- `app/api/routes/query.py` — `POST /query` (builds and `ainvoke`s the LangGraph workflow, returns `QueryResponse`) and `POST /query/stream` (SSE via `EventSourceResponse`; `astream(["tasks","custom","values"])` run in its own task so a client disconnect cancels the graph and LLM; ADR-0017).
- `app/core/exceptions.py` — `UnsafeQueryError`.
- `app/models/query.py` — `QueryRequest`, `Citation` (one per chunk; optional `paper_id`, `section`, `page`, `chunk_index`), `QueryResponse`, SSE event payloads (`StepStartEvent`, `StepEndEvent`, `CitationsEvent`, `TokenEvent`, `ErrorEvent`).

**LLM layer**
- `app/llm/base.py` — `LlmProvider` Protocol with `generate()`, `stream()` and `generate_structured()` (ADR-0018); `ModelStatusProvider` Protocol with `model`, `warmup()`, `is_model_warm()`.
- `app/llm/ollama.py` — `OllamaClient`: async context manager; `generate()` and `stream()` (NDJSON, never yields `thinking`) call `/api/generate` (string prompt) or `/api/chat` (message list); default model `qwen3:0.6b` via `OLLAMA_MODEL` env var, thinking mode off by default via `OLLAMA_THINK`; `OLLAMA_NUM_CTX` (default 8192) is sent as `options.num_ctx` on every call including warmup.
- `app/llm/gemini.py` — `GeminiClient`: opt-in application provider over the `google-genai` SDK (optional extra); satisfies `LlmProvider` (incl. `stream()`) and `ModelStatusProvider` structurally. `generate`/`stream` on both clients are `@traceable` — LangSmith tracing is opt-in via `LANGSMITH_TRACING=true`, a no-op otherwise (ADR-0019).
- `app/llm/factory.py` — `make_llm_provider()` dispatches on `LLM_PROVIDER` (`ollama` default, `gemini` opt-in); `get_llm_provider()` is the FastAPI dependency reading `app.state.llm`.

**Workflow layer** (LangGraph)
- `app/workflow/state.py` — `WorkflowState` TypedDict: `question`, `rewritten`, `is_unsafe`, `citations`, `confidence`, `answer`, `grounded`, `route`.
- `app/workflow/graph.py` — `build_workflow(client, provider, llm)`: compiles the StateGraph; routes unsafe queries directly to END, safe queries through rewrite → retrieve → generate → validate.
- `app/workflow/classify.py` — `classify_node`: phrase-pattern safety filter (frozenset of unsafe patterns); sets `is_unsafe` / `route`.
- `app/workflow/rewrite.py` — `make_rewrite_node(llm)`: Ollama-backed query rewriter; falls back to original question on LLM error.
- `app/workflow/retrieve.py` — `make_retrieve_node(client, provider)`: embeds rewritten query, runs Qdrant search, populates `citations` and `confidence`.
- `app/workflow/generate.py` — `make_generate_node(llm, *, evidence_budget_tokens=4000)`: synthesises answer from evidence (`[n] title — section (p. page)` + excerpt; a chars/3 budget guard drops whole lowest-ranked items, never truncates) via `llm.stream()` (always streams; tokens go to the LangGraph stream writer, a no-op under `ainvoke`); on LLM failure emits an `error` event and returns an error message.
- `app/workflow/validate.py` — `validate_node`: sets `grounded = True` when citations exist and answer is non-empty.

**Retrieval layer**
- `app/retrieval/client.py` — `get_async_client()`: FastAPI dependency for `AsyncQdrantClient`, reads `QDRANT_URL` from env.
- `app/retrieval/embedding.py` — `EmbeddingProvider` Protocol; `get_embedding_provider()` FastAPI dependency; `SentenceTransformer` singleton; `get_model_name()` reads `EMBEDDING_MODEL` (default `BAAI/bge-small-en-v1.5`, ADR-0022; vector size comes from the model, no `VECTOR_DIM`); `SentenceTransformerProvider(model, query_prefix="")` prepends `query_prefix` in `embed()` (queries only, documents are never prefixed); `get_query_prefix()` reads `EMBEDDING_QUERY_PREFIX` (unset: BGE instruction for `BAAI/bge-*`, none otherwise; empty string = off; ADR-0025); `embed_async()` offloads to threadpool.
- `app/retrieval/parsed.py` — `load_parsed(parsed_dir, paper_id)`: loads the cached Docling JSON (`DoclingDocument`) from `data/parsed/`; `None` + warning if missing/invalid/path-traversal.
- `app/retrieval/chunking.py` — `chunk_paper(...)`: Docling `HybridChunker` over the parsed document plus the custom parts of ADR-0021 (heading levels, dropped sections, `Section > Subsection` label, token budget, manifest abstract = chunk 0); returns frozen `Chunk`s. No parsed file → abstract-only.
- `app/retrieval/ingestion.py` — `ingest(papers_dir, client, model, *, model_name, parsed_dir=None, overwrite=False, tokenizer=None)`: chunks every manifest, embeds in batches, point ID = `uuid5(paper_id:chunk_index)`, deletes a paper's points by `paper_id` filter then upserts into collection `papers`; recreates the collection when vector size or `embedding_model` differs.
- `app/retrieval/consistency.py` — `embedding_model_mismatch()`: startup check that logs a warning when the collection's `embedding_model` differs from `EMBEDDING_MODEL` (never fails startup).
- `app/retrieval/search.py` — `search(vector, client, top_k, score_threshold, *, collection_name=...)`: queries Qdrant (`DEFAULT_SCORE_THRESHOLD` = 0.6, measured inert in Phase 9 and kept, ADR-0025), parses payloads defensively (malformed optional fields become `None`), returns `(list[Citation], float)`. `search_hits(...)` returns scored `RankedHit`s (with `chunk_type`) for the eval.
- `scripts/download_papers.py` — downloads the arXiv PDFs named in the manifests to `data/pdfs/` (gitignored); skips existing files, rate-limited.
- `scripts/parse_papers.py` — host-only: Docling converts PDFs to `data/parsed/<paper_id>.json` (gitignored; Compose mounts it read-only). `docling` is in the optional `pdf` extra, never in the image; `docling-core[chunking]` is a main dependency.
- `scripts/ingest.py` — CLI ingestion script; `--papers-dir` / `PAPERS_DIR`, `--parsed-dir` / `PARSED_DIR`; `--collection` (default `papers`, the app collection; eval embedders use `eval_<model-slug>`); `--overwrite` flag; loads `.env`; `sys.exit(1)` on failure.

**Evaluation layer** (Phase 9; ADR-0023, ADR-0024, ADR-0025; no generation, retrieval only)
- `app/evaluation/metrics.py` — pure `RankedHit`, `cut`, `collapse_sections`, `recall_at_k`, `mrr_at_k`, `ndcg_at_k`, `precision_at_k` and `recall_ceiling_at_k` (reported, not in the decision rule), crowding and table-hit helpers over `(paper_id, section)` labels.
- `app/evaluation/golden.py` — `GoldenEntry` / `Label` / `QuestionType`, `QUOTAS`, `load_golden`, `golden_sha256`, `validate_golden` (the golden-set rules).
- `app/evaluation/target.py` — `RetrievalConfig`, `retrieve_ranked` (top 20 at threshold 0, one retrieval per run), the committed rewrite cache (a rewrite fallback is a run failure), `preflight` (stale-label and embedding-model check against Qdrant).
- `app/evaluation/compare.py` — run records (`save_run` / `load_run`), stratified paired bootstrap, `verdict()` (the decision rule), per-type means and win/loss/tie counts.
- `app/evaluation/report.py` — markdown renderers for run summaries, comparisons and threshold curves; every header states the golden set is LLM-labelled, not human-verified.
- `app/evaluation/runner.py` — the `run`, `compare`, `curve` logic and `NAMED_CONFIGS` (`baseline`, `no_rewrite`, `prefix`, `bge_base`); the eval embedder is built with an empty provider prefix so a config's prefix is added exactly once.
- `app/evaluation/authoring.py` — `sections` and `pool` helpers for writing and adjudicating the golden set.
- `app/evaluation/langsmith_sync.py` / `langsmith_eval.py` — dataset sync (git is the source of truth; hash stored on each example) and `aevaluate` adapters over the same metric functions.
- `scripts/eval_retrieval.py` — thin CLI: `run | compare | curve | sections | pool`; loads `.env`; exit 1 on failure.
- `scripts/sync_golden.py` — thin CLI: syncs `golden.jsonl` to the LangSmith dataset `paperlens-retrieval-golden`; refuses a dirty or invalid file.
- `data/eval/` — `golden.jsonl` (50 questions, 150 labels, **LLM-labelled by Claude, not human-verified**), `rewrites.jsonl` (rewrite cache), `runs/*.json`, `reports/*.md`, `CHANGELOG.md`. No paper text. LF line endings are pinned in `.gitattributes` so the golden hash is stable.

- `data/papers/*.json` — 10 curated public medical-AI papers: manifests with metadata, abstract, versioned `arxiv_id` and `pdf_url` (corpus). Full text comes from the PDFs via the parsed cache.

### Key architectural decisions

- **Constrained agentic RAG, not a full autonomous agent** — explicit workflow nodes, no open-ended multi-step web research (ADR-0001).
- **FastAPI backend, Streamlit UI** — UI holds no business logic; backend owns the RAG workflow (ADR-0004, ADR-0006).
- **Qdrant** for vector search over the paper corpus (ADR-0005).
- **Ollama** is the default and only required LLM provider. External providers must be opt-in and disabled by default (ADR-0007). Gemini is available as one such opt-in provider behind `LLM_PROVIDER=gemini` (ADR-0016).
- **LangGraph** is the workflow orchestrator (ADR-0008).
- **Docker Compose** before Kubernetes — do not add K8s until Compose is working (ADR-0002).

### Build order constraint

Do not add the next layer until the current one is stable:
1. ~~FastAPI with `/health` and `/query` endpoints~~ ✓
2. ~~Local retrieval against Qdrant corpus~~ ✓
3. ~~LangGraph workflow nodes + Ollama LLM synthesis~~ ✓
4. ~~Streamlit UI~~ ✓
5. ~~Docker Compose for full-stack local run~~ ✓

Phases 7–11 (streaming, full-text ingestion, retrieval eval, rerank/hybrid, generation eval) are ordered in [`docs/roadmap.md`](docs/roadmap.md). Phase 7 (streaming) is merged (PR #11). Phase 8 (full-text ingestion) is merged (PR #12). Phase 9 (retrieval evaluation) is verified (ADR-0023/0024/0025; the BGE query prefix is the new default, the other defaults stay; `docs/phases/09-retrieval-evaluation/4-verify.md`); its PR is pending; next is Phase 10. Write each phase's ADRs during that phase's planning, not in advance.

### Working process

Each phase runs in four stages: **Discuss → Plan → Execute → Verify**, each in a fresh session and each handing off through a short file in `docs/phases/NN-name/`. Stages, gates and templates are in [`docs/workflow.md`](docs/workflow.md). Small changes (no new ADR, dependency or shared interface, ≈ ≤ 3 files) skip the process: do them directly, then run the quality gates.

### API contract

`GET /health` → `{"status": "ok"}`

`POST /query` — accepts `{"question": "..."}`, returns `{"answer", "citations", "confidence", "grounded", "debug"}`.

`POST /query/stream` — same body; SSE events `step_start {node}`, `step_end {node, elapsed_ms}`, `citations {citations, confidence}`, `token {text}`, `error {message}`, `final` (= the `/query` response, authoritative). Unsafe queries emit no `token`; `final` has `grounded: false`.

Unsafe medical queries (diagnosis, treatment, patient advice) must return a refusal response with `"grounded": false` and `"route": "unsafe_medical_advice"` in debug. Research questions about AI models *performing* diagnosis are allowed — only first-person personal-advice patterns are refused.

## Testing

Tests use `httpx.AsyncClient` with `ASGITransport` (in-process, no running server needed). `pytest-asyncio` is configured in `auto` mode — no `@pytest.mark.asyncio` decorator required. LLM calls must use a mock provider; never require a live Ollama or external API in the test suite. Tests must not require a LangSmith account; leave `LANGSMITH_TRACING` unset in tests and CI.

## Coding standards

- Type hints are mandatory on all shared interfaces.
- Use domain-driven names: `query_rewriter`, `grounding_validator`, not generic utilities.
- Do not add LangChain or heavy frameworks without explicit approval.
- Naming: `snake_case` for functions/modules, `PascalCase` for classes.
- Python version: 3.14 (see `.python-version`).

## Quality gates

All five must pass before a PR is mergeable (enforced in CI via GitHub Actions):
- `pytest`
- `ruff check .`
- `mypy app/`
- `bandit -r app/`
- `pip-audit`

## Absolute constraints

- No patient data, personal health information, or clinical decision support — ever.
- No secrets committed to Git. Use environment variables (`OLLAMA_BASE_URL`, `OLLAMA_MODEL`, `OLLAMA_THINK`, `OLLAMA_NUM_CTX`, `EMBEDDING_MODEL`, `EMBEDDING_QUERY_PREFIX`, `RERANKER_MODEL`, `PAPERS_DIR`, `PARSED_DIR`, `PDF_DIR`, `LLM_PROVIDER`, `GEMINI_API_KEY`, `GEMINI_MODEL`, `QDRANT_URL`, `LOG_LEVEL`, `LANGSMITH_TRACING`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`, optional `LANGSMITH_ENDPOINT` for the EU region). Commit `.env.example`, never `.env`.
- Do not make clinical claims in any generated output.
