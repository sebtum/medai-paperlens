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

# Ingest papers into Qdrant (requires Qdrant running)
.\.venv\Scripts\python.exe scripts/ingest.py
.\.venv\Scripts\python.exe scripts/ingest.py --overwrite   # drop and re-index

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

### Current implementation (Phase 5 complete)

**API layer**
- `app/main.py` — FastAPI app factory; manages `OllamaClient` lifespan; mounts `health_router` and `query_router`; registers `UnsafeQueryError` exception handler.
- `app/api/routes/health.py` — `GET /health` endpoint.
- `app/api/routes/query.py` — `POST /query`. Builds and invokes the LangGraph workflow, returns `QueryResponse`.
- `app/core/exceptions.py` — `UnsafeQueryError`.
- `app/models/query.py` — `QueryRequest`, `Citation`, `QueryResponse`.

**LLM layer**
- `app/llm/base.py` — `LlmProvider` Protocol with `generate()` and `generate_structured()`.
- `app/llm/ollama.py` — `OllamaClient`: async context manager; `generate()` calls `/api/generate` (string prompt) or `/api/chat` (message list); default model `qwen3.5:4b` via `OLLAMA_MODEL` env var.

**Workflow layer** (LangGraph)
- `app/workflow/state.py` — `WorkflowState` TypedDict: `question`, `rewritten`, `is_unsafe`, `citations`, `confidence`, `answer`, `grounded`, `route`.
- `app/workflow/graph.py` — `build_workflow(client, provider, llm)`: compiles the StateGraph; routes unsafe queries directly to END, safe queries through rewrite → retrieve → generate → validate.
- `app/workflow/classify.py` — `classify_node`: phrase-pattern safety filter (frozenset of unsafe patterns); sets `is_unsafe` / `route`.
- `app/workflow/rewrite.py` — `make_rewrite_node(llm)`: Ollama-backed query rewriter; falls back to original question on LLM error.
- `app/workflow/retrieve.py` — `make_retrieve_node(client, provider)`: embeds rewritten query, runs Qdrant search, populates `citations` and `confidence`.
- `app/workflow/generate.py` — `make_generate_node(llm)`: synthesises answer from evidence via Ollama; returns error message on LLM failure.
- `app/workflow/validate.py` — `validate_node`: sets `grounded = True` when citations exist and answer is non-empty.

**Retrieval layer**
- `app/retrieval/client.py` — `get_async_client()`: FastAPI dependency for `AsyncQdrantClient`, reads `QDRANT_URL` from env.
- `app/retrieval/embedding.py` — `EmbeddingProvider` Protocol; `get_embedding_provider()` FastAPI dependency; `SentenceTransformer` singleton (reads `EMBEDDING_MODEL` env var); `embed_async()` offloads to threadpool.
- `app/retrieval/ingestion.py` — `ingest(papers_dir, client, model, *, overwrite)`: validates abstracts, processes in batches of 32, upserts to Qdrant collection `papers`.
- `app/retrieval/search.py` — `search(vector, client, top_k)`: queries Qdrant, parses payloads defensively, returns `(list[Citation], float)`.
- `scripts/ingest.py` — CLI ingestion script; `--papers-dir` / `PAPERS_DIR` env var; `--overwrite` flag; `sys.exit(1)` on failure.
- `data/papers/*.json` — 10 curated public medical-AI papers (corpus).

### Key architectural decisions

- **Constrained agentic RAG, not a full autonomous agent** — explicit workflow nodes, no open-ended multi-step web research (ADR-0001).
- **FastAPI backend, Streamlit UI** — UI holds no business logic; backend owns the RAG workflow (ADR-0004, ADR-0006).
- **Qdrant** for vector search over the paper corpus (ADR-0005).
- **Ollama** is the default and only required LLM provider. External providers must be opt-in and disabled by default (ADR-007).
- **LangGraph** is the workflow orchestrator (ADR-008).
- **Docker Compose** before Kubernetes — do not add K8s until Compose is working (ADR-0002).

### Build order constraint

Do not add the next layer until the current one is stable:
1. ~~FastAPI with `/health` and `/query` endpoints~~ ✓
2. ~~Local retrieval against Qdrant corpus~~ ✓
3. ~~LangGraph workflow nodes + Ollama LLM synthesis~~ ✓
4. ~~Streamlit UI~~ ✓
5. Docker Compose for full-stack local run  ← next

### API contract

`GET /health` → `{"status": "ok"}`

`POST /query` — accepts `{"question": "..."}`, returns `{"answer", "citations", "confidence", "grounded", "debug"}`.

Unsafe medical queries (diagnosis, treatment, patient advice) must return a refusal response with `"grounded": false` and `"route": "unsafe_medical_advice"` in debug. Research questions about AI models *performing* diagnosis are allowed — only first-person personal-advice patterns are refused.

## Testing

Tests use `httpx.AsyncClient` with `ASGITransport` (in-process, no running server needed). `pytest-asyncio` is configured in `auto` mode — no `@pytest.mark.asyncio` decorator required. LLM calls must use a mock provider; never require a live Ollama or external API in the test suite.

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
- No secrets committed to Git. Use environment variables (`OLLAMA_BASE_URL`, `QDRANT_URL`, `LOG_LEVEL`). Commit `.env.example`, never `.env`.
- Do not make clinical claims in any generated output.
