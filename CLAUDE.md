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

### Request flow (planned)

```
Streamlit UI → FastAPI Backend → Constrained Agentic RAG Workflow
                                   ├── Query Classifier
                                   ├── Query Rewriter
                                   ├── Paper Retriever   (Qdrant)
                                   ├── Evidence Extractor
                                   ├── Answer Generator  (Ollama, local-first)
                                   └── Grounding Validator
                                 → Citation-grounded response
```

### Current implementation (Phase 3 complete)

- `app/main.py` — FastAPI app factory; mounts `health_router` and `query_router`; registers `UnsafeQueryError` exception handler.
- `app/api/routes/health.py` — `GET /health` endpoint.
- `app/api/routes/query.py` — `POST /query`. Checks `_is_unsafe()` (phrase-pattern placeholder — Phase 4 replaces with classifier/Llama Guard). Safe queries run `embed_async → search`, return grounded citations with real Qdrant confidence scores.
- `app/core/exceptions.py` — `UnsafeQueryError`.
- `app/models/query.py` — `QueryRequest`, `Citation`, `QueryResponse`.
- `app/retrieval/client.py` — `get_client()`: `@lru_cache` singleton for `QdrantClient`, reads `QDRANT_URL` from env.
- `app/retrieval/embedding.py` — `get_model()`: `@lru_cache` singleton for `SentenceTransformer` (reads `EMBEDDING_MODEL` env var). `embed()` returns `np.ndarray`. `embed_async()` offloads to threadpool via `asyncio.to_thread`.
- `app/retrieval/ingestion.py` — `ingest(papers_dir, client, model, *, overwrite)`: validates abstracts, processes in batches of 32, upserts to Qdrant collection `papers`.
- `app/retrieval/search.py` — `search(vector, client, top_k)`: queries Qdrant, parses payloads defensively, returns `(list[Citation], float)` where float is the top cosine score.
- `scripts/ingest.py` — CLI ingestion script; `--papers-dir` / `PAPERS_DIR` env var; `--overwrite` flag with production warning; `sys.exit(1)` on failure.
- `data/papers/*.json` — 10 curated public medical-AI papers (corpus).

### Key architectural decisions

- **Constrained agentic RAG, not a full autonomous agent** — explicit workflow nodes with a single optional retry (ADR-0001). Do not implement open-ended multi-step web research.
- **FastAPI backend, Streamlit UI** — UI holds no business logic; backend owns the RAG workflow (ADR-0004, ADR-0006).
- **Qdrant** for vector search over the paper corpus (ADR-0005).
- **Ollama** is the default and only required LLM provider. External providers must be opt-in and disabled by default (ADR-007).
- **Docker Compose** before Kubernetes — do not add K8s until Compose is working (ADR-0002).
- **LangGraph** is the planned workflow orchestrator — do not add it until basic retrieval works (ADR-008).

### Build order constraint

Do not add the next layer until the current one is stable:
1. ~~FastAPI with `/health` and `/query` endpoints~~ ✓
2. ~~Local retrieval against Qdrant corpus~~ ✓
3. LangGraph workflow nodes  ← next
4. Streamlit UI
5. Docker Compose for full-stack local run

### API contract

`GET /health` → `{"status": "ok"}`

`POST /query` — accepts `{"question": "..."}`, returns `{"answer", "citations", "confidence", "grounded", "debug"}`.

Unsafe medical queries (diagnosis, treatment, patient advice) must return a refusal response with `"grounded": false` and `"route": "unsafe_medical_advice"` in debug. Research questions about AI models *performing* diagnosis are allowed — only first-person personal-advice patterns are refused.

## Testing

Tests use `httpx.AsyncClient` with `ASGITransport` (in-process, no running server needed). `pytest-asyncio` is configured in `auto` mode — no `@pytest.mark.asyncio` decorator required. LLM calls must use a mock provider; never require a live Ollama or external API in the test suite.

## Coding standards

- Type hints are mandatory on all shared interfaces.
- Use domain-driven names: `evidence_extractor`, `grounding_validator`, not generic utilities.
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
