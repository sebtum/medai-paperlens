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

# Run all tests
.\.venv\Scripts\pytest.exe tests/

# Run a single test file
.\.venv\Scripts\pytest.exe tests/test_health.py

# Run a single test by name
.\.venv\Scripts\pytest.exe tests/test_health.py::test_health_ok
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

### Key architectural decisions

- **Constrained agentic RAG, not a full autonomous agent** — explicit workflow nodes with a single optional retry (ADR-0001). Do not implement open-ended multi-step web research.
- **FastAPI backend, Streamlit UI** — UI holds no business logic; backend owns the RAG workflow (ADR-0004, ADR-0006).
- **Qdrant** for vector search over the paper corpus (ADR-0005).
- **Ollama** is the default and only required LLM provider. External providers must be opt-in and disabled by default (ADR-007).
- **Docker Compose** before Kubernetes — do not add K8s until Compose is working (ADR-0002).
- **LangGraph** is the planned workflow orchestrator — do not add it until basic retrieval works (ADR-008).

### Build order constraint

Do not add the next layer until the current one is stable:
1. FastAPI with `/health` and `/query` endpoints
2. Local retrieval against Qdrant corpus
3. LangGraph workflow nodes
4. Streamlit UI
5. Docker Compose for full-stack local run

### API contract

`GET /health` → `{"status": "ok"}`

`POST /query` — accepts `{"question": "..."}`, returns `{"answer", "citations", "confidence", "grounded", "debug"}`.

Unsafe medical queries (diagnosis, treatment, patient advice) must return a refusal response with `"grounded": false` and `"route": "unsafe_medical_advice"` in debug.

## Coding standards

- Type hints are mandatory on all shared interfaces.
- Use domain-driven names: `evidence_extractor`, `grounding_validator`, not generic utilities.
- LLM calls in tests must use a mock provider — never require a live Ollama or external API in the test suite.
- Do not add LangChain or heavy frameworks without explicit approval.
- Naming: `snake_case` for functions/modules, `PascalCase` for classes.

## Quality gates

All three must pass before a PR is mergeable (run locally and in CI via GitHub Actions):
- `pytest`
- `ruff check .`
- `mypy .`

## Absolute constraints

- No patient data, personal health information, or clinical decision support — ever.
- No secrets committed to Git. Use environment variables (`OLLAMA_BASE_URL`, `QDRANT_URL`, `LOG_LEVEL`). Commit `.env.example`, never `.env`.
- Do not make clinical claims in any generated output.
