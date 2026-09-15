# MedAI PaperLens

MedAI PaperLens is a local-first AI engineering portfolio project for medical AI literature intelligence.

It accepts research questions and returns citation-grounded summaries over a constrained corpus of public medical AI papers.

## Scope

The MVP focuses on medical AI papers related to:

- chest X-ray visual question answering
- medical vision-language models
- radiology report generation
- evaluation methods and limitations

## Non-goals

This project does not provide:

- diagnosis
- treatment recommendations
- symptom triage
- patient-specific medical advice
- clinical decision support

## Architecture

```
Streamlit UI
 ↓
FastAPI Backend
 ↓
LangGraph Workflow
 ├── Query Classifier   — phrase-pattern safety filter
 ├── Query Rewriter     — Ollama-backed retrieval query improvement
 ├── Paper Retriever    — Qdrant vector search, fills citations + evidence
 ├── Answer Generator   — Ollama LLM synthesis over retrieved evidence
 └── Grounding Validator — checks citations exist and answer is non-empty
 ↓
Citation-grounded Research Summary
```

## Tech Stack

- Python 3.14
- FastAPI
- LangGraph
- Qdrant
- Ollama (local LLM, default model: `qwen3:0.6b`), with an opt-in Gemini provider
- sentence-transformers (embeddings)
- Streamlit
- Docker Compose (planned)
- GitHub Actions CI
- pytest / ruff / mypy / bandit / pip-audit

## Status

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | FastAPI `/health` + `/query` endpoints | ✅ Done |
| 2 | Query API contract + unsafe-query refusal | ✅ Done |
| 3 | Qdrant retrieval layer | ✅ Done |
| 4 | LangGraph workflow + Ollama LLM synthesis | ✅ Done |
| 5 | Streamlit UI | ✅ Done |
| 6 | Docker Compose full-stack | Next |

## Running locally

**Requirements:** Qdrant and Ollama must be running locally.

```powershell
# Start the API server
.\.venv\Scripts\uvicorn.exe app.main:app --reload

# Start the Streamlit UI
.\.venv\Scripts\streamlit.exe run ui/app.py

# Ingest papers into Qdrant
.\.venv\Scripts\python.exe scripts/ingest.py

# Run tests
.\.venv\Scripts\pytest.exe tests/
```

Environment variables (see `.env.example`):

```
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3:0.6b
OLLAMA_THINK=false
LLM_PROVIDER=ollama
QDRANT_URL=http://localhost:6333
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

## Model selection

Ollama is the required default LLM provider; `qwen3:0.6b` with thinking disabled
is the default model. See [`docs/benchmarks.md`](docs/benchmarks.md) for the
speed measurements behind that choice and
[ADR-0015](docs/adr/0015-default-model-qwen3-06b-no-thinking.md) for the
decision record. A hosted Gemini provider is available opt-in behind
`LLM_PROVIDER=gemini` and `GEMINI_API_KEY` — see
[ADR-0016](docs/adr/0016-optional-hosted-llm-provider.md).

## API

`GET /health` → `{"status": "ok"}`

`POST /query` — accepts `{"question": "..."}`, returns:

```json
{
  "answer": "...",
  "citations": [{"title": "...", "source_url": "...", "chunk_id": "...", "excerpt": "..."}],
  "confidence": 0.87,
  "grounded": true,
  "debug": {"route": "retrieval"}
}
```

Unsafe personal-advice queries (e.g. "diagnose me", "my symptoms") return a refusal with `"grounded": false`.
