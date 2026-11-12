# MedAI PaperLens

MedAI PaperLens is a local-first AI engineering portfolio project for medical AI literature intelligence.

It answers research questions over a curated corpus of full-text public medical AI papers. Answers are streamed as they are generated and cite the specific paper, section and page they draw on. Every retrieval and generation choice is backed by measurement rather than intuition.

## Goals

- **Real document RAG.** Full-text PDFs are parsed into structure-aware chunks, not abstracts, so retrieval has to find the right passage inside a paper rather than the right paper among a handful.
- **Grounded answers.** Each claim points back to a paper, section and page, and a grounding validator checks that the cited evidence supports the answer.
- **Retrieval quality.** Dense vector search is combined with keyword search and a cross-encoder reranker, so that exact model names and acronyms (MAIRA-2, CONCH, Med-PaLM M) are found as reliably as paraphrased concepts.
- **Measurable evaluation.** A hand-labeled golden question set drives retrieval metrics (Recall@k, MRR, nDCG@k) across configurations. An LLM-as-judge scores faithfulness, citation correctness and relevance, and is itself calibrated against human ratings. The safety filter is evaluated for precision and recall. Datasets, experiments and traces are tracked in LangSmith.
- **Responsive UX on modest hardware.** Answers stream token by token with visible pipeline steps. A small local model is the default, and a hosted model is available as an opt-in.
- **Local-first and safe by design.** Runs fully offline with Ollama. External providers are opt-in only. Public papers only, never patient data.

## Scope

The corpus focuses on medical AI papers related to:

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
Ingestion
 arXiv PDFs → document parsing → structure-aware chunks
            → embeddings (+ keyword index) → Qdrant

Query
 Streamlit UI  (streamed answer + citations)
  ↓
 FastAPI Backend
  ↓
 LangGraph Workflow
  ├── Query Classifier    — safety filter for personal medical advice
  ├── Query Rewriter      — retrieval query improvement
  ├── Paper Retriever     — hybrid search over chunks
  ├── Reranker            — cross-encoder relevance scoring
  ├── Answer Generator    — LLM synthesis over retrieved evidence
  └── Grounding Validator — checks the answer against its citations
  ↓
 Citation-grounded Research Summary

Evaluation
 golden question set → retrieval metrics per configuration
 LLM-as-judge (calibrated against human ratings) → generation quality
 labeled safe/unsafe questions → safety filter precision/recall
```

## Tech Stack

- Python 3.14
- FastAPI
- LangGraph
- Qdrant (dense + sparse vectors)
- Ollama (local LLM, default model: `qwen3:0.6b`), with an opt-in Gemini provider
- Docling (PDF parsing and chunking)
- sentence-transformers (embeddings and cross-encoder reranking)
- Streamlit
- LangSmith (tracing and evaluation, opt-in)
- Docker Compose (full local stack)
- GitHub Actions CI
- pytest / ruff / mypy / bandit / pip-audit

## Documentation

- [`docs/architecture.md`](docs/architecture.md) — system design
- [`docs/adr/`](docs/adr/) — architecture decision records
- [`docs/benchmarks.md`](docs/benchmarks.md) — LLM speed measurements
- [`docs/roadmap.md`](docs/roadmap.md) — project direction

## Running with Docker Compose

The easiest way to run the full stack — no manual setup required:

```powershell
docker compose up --build
```

This starts Qdrant, Ollama (pulling `qwen3:0.6b` on first run), a one-shot
ingest job that loads the paper corpus into Qdrant, the FastAPI backend, and
the Streamlit UI, all wired together automatically. First run takes a few
minutes (model pull); later runs are fast since the Qdrant/Ollama/embedding
caches persist in Docker volumes.

Once it's up:

- UI: http://localhost:8501
- API: http://localhost:8000 (`/health`, `/query`, `/query/stream`)
- Qdrant dashboard: http://localhost:6333/dashboard

Stop everything with `docker compose down` (add `-v` to also wipe the cached
model/data volumes).

## Running locally (without Docker)

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
  "citations": [
    {"title": "...", "authors": ["..."], "year": 2024, "doi": "10.48550/arXiv...", "excerpt": "..."}
  ],
  "confidence": 0.87,
  "grounded": true,
  "debug": {"route": "retrieval"}
}
```

Unsafe personal-advice queries (e.g. "diagnose me", "my symptoms") return a refusal with `"grounded": false`.

`POST /query/stream` — same request body as `/query`; returns Server-Sent Events (`event:` = type, `data:` = JSON):

| Event | Payload |
|---|---|
| `step_start` | `{"node"}` |
| `step_end` | `{"node", "elapsed_ms"}` |
| `citations` | `{"citations", "confidence"}` (after retrieve) |
| `token` | `{"text"}` |
| `error` | `{"message"}` |
| `final` | the full `/query` response; authoritative, replaces streamed text |

Unsafe queries emit no `token` events; `final` carries the refusal. A client disconnect cancels the workflow and the LLM request. Try it: `curl -N -X POST localhost:8000/query/stream -H "content-type: application/json" -d "{\"question\": \"vision transformers for chest X-rays\"}"`.

Tracing is opt-in: set `LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY` (and optionally `LANGSMITH_PROJECT`). Unset, no client is created and nothing leaves the machine.
