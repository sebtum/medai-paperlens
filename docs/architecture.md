# Architecture: MedAI PaperLens

## 1. Purpose

MedAI PaperLens is a local-first AI engineering project for medical AI literature intelligence.

The system helps users explore medical AI papers by retrieving relevant literature, extracting structured evidence, comparing methods, and generating citation-grounded summaries.

The MVP focuses on medical AI literature, especially:

- chest X-ray visual question answering
- medical vision-language models
- radiology report generation
- datasets, metrics, methods, and limitations in medical AI papers

This project is not a clinical decision support system.

It does not provide:

- diagnosis
- treatment recommendations
- symptom triage
- patient-specific medical advice
- medical device functionality

## 2. Architectural Goals

The architecture is designed to demonstrate production-oriented AI engineering practices:

- local-first development
- constrained agentic RAG
- citation-grounded generation
- structured evidence extraction
- clear safety boundaries
- reproducible local deployment
- automated testing
- CI/CD with GitHub Actions
- documented architecture decisions
- modular design for future extension

The project prioritizes correctness, traceability, and explainability over open-ended agent autonomy.

## 3. Non-Goals

The MVP intentionally does not include:

- patient data
- personal health information
- real-time clinical use
- autonomous web browsing
- unrestricted tool-using agents
- full deep research agent behavior
- paid APIs as required dependencies
- Kubernetes production deployment
- fine-tuning of large language models

These are excluded to keep the project safe, reproducible, and achievable within the MVP scope.

## 4. High-Level System Overview

```text
User
 ↓
Streamlit UI  [planned]
 ↓
FastAPI Backend
 ↓
LangGraph Workflow
 ├── classify_node   — phrase-pattern safety filter
 ├── rewrite_node    — Ollama-backed query rewriter (falls back to original on error)
 ├── retrieve_node   — Qdrant vector search over paper corpus (top 20)
 ├── rerank_node     — cross-encoder re-scores the 20 and keeps the best 5
 ├── generate_node   — Ollama LLM synthesis over retrieved evidence
 └── validate_node   — grounding check (citations present + answer non-empty)
 ↓
Citation-grounded Research Summary
```

## 5. Core Components

### Streamlit UI (planned — Phase 5)

Will provide a lightweight demo interface.

Responsibilities:

- collect user research question
- call FastAPI backend
- display answer, citations, confidence, and grounding status

The UI must not contain core business logic.

### FastAPI Backend

Exposes the system through HTTP endpoints.

Endpoints:

```text
GET /health
POST /query
```

Responsibilities:

- validate requests
- manage the active LLM provider's lifespan (async context manager via `app.state.llm`)
- build and invoke the LangGraph workflow
- return structured `QueryResponse`

### LangGraph Workflow

Uses explicit, named nodes instead of a fully autonomous agent.

```text
START → classify_node
  ↓ (unsafe) → END  [returns refusal, grounded=false]
  ↓ (safe)   → rewrite_node → retrieve_node → rerank_node → generate_node → validate_node → END
```

State is a `WorkflowState` TypedDict passed between nodes.

### Retrieval Layer

Searches a controlled corpus of 10 public medical AI papers, indexed as full-text chunks
(about 595 in total). PDFs are parsed on the host with Docling into `data/parsed/`; ingest
chunks them with Docling's HybridChunker and indexes one Qdrant point per chunk
(ADR-0020, ADR-0021). Without a parsed file a paper falls back to its abstract.

```text
question
 → embed via SentenceTransformer (BAAI/bge-small-en-v1.5 default),
   with a query prefix in front of the question (BGE models; documents are not prefixed)
 → Qdrant cosine search, top 20 (RETRIEVE_DEPTH)
 → rerank_node: cross-encoder scores (rewritten question, passage) pairs, keeps the best 5
 → list[Citation] + confidence score (cosine of the vector top hit)
```

The reranker is `cross-encoder/ms-marco-MiniLM-L-6-v2` by default (`RERANKER_MODEL`; empty or `none` switches it off, and the step then only keeps the first 5 hits). It is loaded on the first request, not at startup. If it fails, or cannot be loaded (a failed load is remembered for 60 s; only one request loads the model, others get vector order meanwhile), the vector order is kept, `debug.rerank_fallback` is set and `debug.rerank_model` is `null` (ADR-0026). SSE `citations` is sent when the rerank step ends.

Hybrid search (dense plus BM25 sparse, merged by RRF) and smaller chunks were measured as eval-only experiments and are not part of the app (ADR-0028). The app collection `papers` holds dense vectors only.

Each retrieved chunk becomes one citation with paper metadata (title, authors, year, DOI), section path, page, chunk index and the chunk text as excerpt.

The query prefix is set by `EMBEDDING_QUERY_PREFIX`. Unset, `BAAI/bge-*` models get the BGE query instruction and other models get none. It applies to queries only, so changing it needs no re-ingest (ADR-0025).

### Evaluation Layer

Retrieval is measured, not assumed (ADR-0023, ADR-0024, ADR-0025, ADR-0026, ADR-0028). The eval covers retrieval only, with no generation.

```text
data/eval/golden.jsonl  (65 questions, 164 labels = paper + section + grade)
 → runner: optional cached rewrite → embed → Qdrant top 20 at threshold 0 (or dense + BM25 by RRF)
 → optional rerank (cross-encoder, depth 20, keep 5)
 → data/eval/runs/<name>.json   (ranked hits and scores, no paper text)
 → metrics (Recall@k, MRR, nDCG@k), paired bootstrap, threshold curve
 → data/eval/reports/*.md
```

- **Code:** core logic in `app/evaluation/` (`metrics`, `golden`, `target`, `compare`, `report`, `runner`, `authoring`, `langsmith_sync`, `langsmith_eval`), so mypy and bandit cover it. `scripts/eval_retrieval.py` and `scripts/sync_golden.py` are thin CLIs.
- **One collection per embedder.** `papers` is the app collection. Other embedders are ingested into `eval_<model-slug>` (`scripts/ingest.py --collection`), so a sweep never overwrites the app index. A pre-flight check fails the run if a label points at a missing section or the collection's `embedding_model` differs from the run's.
- **Git is the source of truth for the golden set.** LangSmith holds a synced copy (dataset `paperlens-retrieval-golden`) and optional experiments. It is opt-in: the local run, the tests and CI never need an account.
- **Eval-only collections.** `eval_hybrid` (dense plus BM25 sparse vector, `ingest --hybrid`) and `eval_chunk256_merge` / `eval_chunk256_nomerge` (`ingest --chunk-tokens 256`, with and without merging neighbours) are built beside `papers`; the ingest options refuse the app collection. `run --rechunked` skips the source-chunk-number checks for a re-cut collection.
- **Score kinds.** Each run records `score_kind` (`cosine`, `rerank`, `rrf`). Only cosine runs are cut at the 0.6 threshold; reranker (0..1) and RRF scores are not cosine values, so they are not cut.
- **Rewrites are cached** in `data/eval/rewrites.jsonl`, so every sweep reuses the same rewrites and reproduces without Ollama.
- **Query prefix in the eval.** A config sets its own prefix. The eval embedder is built with an empty provider prefix, so the app default cannot add a second one.
- **Limits:** the golden set is LLM-labelled (Claude), not human-verified, and has 20 off-corpus questions (15 added in Phase 10 as near misses). Reports say so.

### LLM Provider

`LlmProvider` is a Protocol with `generate()` and `generate_structured()`. A separate
`ModelStatusProvider` Protocol (`model`, `warmup()`, `is_model_warm()`) backs
`GET /model/status`.

`OllamaClient` is the default, required implementation. Default model: `qwen3:0.6b`,
thinking mode off (configurable via `OLLAMA_MODEL` / `OLLAMA_THINK`) — see
`docs/benchmarks.md` for the speed measurements behind this default.

`GeminiClient` is an opt-in second implementation, selected via `LLM_PROVIDER=gemini`
and `app/llm/factory.py`'s `make_llm_provider()`. It stays disabled by default; the
`google-genai` SDK is an optional dependency imported lazily so the app still runs
without it installed.

Tests use a mock provider — no live Ollama or Gemini required in the test suite.

### Grounding Validator

`validate_node` sets `grounded = True` when citations are present and the answer is non-empty. A weak-grounding retry path may be added in a later phase.

## 6. API Contract

### GET /health

```json
{
  "status": "ok"
}
```

### POST /query

Request:

```json
{
  "question": "What are recent methods for chest X-ray visual question answering?"
}
```

Response:

```json
{
  "answer": "...",
  "citations": [
    {
      "title": "...",
      "source_url": "...",
      "chunk_id": "...",
      "excerpt": "..."
    }
  ],
  "confidence": 0.87,
  "grounded": true,
  "debug": {
    "route": "retrieval"
  }
}
```

Error Handling

The API should return clear errors for:

- empty question
- unsupported query type
- unsafe medical advice request
- retrieval failure
- LLM provider failure
- grounding failure

Example unsafe response:

```json
{
  "answer": "Literature summaries only — no medical advice.",
  "citations": [],
  "confidence": 0.0,
  "grounded": false,
  "debug": {
    "route": "unsafe_medical_advice"
  }
}
```

## 7. Security and Privacy Principles

The system follows these principles:

- no patient data
- no personal health data
- no secrets committed to Git
- local LLM provider by default
- external providers disabled by default
- no diagnosis or treatment recommendations
- citation metadata preserved for traceability
- user input not stored by default
- logs minimized and sanitized

### 7.1 Environment Variables

Secrets and configuration should be loaded from environment variables.

Example:

```text
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3:0.6b
OLLAMA_THINK=false
LLM_PROVIDER=ollama
GEMINI_API_KEY=
GEMINI_MODEL=gemini-2.5-flash
QDRANT_URL=http://localhost:6333
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
# EMBEDDING_QUERY_PREFIX=   (optional; unset = BGE instruction for BAAI/bge-* models)
LOG_LEVEL=INFO
```

`.env.example` may be committed.

`.env` must not be committed.

## 8. Current Architectural Constraints

The project should obey these constraints during MVP development:

- do not add Kubernetes before Docker Compose works
- do not add LangGraph before basic retrieval works
- do not add external LLM providers before local or mock provider works
- do not add complex frontend logic before the API contract is stable
- do not add web search before the local corpus is evaluated
- do not process patient data
- do not make clinical claims

These constraints prevent the project from becoming too broad too early.

