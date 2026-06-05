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
 ├── retrieve_node   — Qdrant vector search over paper corpus
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
- manage Ollama client lifespan (async context manager via `app.state.ollama`)
- build and invoke the LangGraph workflow
- return structured `QueryResponse`

### LangGraph Workflow

Uses explicit, named nodes instead of a fully autonomous agent.

```text
START → classify_node
  ↓ (unsafe) → END  [returns refusal, grounded=false]
  ↓ (safe)   → rewrite_node → retrieve_node → generate_node → validate_node → END
```

State is a `WorkflowState` TypedDict passed between nodes.

### Retrieval Layer

Searches a controlled corpus of 10 public medical AI papers.

```text
question
 → embed via SentenceTransformer (all-MiniLM-L6-v2 default)
 → Qdrant top-k cosine search
 → list[Citation] + confidence score
```

Retrieved chunks include citation metadata (title, source URL, excerpt).

### LLM Provider

`LlmProvider` is a Protocol with `generate()` and `generate_structured()`.

`OllamaClient` is the only concrete implementation. Default model: `qwen2.5:3b` (configurable via `OLLAMA_MODEL`).

External providers can be added behind the protocol but must be opt-in and disabled by default.

Tests use a mock provider — no live Ollama required in the test suite.

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
QDRANT_URL=http://localhost:6333
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

