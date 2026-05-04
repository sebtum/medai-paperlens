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
Streamlit UI
 ↓
FastAPI Backend
 ↓
Constrained Agentic RAG Workflow
 ├── Query Classifier
 ├── Query Rewriter
 ├── Paper Retriever
 ├── Evidence Extractor
 ├── Answer Generator
 └── Grounding Validator
 ↓
Citation-grounded Research Summary
```

## 5. Core Components

### Streamlit UI

Provides a lightweight demo interface.

Responsibilities:

- collect user research question
- call FastAPI backend
- display answer, citations, confidence, and grounding status

The UI should not contain core business logic.

### FastAPI Backend

Exposes the system through HTTP endpoints.

Initial endpoints:

```text
GET /health
POST /query
```

Responsibilities:

- validate requests
- call the RAG workflow
- return structured responses
- keep UI and backend separated

### Constrained Agentic RAG Workflow

The MVP uses explicit workflow steps instead of a fully autonomous agent.

```text
classify_query
 → rewrite_query
 → retrieve_papers
 → extract_evidence
 → generate_answer
 → validate_grounding
 → return_response
```

A one-step retry may be added later if grounding is weak.

### Retrieval Layer

The retrieval layer searches a small controlled corpus of public medical AI papers or abstracts.

Planned stack:

```text
documents
 → chunking
 → embeddings
 → Qdrant
 → top-k retrieval
```

Retrieved chunks must include citation metadata.

### LLM Provider

Ollama is the default local LLM provider.

External providers may be added later behind an interface, but they must be disabled by default.

Tests should use a mock provider instead of requiring a live LLM.

### Grounding Validator

Checks whether generated answers are supported by retrieved evidence.

Responsibilities:

- verify citations exist
- flag unsupported claims
- lower confidence when evidence is weak
- prevent hallucinated paper details from being presented as facts

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

Planned response:

```json
{
  "answer": "...",
  "citations": [
    {
      "title": "...",
      "source_url": "...",
      "chunk_id": "...",
      "evidence": "..."
    }
  ],
  "confidence": "low|medium|high",
  "grounded": true,
  "debug": {
    "route": "paper_search",
    "rewritten_query": "...",
    "retrieved_chunks": 5
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
  "answer": "This system does not provide diagnosis, treatment recommendations, or patient-specific medical advice.",
  "citations": [],
  "confidence": "low",
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

