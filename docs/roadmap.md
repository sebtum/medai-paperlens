# Roadmap: Phases 7–11

This document fixes the **order and goals** of the post-MVP phases. It is not a
decision record: each phase's ADRs are written just in time, during that phase's
own planning, and committed in the same PR as the code. Decisions that depend on
measurements (parser, embedder, reranker) are recorded *after* measuring, following
the pattern of [ADR-0015](adr/0015-default-model-qwen3-06b-no-thinking.md).

## Baseline (Phase 6 complete)

The full stack (FastAPI → LangGraph → Qdrant/Ollama → Streamlit, Docker Compose)
works end to end, but it is not yet a meaningful RAG system:

- **Abstract-only corpus.** `data/papers/*.json` holds 10 abstracts, indexed as one
  vector per paper. Retrieval is effectively "pick 1 of 10".
- **Weak embeddings for long text.** `all-MiniLM-L6-v2` is tuned for short inputs.
- **No reranking**, no keyword/hybrid search.
- **Blocking UI.** `POST /query` returns only after generation finishes, and the
  UI shows a spinner for the whole duration.
- **Stub grounding check.** `validate_node` sets `grounded = True` whenever
  citations exist and the answer is non-empty.
- **No retrieval eval.** `scripts/eval_llm.py` produces a human-scored sheet over
  hardcoded (faked) retrieval. There are no retrieval metrics.
- **Small default model.** `qwen3:0.6b` is fast enough for query rewriting but too
  weak for multi-chunk synthesis with correct citations (to be quantified in Phase 11).
- **Open:** `pip-audit` CVE drift in transitive dependencies.

## Cross-phase decisions (fixed now)

These shape more than one phase, so they are settled up front.

- **Corpus.** Keep the same 10 papers but use their full arXiv PDFs. PDFs stay
  gitignored. A committed manifest (arXiv IDs + URLs) and a download script make
  the corpus reproducible. Public papers only, no PHI
  ([ADR-0003](adr/0003-avoid-patient-data-and-clinical-decision-support.md)).
- **Chunk identity model.** Every chunk carries `paper_id`, `section`, `page`, and
  `chunk_index`. Citations display paper + section + page. Evaluation labels are
  at **paper + section** level, so they survive re-chunking experiments.
- **LLM roles.** Ollama remains the default and only required provider
  ([ADR-0007](adr/0007-use-ollama-as-default-local-llm-provider.md)). Gemini via a
  Google AI Studio API key stays opt-in
  ([ADR-0016](adr/0016-optional-hosted-llm-provider.md)) and is used as the demo
  generator, the eval judge, and the drafter of candidate golden-set questions.
- **Observability and evaluation via LangSmith (opt-in).** Tracing is enabled
  with `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY` (optionally
  `LANGSMITH_PROJECT`); LangGraph nodes are traced without code changes.
  Evaluation runs as LangSmith experiments over versioned LangSmith datasets.
  Off by default: the app runs fully offline without it, evaluation requires an
  account, and the test suite never depends on it. The `langsmith` SDK is already
  an indirect dependency via langgraph.
- **Measure before improving.** Retrieval improvements (reranking, hybrid search)
  come only after the retrieval eval exists, so every change ships with a
  measured delta.

## Phases

| # | Phase | Goal | Expected ADR(s) |
|---|---|---|---|
| 7 ✓ | Streaming | Tokens and step status (rewriting, retrieving, generating) are visible in the UI as they happen. `POST /query` stays unchanged for tests and eval. Opt-in LangSmith tracing for per-node latency and inputs/outputs. | SSE `POST /query/stream` endpoint; `stream()` on the LLM provider interface; LangSmith for observability and evaluation |
| 8 | Full-text ingestion | Parse PDFs (Docling), chunk by document structure, upgrade the embedding model, and show section/page citations. | Parser choice (Docling vs PyMuPDF4LLM); chunking strategy; embedding model |
| 9 | Retrieval evaluation | Golden set of 40–60 questions (Gemini-drafted, human-filtered) stored as a LangSmith dataset. Each retrieval configuration runs as a LangSmith experiment. Recall@k, MRR, and nDCG@k are custom evaluator functions passed to `evaluate()`, since there are no built-in metrics for paper + section labels. | Golden-set methodology; dataset versioning |
| 10 | Rerank and hybrid search | Retrieve top 20, rerank with a cross-encoder, keep top 5. Optionally BM25 + dense fusion (RRF) in Qdrant. Each change is justified by Phase 9 numbers. | Reranker model; whether hybrid search is worth it |
| 11 | Generation and safety eval | LangSmith LLM-as-judge evaluators (Gemini as judge model) scoring faithfulness, citation correctness, and relevance. Human ratings collected through LangSmith annotation queues, replacing the CSV sheet from [`scripts/eval_llm.py`](../scripts/eval_llm.py) as the calibration source; judge-human agreement reported as Cohen's κ. Precision/recall of the safety filter. Replace the stub `validate_node` based on findings. | Judge model, rubric and calibration method (amends ADR-0016); grounding validator design |

### Exit criteria (every phase)

- All five quality gates pass (pytest, ruff, mypy, bandit, pip-audit).
- Tests use mock LLMs only. No live Ollama or Gemini in the suite.
- The README Docker Compose demo path still works.

## Open questions

Decided in each phase's own discussion, not now:

- Chunk size and overlap (Phase 8)
- `bge-small-en-v1.5` vs `bge-base-en-v1.5` (Phase 8, confirmed by Phase 9 metrics)
- **Domain-specific embedding model (Phase 8, measured in Phase 9).** The Phase 8
  default is a general model (`bge-*`). When evaluation starts, also test a model
  trained on biomedical text or literature and compare it on the same golden set
  instead of assuming it wins. Candidates to check (confirm they still exist and
  check their licences first):
  - `MedCPT`: trained on PubMed search logs. It is asymmetric, with separate query
    and article encoders, so `embed()` would need a query/document split.
  - PubMedBERT-based sentence embeddings (for example `NeuML/pubmedbert-base-embeddings`).
  Things the choice depends on: the input unit (abstracts versus full-text chunks,
  mind the 512-token limit), the vector dimension (`VECTOR_DIM` and the Qdrant
  collection change, so a re-ingest with `--overwrite` is needed), and CPU latency
  in the local-first setup. The corpus is only 10 papers, so differences may be
  within noise until it grows.
- Whether query rewriting helps retrieval (measured in Phase 9)
- Reranker candidates: `ms-marco-MiniLM-L-6-v2`, `bge-reranker-base`, optionally a
  Jina reranker (non-commercial license) (Phase 10)
- Which Gemini model name is current at implementation time (`GEMINI_MODEL` is
  configurable)

## Out of scope / parked

- **JEPA / VL-JEPA.** A vision-language research architecture with no off-the-shelf
  text reranker. A cross-encoder already gives a single-score relevance judgment
  without token generation.
- **Langfuse.** Open-source, self-hostable alternative to LangSmith. Considered
  but not chosen: it adds Postgres, ClickHouse, Redis and its own server to the
  Compose stack. Revisit if offline evaluation becomes a requirement.
- **Kubernetes.** Parked per [ADR-0002](adr/0002-docker-before-k8.md).
- **Expanding the corpus beyond 10 papers.** Revisit after Phase 10.
