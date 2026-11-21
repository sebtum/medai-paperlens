# MedAI PaperLens

MedAI PaperLens is a local-first AI engineering portfolio project for medical AI literature intelligence.

It answers research questions over a curated corpus of full-text public medical AI papers. Answers are streamed as they are generated and cite the specific paper, section and page they draw on. Every retrieval and generation choice is backed by measurement rather than intuition.

## Goals

- **Real document RAG.** Full-text PDFs are parsed into structure-aware chunks, not abstracts, so retrieval has to find the right passage inside a paper rather than the right paper among a handful.
- **Grounded answers.** Each claim points back to a paper, section and page, and a grounding validator checks that the cited evidence supports the answer.
- **Retrieval quality.** Dense vector search fetches 20 chunks and a cross-encoder reranker keeps the best 5. Keyword (hybrid) search was measured too and not adopted, because it showed no safe gain on top of the reranker ([ADR-0028](docs/adr/0028-hybrid-search.md)).
- **Measurable evaluation.** An LLM-labelled golden question set (written by Claude, not human-verified) drives retrieval metrics (Recall@k, MRR, nDCG@k) across configurations. An LLM-as-judge scores faithfulness, citation correctness and relevance, and is itself calibrated against human ratings (planned). The safety filter is evaluated for precision and recall. Datasets, experiments and traces are tracked in LangSmith.
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
            → embeddings → Qdrant

Query
 Streamlit UI  (streamed answer + citations)
  ↓
 FastAPI Backend
  ↓
 LangGraph Workflow
  ├── Query Classifier    — safety filter for personal medical advice
  ├── Query Rewriter      — retrieval query improvement
  ├── Paper Retriever     — dense search, top 20 chunks
  ├── Reranker            — cross-encoder re-scores the 20, keeps the best 5
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
- Qdrant (dense vectors; the eval-only hybrid collection adds a sparse BM25 vector)
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
- [`data/eval/reports/`](data/eval/reports/) — retrieval evaluation reports
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

The ingest job mounts `data/parsed/` read-only. On a fresh clone that folder is
empty, so only the abstracts are indexed (one chunk per paper). For full-text
retrieval, build the parsed cache on the host first; see
[Full-text corpus](#full-text-corpus).

Once it's up:

- UI: http://localhost:8501
- API: http://localhost:8000 (`/health`, `/query`, `/query/stream`)
- Qdrant dashboard: http://localhost:6333/dashboard

Stop everything with `docker compose down` (add `-v` to also wipe the cached
model/data volumes).

## Full-text corpus

The repository holds only small manifests (`data/papers/*.json`: metadata,
abstract, versioned `arxiv_id` and `pdf_url`). The full text is built on the
host in three steps; PDFs and the parsed cache are gitignored.

```powershell
# 1. Download the arXiv PDFs to data/pdfs/ (skips existing files, --force re-downloads)
.\.venv\Scripts\python.exe -m scripts.download_papers

# 2. Parse them with Docling into data/parsed/<paper_id>.json
#    Needs the optional `pdf` extra (uv sync --extra pdf); the first run downloads
#    the layout and table models, and a paper takes from about a minute to four.
.\.venv\Scripts\python.exe -m scripts.parse_papers   # --only <paper_id>, --force

# 3. Chunk, embed and index into Qdrant (add --overwrite to re-index from scratch)
.\.venv\Scripts\python.exe -m scripts.ingest
```

Parsing runs only on the host: the Docker image has no Docling, only the chunker
(`docling-core`). A paper without a parsed file falls back to its abstract, with
a warning. Two papers are indexed from the arXiv preprint rather than the published
version (`conch`, `cxr_vlm_generation`; see `source_note` in their manifests), so the
full text may differ from the published paper. Ingest is idempotent: re-running it
replaces a paper's chunks, and it recreates the collection if the embedding model
changed. See [ADR-0020](docs/adr/0020-pdf-parser.md),
[ADR-0021](docs/adr/0021-structure-aware-chunking.md) and
[ADR-0022](docs/adr/0022-default-embedder-bge-small.md).

## Retrieval evaluation

A golden set of 65 questions and 164 labels (`data/eval/golden.jsonl`: 45 on-corpus, 20 off-corpus) scores retrieval by
Recall@k, MRR and nDCG@k over `(paper, section)` labels. The runner needs Qdrant with the
full-text corpus ingested (its pre-flight check fails on a mismatch), and Ollama only
to create new query rewrites. **The golden set is LLM-labelled:
Claude wrote and labelled it, and no human has verified it.** Treat the numbers as
indicative, not as proof. Method: [ADR-0023](docs/adr/0023-golden-set-methodology.md),
[ADR-0024](docs/adr/0024-eval-dataset-versioning-and-runner.md); outcome and the retrieval
defaults it led to: [ADR-0025](docs/adr/0025-retrieval-defaults-after-eval.md); reranker:
[ADR-0026](docs/adr/0026-reranker.md); hybrid search and chunk size (both measured, not
adopted): [ADR-0028](docs/adr/0028-hybrid-search.md).

```powershell
# Run one named config: pre-flight check, retrieve, write a run file and a report
.\.venv\Scripts\python.exe -m scripts.eval_retrieval run --config baseline --no-llm

# Compare run B with run A (paired bootstrap on nDCG@5), threshold curve of one run
.\.venv\Scripts\python.exe -m scripts.eval_retrieval compare baseline prefix
.\.venv\Scripts\python.exe -m scripts.eval_retrieval curve baseline

# Authoring helpers: list sections and chunk ids; unlabelled sections in the top 10
.\.venv\Scripts\python.exe -m scripts.eval_retrieval sections --paper maira_2
.\.venv\Scripts\python.exe -m scripts.eval_retrieval pool baseline prefix

# The bge_base config needs its own collection (the `papers` collection stays untouched)
$env:EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
.\.venv\Scripts\python.exe -m scripts.ingest --collection eval_bge-base-en-v1.5

# Phase 10 experiments, all eval only (each refuses the app collection `papers`)
.\.venv\Scripts\python.exe -m scripts.ingest --collection eval_hybrid --hybrid          # dense + BM25 sparse vector
.\.venv\Scripts\python.exe -m scripts.ingest --collection eval_chunk256_merge --chunk-tokens 256
.\.venv\Scripts\python.exe -m scripts.ingest --collection eval_chunk256_nomerge --chunk-tokens 256 --no-merge-peers
.\.venv\Scripts\python.exe -m scripts.eval_retrieval run --config rerank_minilm --collection eval_chunk256_merge --name chunk256_merge --rechunked --no-llm
```

`run` also takes `--depth N` (candidates fetched before the cut), `--rerank-query original`
and `--reranker <model>`; each needs `--name` so it does not overwrite the config's run.
`--rechunked` skips the source-chunk-number checks, because re-chunking changes the numbers
(the labelled-section check stays).

| Config | Embedder | Collection | Query prefix | Rewrite | Search | Reranker |
|---|---|---|---|---|---|---|
| `baseline` | `bge-small-en-v1.5` | `papers` | none | on | dense | none |
| `no_rewrite` | `bge-small-en-v1.5` | `papers` | none | off | dense | none |
| `prefix` | `bge-small-en-v1.5` | `papers` | BGE instruction | on | dense | none |
| `bge_base` | `bge-base-en-v1.5` | `eval_bge-base-en-v1.5` | none | on | dense | none |
| `rerank_minilm` (the app today) | `bge-small-en-v1.5` | `papers` | BGE instruction | on | dense, top 20 | MiniLM |
| `rerank_bge`, `rerank_medcpt` | same | `papers` | BGE instruction | on | dense, top 20 | bge-reranker-base, MedCPT (not used) |
| `hybrid` | `bge-small-en-v1.5` | `eval_hybrid` | BGE instruction | on | dense + BM25, RRF | none |
| `hybrid_rerank` | same | `eval_hybrid` | BGE instruction | on | dense + BM25, RRF | MiniLM |

Where things live (all committed, none holds paper text):
- `data/eval/golden.jsonl`: the questions and labels. `data/eval/CHANGELOG.md`: what
  changed between versions.
- `data/eval/rewrites.jsonl`: cached query rewrites. With `--no-llm` the runner reads
  only this file, so a run is reproducible without Ollama. A missing rewrite is an
  error.
- `data/eval/runs/<name>.json`: one file per run. `data/eval/reports/*.md`: run
  summaries, comparisons and threshold curves.

Git is the source of truth for the golden set. An optional copy goes to LangSmith as
the dataset `paperlens-retrieval-golden`:

```powershell
.\.venv\Scripts\python.exe -m scripts.sync_golden                       # needs a clean, committed golden.jsonl
.\.venv\Scripts\python.exe -m scripts.eval_retrieval run --config baseline --no-llm --langsmith
```

LangSmith setup, optional (the local run needs none of it):
1. Create an account and an API key.
2. Put `LANGSMITH_API_KEY=...` in `.env` (never commit it).
3. On the EU region, also set `LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com`
   (read by the LangSmith SDK).
4. Run the two commands above. `--langsmith` also logs an experiment and checks that
   its scores equal the local run. Only questions, labels and scores are sent, never
   paper text.

## Running locally (without Docker)

**Requirements:** Qdrant and Ollama must be running locally.

```powershell
# Start the API server
.\.venv\Scripts\uvicorn.exe app.main:app --reload

# Start the Streamlit UI
.\.venv\Scripts\streamlit.exe run ui/app.py

# Ingest papers into Qdrant
.\.venv\Scripts\python.exe -m scripts.ingest

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
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
OLLAMA_NUM_CTX=8192
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
# EMBEDDING_QUERY_PREFIX=   (optional, see below)
```

- `EMBEDDING_MODEL`: the embedder for both ingest and queries. The vector size
  comes from the model. Ingest and the API must use the same value; the API logs
  a warning at startup if the indexed `embedding_model` differs. Changing it needs a
  re-ingest (done automatically, since ingest recreates a mismatched collection).
- `EMBEDDING_QUERY_PREFIX`: optional text put in front of queries only, never in
  front of documents, so changing it needs no re-ingest. Unset: `BAAI/bge-*` models get
  `Represent this sentence for searching relevant passages: `, other models get none.
  Set it (to an empty string to switch the prefix off) to override. A non-BGE model
  with its own query instruction needs it set. It is a plain setting, not a secret
  ([ADR-0025](docs/adr/0025-retrieval-defaults-after-eval.md)).
- `RERANKER_MODEL`: the cross-encoder that re-scores the 20 retrieved chunks and keeps
  the best 5 (the "rerank" step). Unset: `cross-encoder/ms-marco-MiniLM-L-6-v2`. Empty or
  `none`: reranking off, the step only keeps the first 5 hits. The model (91 MB of weights)
  downloads from Hugging Face on first use into the Hugging Face cache (the shared
  `huggingface_cache` volume under Compose) and loads on the first request, so that
  request is slower; there is no warm-up at startup. On the evaluation machine reranking
  20 chunks took a median of 1.6 s ([ADR-0026](docs/adr/0026-reranker.md)).
- `OLLAMA_NUM_CTX`: context window sent to Ollama on every call, warmup included,
  so the model is not reloaded on the first query. It has to fit the retrieved chunks.

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
    {"title": "...", "authors": ["..."], "year": 2024, "doi": "10.48550/arXiv...", "excerpt": "...",
     "paper_id": "maira_2", "section": "Methods > RadFact", "page": 5, "chunk_index": 12}
  ],
  "confidence": 0.87,
  "grounded": true,
  "debug": {"route": "retrieval", "rewritten": "...", "rerank_model": "cross-encoder/ms-marco-MiniLM-L-6-v2", "rerank_top_score": 0.95, "rerank_fallback": false}
}
```

Each citation is one retrieved chunk, so several citations can come from the same paper. The citations are the 5 chunks the reranker kept, best first. `paper_id`, `section`, `page` and `chunk_index` are optional (`null` for an abstract-only paper or an old index), and `excerpt` is the chunk text. `confidence` is the top cosine similarity of the 20 retrieved chunks (the reranker does not change it). It is not a calibrated probability, and its scale depends on the embedding model. In `debug`, `rerank_model` and `rerank_top_score` are `null` when reranking is off, and `rerank_fallback` is `true` when the reranker failed and the vector order was kept (the request still succeeds).

Unsafe personal-advice queries (e.g. "diagnose me", "my symptoms") return a refusal with `"grounded": false`.

`POST /query/stream` — same request body as `/query`; returns Server-Sent Events (`event:` = type, `data:` = JSON):

| Event | Payload |
|---|---|
| `step_start` | `{"node"}` |
| `step_end` | `{"node", "elapsed_ms"}` |
| `citations` | `{"citations", "confidence"}` (after rerank: the final 5) |
| `token` | `{"text"}` |
| `error` | `{"message"}` |
| `final` | the full `/query` response; authoritative, replaces streamed text |

The steps run in this order: `classify`, `rewrite`, `retrieve`, `rerank`, `generate`, `validate` (the UI shows the rerank step as "Reranking passages"). Unsafe queries emit no `token` events; `final` carries the refusal. A client disconnect cancels the workflow and the LLM request. Try it: `curl -N -X POST localhost:8000/query/stream -H "content-type: application/json" -d "{\"question\": \"vision transformers for chest X-rays\"}"`.

Tracing is opt-in: set `LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY` (and optionally `LANGSMITH_PROJECT`). Unset, no client is created and nothing leaves the machine.
