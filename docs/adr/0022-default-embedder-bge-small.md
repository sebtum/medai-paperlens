# ADR-0022: Default embedder `bge-small-en-v1.5`, dimension from the model

## Status

Accepted (Phase 8 Plan). Extends ADR-0012.

## Context

`all-MiniLM-L6-v2` was chosen for abstract-only retrieval. Its position embeddings
cover 512 tokens, but its training used inputs of about 128 tokens and sentence-transformers
truncates it at 256. Phase 8 indexes full-text chunks that fill the embedder's 510-token
budget (ADR-0021), which MiniLM would cut short. `VECTOR_DIM = 384` was a hard-coded constant in
`app/retrieval/ingestion.py`, so changing the model meant editing code. The model must
stay small enough for CPU-only ingest and query latency (A3).

## Decision

- Default `EMBEDDING_MODEL` = `BAAI/bge-small-en-v1.5` (384 dim, 512-token limit, about
  33M parameters). Overridable through the `EMBEDDING_MODEL` env var, as before. The
  default is defined in `app/retrieval/embedding.py`.
- The vector dimension comes from `model.get_sentence_embedding_dimension()` and the
  chunk limit from `model.max_seq_length`. The `VECTOR_DIM` constant is removed.
- Each point stores `embedding_model` in its payload. If an existing collection has a
  different vector size or a different `embedding_model`, ingest logs a warning and
  recreates the collection. Old MiniLM volumes have the same 384 dim, so the name check
  is needed to catch them.
- In Docker Compose, `ingest` and `api` read the same `EMBEDDING_MODEL` variable, so
  query and document vectors always come from the same model.
- No bge query instruction prefix yet. Splitting `EmbeddingProvider` into query and
  document embedding is Phase 9.

## Alternatives Considered

### Keep `all-MiniLM-L6-v2`

Pros:
- No re-index, already cached.

Cons:
- Chunks would be truncated at 256 tokens. Weaker on retrieval benchmarks.

### `bge-base-en-v1.5` (768 dim)

Pros:
- Usually better retrieval quality.

Cons:
- About 3× slower on CPU, with double the vector size. To be decided on Phase 9 numbers.

### Domain-specific biomedical embedder

Pros:
- May handle medical terminology better.

Cons:
- Unmeasured. Phase 9 needs a golden set before we can choose one.

## Consequences

Positive:
- Chunks embed without truncation. The model can be swapped with an env var, no code
  change.
- Stale collections heal themselves on the next ingest.

Negative:
- Scores shift: bge cosine scores cluster higher than MiniLM's. The `score_threshold`
  default went from 0.4 (tuned for MiniLM) to **0.6**, measured in Phase 8 Execute: ten
  on-topic questions scored 0.74–0.92 in the top 5, three off-topic ones 0.44–0.53, so 0.4
  would have let every off-topic question through. The on-topic questions name their
  paper, which inflates their scores; Phase 9 tunes the value. `confidence` must not be
  shown as a calibrated probability (the UI says "not calibrated").
- A stale `EMBEDDING_MODEL` in a local `.env` (still MiniLM, also 384-dim) broke retrieval
  silently once: ingest used bge-small while the API queried with MiniLM, and nothing failed.
  `scripts/ingest.py` now loads `.env` like the API, and `app/retrieval/consistency.py`
  logs a warning at API startup when the indexed `embedding_model` differs from the
  configured one.
- The model is downloaded on first run (about 130 MB).
- The Docker image installs the CPU build of torch (Linux only, `[tool.uv.sources]` in
  `pyproject.toml`): the default CUDA wheels added 1.7 GB of NVIDIA libraries to an image
  that never uses a GPU. `torch` is therefore a direct dependency, pinned `<2.15`.

## Follow-up

- Phase 9: bge-small vs bge-base vs a domain embedder on Recall@k / nDCG; query prefix;
  threshold tuned on metrics.
