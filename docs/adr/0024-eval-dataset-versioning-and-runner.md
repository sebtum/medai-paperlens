# ADR-0024: Eval dataset versioning and retrieval eval runner

## Status

Accepted (Phase 9 Plan). Extends ADR-0019 (its Phase 9 scope: datasets and evaluation).

## Context

ADR-0019 adopted LangSmith (opt-in) for tracing and left datasets and evaluation to
Phases 9 and 11. The golden set (ADR-0023) must stay reviewable in git, the eval must run
with no LangSmith account (tests and CI never use one), and LangSmith's free tier keeps
experiment traces for only about 14 days. Some sweeps change the embedder, which must not
overwrite the dev collection.

## Decision

**Source of truth = git.** `data/eval/golden.jsonl` is edited only in git; the LangSmith
dataset is never edited in its UI.

**Sync** (`scripts/sync_golden.py`) to the dataset `paperlens-retrieval-golden`:
- It hashes the file (sha256). If the dataset exists, its example ids equal the file's and
  every example's metadata `golden_sha256` equals the hash, it does nothing. (Changed in
  Execute: `langsmith` 0.12.5 cannot update dataset metadata after creation, so the hash is
  stored on each example, not on the dataset.)
- Otherwise it creates new examples and updates existing ones (example id = `uuid5` of the
  entry id, so ids are stable), deletes those no longer in the file, and tags the dataset
  version with the short git hash. Writing the hash on every example also lets an
  interrupted sync repair itself on the next run.
- It validates the file with `validate_golden` first and refuses an invalid set.
- It refuses to run while `golden.jsonl` has uncommitted changes, so every tag matches a
  commit.

**Runner** (`scripts/eval_retrieval.py`, core in `app/evaluation/`). It runs retrieval only:
optional rewrite → embed → `search_hits()`, with no generation.
- *Local mode* (default, offline apart from Qdrant):
  - each run writes `data/eval/runs/<name>.json` (config, golden hash + git hash, chunk
    count, per-question ranked sections with scores) and a markdown report in
    `data/eval/reports/`;
  - runs, reports and comparisons are committed, so the numbers outlive LangSmith
    retention;
  - `compare` and `curve` work from the JSON files alone.
- *LangSmith mode* (`--langsmith`): `aevaluate` with the same target. The evaluators are thin
  adapters over the same metric functions, so both modes give the same numbers, and the run
  config goes into the experiment metadata.
- A run fetches the top 20 at threshold 0 once. Every (k, threshold) is computed from that
  list afterwards; this is exact because a collection this small is searched exhaustively.

**Stale labels fail loudly.** Before scoring, a pre-flight scrolls the target collection's
payloads. It fails the run, listing every problem, on: an unknown `(paper_id, section)`
label, an unknown `source_chunks` ref, a `table` entry without a table source chunk, or an
`embedding_model` that differs from the run's embedder. CI has no Qdrant, so it runs only a
schema test of `golden.jsonl`.

**Rewrites are cached.** `data/eval/rewrites.jsonl` is committed, keyed by question id and
hashes of the question, rewrite model and prompt. Rewrites are generated once at
temperature 0 / seed 0 through the app's own `make_rewrite_node`. Any fallback to the
original question is a run failure: it would silently turn "rewrite on" into "rewrite off".

**One collection per embedder.** `papers` stays the app collection. Other embedders are
ingested into `eval_<model-slug>` through a new `collection_name` keyword on `ingest()` and
`search()`/`search_hits()` and `scripts/ingest.py --collection`; the defaults are unchanged.

## Alternatives Considered

### LangSmith dataset as the source of truth

Pros:
- Edits in a UI, built-in versioning.

Cons:
- Not reviewable in a PR; needs an account for every run; history can be lost with
  vendor changes.

### Committed section index for label validation in CI

Pros:
- Catches stale labels in CI.

Cons:
- CI has no parsed cache to regenerate it, so it would drift from the real index.

### Re-ingest `papers` with `--overwrite` per embedder

Pros:
- No new parameter.

Cons:
- Overwrites the dev collection and needs a re-ingest back after every sweep.

### No rewrite cache, temperature 0 only

Pros:
- Less state.

Cons:
- Ollama output can still vary across versions and hardware, and every sweep would need
  a running Ollama.

## Consequences

Positive:
- The eval runs fully offline (Qdrant only); comparisons reproduce from committed files.
- LangSmith adds experiment views without being required.

Negative:
- Committed run files grow with each experiment (small: no text, about 50 × 20 hits).
- `ingest()` and `search()` gain a parameter that only the eval uses today.

## Follow-up

- Phase 11 reuses the dataset and sync for generation and faithfulness evaluation.
