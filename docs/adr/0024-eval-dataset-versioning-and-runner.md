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
  version with the short git hash. Writing the hash on every example lets an
  interrupted sync repair itself on the next run. If the examples are current but the
  newest version carries no git-hash tag (tagging failed), the next sync writes only the
  tag (fixed in the Phase 9 verify; before that it reported "unchanged" and left the
  version untagged).
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

## Changes during Execute

The decision above stands. The following differs from the Plan or was added while
building and running it (Phase 9 Execute, 2026-11-17).

- **Where the code lives.** The runner logic is in `app/evaluation/` (`runner.py`,
  `target.py`, `compare.py`, `report.py`, `authoring.py`, `langsmith_sync.py`,
  `langsmith_eval.py`), and `scripts/eval_retrieval.py` and `scripts/sync_golden.py` are
  thin CLIs, so mypy and bandit cover the logic. `numpy` is now a declared dependency (the
  bootstrap uses it).
- **Sync.** The hash is on each example, as described above. The sync also refuses a
  `golden.jsonl` that fails `validate_golden` (no opt-out flag) and one with uncommitted
  changes. The tag is the short git hash of HEAD. Run on 2026-11-17 it created 50
  examples on the EU LangSmith host (tag `11318b7`), and a second sync was a no-op.
- **Run files.** Besides the plan's fields, a run stores `query` (the text used),
  `ngram_overlap` and `chunk_tokens` (a token estimate per hit, so the k = 3 / 5 / 10
  report can show evidence tokens; no paper text is stored). A run is joined to the
  golden set strictly: a missing or extra id raises.
- **Rewrite cache.** Keyed as planned. Rewrites ran through the app's own
  `make_rewrite_node` at `temperature=0, seed=0` with `qwen3:0.6b`: 50 rewrites, 0
  fallbacks. `--no-llm` makes the runner read only the cache and fail on a missing
  rewrite. A rewrite equal to the question counts as a legitimate output (13 of 50).
- **Truncation exactness.** The check against a direct Qdrant top-5 query found 0
  mismatches for all four configs on all 50 questions, and stored hits equal a fresh
  search.
- **Query prefix.** The eval embedder is built with an empty provider prefix, and the
  `prefix` config adds the BGE instruction itself, so each config sets its prefix exactly
  once, even though the app default changed (ADR-0025).
- **Files and line endings.** `data/eval/CHANGELOG.md` holds the pooling log.
  `.gitattributes` pins LF line endings for `data/eval/*.jsonl` and `data/eval/**/*.json`
  so the golden sha256 is the same on every checkout.
- **LangSmith experiment.** `run --langsmith` ran for real once (twice by accident, so
  one duplicate experiment exists). Every per-question score equalled the local run; the
  runner enforces this and exits with an error otherwise. Only questions, labels,
  ranked hits and scores are sent, never paper text.
- **Runs.** `bge-base-en-v1.5` was ingested into `eval_bge-base-en-v1.5` (595 chunks,
  768 dimensions) while `papers` stayed untouched. Runs and reports are committed in
  `data/eval/runs/` and `data/eval/reports/` (v0 files carry a `_v0` suffix).
- **Not run for real:** the sync's update and delete path against LangSmith (only
  tested with a fake client), and `scripts/ingest.py --collection` outside the bge-base
  ingest.

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
