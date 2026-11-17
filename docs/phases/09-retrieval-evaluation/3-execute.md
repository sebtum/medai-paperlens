# Phase 9: Retrieval evaluation — Execute log

Wave 1 (G1: T1, T2, T3, T4) ran as four subagents in one working tree, 2026-11-16.
Nothing is committed. The checkboxes in `2-plan.md` are not ticked yet.

## Verification of the combined tree (run by the orchestrating session)

| Gate | Result |
|---|---|
| `pytest tests/` | 289 passed |
| `ruff check .` | all checks passed |
| `mypy app/` | no issues, 36 files |
| `bandit -r app/` | no issues |
| `pip-audit` | not run (no dependency added) |

Per-task re-runs by the orchestrator: T1 35 passed, T2 46 passed, T3 46 passed (`tests/test_retrieval.py`).
The T4 count (19 passed) is from the agent's own report; the 289 total includes it.

## Task results

| Task | Status | Files | Tests |
|---|---|---|---|
| T1 Metric functions | done | `app/evaluation/__init__.py`, `app/evaluation/metrics.py`, `tests/test_eval_metrics.py` | 35 |
| T2 Golden schema and loader | done | `app/evaluation/golden.py`, `tests/test_eval_golden.py` | 46 |
| T3 `collection_name` + scored hits | done, 1 deviation | `app/retrieval/search.py`, `app/retrieval/ingestion.py`, `scripts/ingest.py`, `tests/test_retrieval.py` (+265 lines) | 46 in file |
| T4 LangSmith dataset sync | done, 1 deviation | `app/evaluation/langsmith_sync.py`, `scripts/sync_golden.py`, `tests/test_eval_sync.py` | 19 |

### T1 — `app/evaluation/metrics.py`
- Types: frozen `RankedHit(paper_id, section, chunk_index, chunk_type, score)`, `SectionKey = (paper_id, section)`, `Labels = Mapping[SectionKey, int]` (grade 1 or 2).
- Functions: `cut`, `collapse_sections`, `recall_at_k`, `mrr_at_k`, `ndcg_at_k`, `is_empty`, `distinct_papers`, `table_hit`. Scoring functions take `(hits, labels, k, threshold)`.
- Threshold is inclusive (`score >= t`). Collapsing keeps the first occurrence of each `(paper_id, section)`; the same section name in two papers counts twice.
- Additions beyond the plan: empty labels also raise in `table_hit`; a grade outside {1, 2} raises `ValueError`.
- Feeding from T2: `{(l.paper_id, l.section): l.grade for l in entry.relevant}`.
- Tests use hand-computed fixtures: duplicate sections, grade-1-only hit, IDCG cut at k, hit below threshold, empty labels.

### T2 — `app/evaluation/golden.py`
- Models: `QuestionType` (StrEnum), `QUOTAS`, `Label`, `GoldenEntry` (both frozen, `extra="forbid"`, so a field carrying paper text is rejected at load).
- `load_golden(path)` raises `GoldenFormatError` with `path:line:`. `golden_sha256(path)` hashes the raw bytes. `load_manifests(papers_dir)` and `derive_aliases(manifests)` were added.
- `validate_golden(entries, manifests, *, extra_aliases=())` returns all violations: unique ids, quotas, ≥ 3 entries per paper, grades, per-type rules, paraphrased alias check, banned labels. Also flags duplicate `(paper_id, section)` in one entry and malformed `source_chunks` (agent's addition).
- `Label.grade` is a plain `int`, so a bad grade is reported with the other violations instead of failing the load.
- Alias list adds Gaze2Report, DeepTumorVQA and short forms (MAIRA, LLaVA, Med-PaLM) to the plan's list. Matching is case-insensitive with word boundaries.
- Not verified: the `maira_2` appendix banned-label rule is a heuristic inferred from Phase 8 notes. T8 must check it against real section names.
- The "table entry has a table source chunk" rule is left to the T5 pre-flight.

### T3 — retrieval
- `ingest(..., *, collection_name=COLLECTION_NAME)` is threaded through prepare, stale check, delete and upsert. `scripts/ingest.py --collection`.
- `search_hits(vector, client, *, top_k, score_threshold, collection_name=...) -> list[RankedHit]` is new. `search()` gained only the `collection_name` keyword.
- Tests: a non-default collection leaves a seeded `papers` untouched (every call hits only the eval collection); `search_hits` returns scores and `chunk_type`; existing `/query` tests pass unchanged.
- **Deviation:** `search()` is not built on `search_hits()`. `RankedHit` carries no title, authors, excerpt or page, so `Citation`s would differ. They share a private `_query_points()` and each parses the points itself; `search()`'s parsing is copied from the original. Cost: duplicated parsing.
- Choices: `search_hits` skips points without `paper_id`, `section` or a usable `chunk_index` (warning); a missing `chunk_type` becomes `"text"`.
- `app/retrieval/search.py` now imports `RankedHit` from `app.evaluation.metrics` (retrieval depends on the evaluation package).
- Not verified: `scripts/ingest.py --collection` has no automated test and was not run against Qdrant.

### T4 — `app/evaluation/langsmith_sync.py`, `scripts/sync_golden.py`
- Dataset `paperlens-retrieval-golden`; example id `uuid5(GOLDEN_NAMESPACE, entry id)` (the namespace must never change); inputs `{id, question}`, outputs `{type, relevant}`.
- No-op (zero writes) when the dataset's example ids equal the file's and every example carries the file's hash. Otherwise: create dataset if missing, create new examples, update existing ones, soft-delete removed ones, tag the version with the short git hash.
- Refuses a missing, untracked or dirty `golden.jsonl`; a missing `LANGSMITH_API_KEY` exits 1 (the agent ran this for real). Git and the client are injected, so tests need no network.
- `langsmith` 0.12.5 methods were checked against the installed package, and a test asserts they exist on the real `Client`. `update_dataset_tag` exists; `update_dataset` and `upsert_examples` do not.
- **Deviation 1:** the plan stores `golden_sha256` in dataset metadata, but 0.12.5 cannot update dataset metadata after creation. The hash is stored in each example's metadata instead.
- **Deviation 2:** the script also runs `validate_golden` and refuses an invalid set, so a work-in-progress file can't be synced. An opt-out flag was offered.
- The tag is the short hash of HEAD, not of the commit that last touched `golden.jsonl`, so a no-op sync doesn't re-tag.
- Not verified: behaviour against real LangSmith (whether `read_dataset_version(tag="latest").as_of` is an exact version, create/update on existing ids, soft-delete in the latest version). The first live sync will show.

## Decisions (owner sign-off, 2026-11-17)

1. T4 hash location (example metadata instead of dataset metadata): **accepted**. Plan and ADR-0024 wording updated.
2. T3 `search()` kept separate from `search_hits()`, duplicated parsing: **accepted for now**.
3. T4 `validate_golden` gate in the sync script: **kept** (no opt-out flag).
4. T1–T4 ticked in `2-plan.md` with one-line notes.

## Next

Wave 2 (G2): T5 (target, rewrite cache, pre-flight; needs T1, T2, T3) and T6 (bootstrap, comparison, reports; needs T1, T2) can run in parallel.

---

# Wave 2 (G2: T5, T6), 2026-11-17

Two subagents in one working tree, files disjoint. Nothing is committed. T5 and T6 are not ticked in `2-plan.md` yet (waiting for sign-off, as in Wave 1).

## Verification of the combined tree (run by the orchestrating session)

| Gate | Result |
|---|---|
| `pytest tests/` | 366 passed (289 from Wave 1 + 48 T5 + 29 T6) |
| `ruff check .` | all checks passed |
| `mypy app/` | no issues, 39 files |
| `bandit -r app/` | no issues |
| `pip-audit` | not run (no dependency declared), see open point 1 |

## Task results

| Task | Status | Files | Tests |
|---|---|---|---|
| T5 Retrieval target, rewrite cache, pre-flight | done, small deviations | `app/evaluation/target.py`, `tests/test_eval_target.py` | 48 |
| T6 Bootstrap, comparison, reports | done, small deviations | `app/evaluation/compare.py`, `app/evaluation/report.py`, `tests/test_eval_compare.py` | 29 |

### T5: `app/evaluation/target.py`
- `RetrievalConfig` (frozen; embedder, collection, prefix, rewrite, rewrite_model, depth=20). `retrieve_ranked(entry, config, *, embedder, client, cache=None, llm=None)` searches once at t = 0 with limit = depth via `search_hits`. The prefix is added only for embedding; the returned `query` is the text without it.
- Rewrite cache: `RewriteCache` (JSONL, keyed by question id, question sha256, model, prompt sha256). The prompt hash is taken from `app.workflow.rewrite._PROMPT` (imported, a private name). A miss runs the real `make_rewrite_node` through a wrapper with `temperature=0, seed=0`. An LLM error or empty output raises `RewriteFallbackError` and caches nothing. A miss with no LLM raises `RewriteCacheMissError`.
- `preflight(entries, client, config)` returns a `CorpusIndex` or raises one `PreflightError` listing all problems. The plan's four checks are in. Added: empty collection, points missing `paper_id` / `section` / `chunk_index`, points missing `embedding_model`.
- Overlap probe: `longest_shared_ngram`, `source_overlap` (reads chunk text via `client.retrieve`, so the pre-flight scroll stays payload-only as the plan says).
- Deviations: `RewriteCacheMiss` is named `...Error` (ruff N818); the `preflight` signature was chosen by the agent (the plan did not fix it); a rewrite equal to the question is accepted as a legitimate output; the cache file stores hashes and the rewrite, not the question text.
- Not verified: nothing ran against real Qdrant or Ollama. The fake Qdrant mimics `query_points`, `scroll`, `retrieve`. `OllamaClient` was not tested with `temperature` / `seed` (passed as kwargs, turned into `options`).

### T6: `app/evaluation/compare.py`, `report.py`
- The run-result model (T7 will write and read it): frozen `RunMeta`, `QuestionRun`, `RunRecord`; `save_run` / `load_run`. The JSON layout matches the plan's `runs/<name>.json`; hits are arrays `[paper_id, section, chunk_index, chunk_type, score]`. Joining a run to the golden set is strict (a missing or extra id, or a type mismatch, raises with the full list).
- Measures: `question_scores` (recall / mrr / ndcg at any `Cutoff(k, threshold)`, on-corpus only), `crowding`, `table_hit_rate`, `evidence_tokens`, empty rate, off-corpus no-result rate, max off-corpus top-1, `summarize_run`, `threshold_curve` (0.40 to 0.80, 21 steps).
- `paired_bootstrap_range`: numpy `default_rng(0)`, 1,000 resamples, within each type, 2.5 / 97.5 percentiles. `compare_runs` gives overall and per-type means and W/L/T. `verdict()` follows the plan: adopt only if lo > 0 and no type is clearly worse (mean delta <= -0.05 and more losses than wins).
- `report.py`: three Markdown renderers. Every header has the R9 sentence and the run config. A "few questions" note appears for any type with n < 10, and a "coarse" note for the off-corpus threshold part. Output is ASCII only (tested).
- Deviations: cut-offs are separate for A and B (so a threshold comparison is the same run at two thresholds); added `threshold_verdict` (also rejects B if the off-corpus no-result rate drops); `verdict()` does not check that the comparison is nDCG@5 at 0.6 (the caller picks it).
- **Evidence tokens gap:** no paper text is stored, so the token cost of k = 3/5/10 cannot be computed from the run fields. T6 added an optional `QuestionRun.chunk_tokens` (one number per hit). T7 must fill it at run time (the chunk text is already read for the overlap probe). Until then the report shows `n/a`.
- Not verified: only in-code fixtures; the bootstrap numbers were not checked against an independent implementation.

## Open points for the owner

1. **numpy is used directly but not declared.** `compare.py` imports numpy. It is only a transitive dependency today (in `uv.lock`, version 2.4.6). Recommendation: add `numpy` to `pyproject.toml` dependencies, then run `pip-audit`. This is a dependency change, so it is your call.
2. **T7 must fill `chunk_tokens`** (see above), otherwise the k-table has no evidence-token column. Recommendation: add this to T7's acceptance criteria.
3. **`verdict()` does not enforce nDCG@5 at 0.6.** Recommendation: leave it; the report states the cut-offs. Change only if you want a hard guard.
4. **Private import:** T5 imports `_PROMPT` from `app/workflow/rewrite.py`. Recommendation: accept; the alternative is a public name in app code for an eval need.
5. Tick T5 and T6 in `2-plan.md` after your sign-off.

## Next

Wave 3 (G3): T7 (runner CLI, needs T4, T5, T6) and T8 (golden set v0, needs T2 and T7's `sections` subcommand). T8 depends on T7, so T7 runs first, or the `sections` part is split out.
