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
