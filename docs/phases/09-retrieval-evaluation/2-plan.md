# Phase 9: Retrieval evaluation — Plan

## Decisions (resolving U2, U4, U6 and open method details)
- **U2 label validation = pre-flight against Qdrant at run time.** Before scoring, the runner
  scrolls the whole target collection (payload only: `paper_id, section, chunk_index,
  chunk_type, embedding_model`; about 600 points) and fails loudly (R5), listing every problem, if:
  a labelled `(paper_id, section)` doesn't exist; a `source_chunks` ref doesn't exist; a
  `table` entry has no source chunk of `chunk_type == "table"`; or the collection's
  `embedding_model` differs from the run's embedder (catches the Phase 8 stale-`.env` bug).
  The same scroll gives the chunk count for R4. CI has no Qdrant or parsed cache, so CI runs
  only the schema test (R6). A committed section index was rejected: CI couldn't regenerate it,
  so it would only drift.
- **U4 one collection per embedder.** `papers` stays the app/dev collection (bge-small, also
  used by the baseline). Other embedders go into `eval_<model-slug>` (e.g.
  `eval_bge-base-en-v1.5`). `ingest()` and `search()` get a keyword
  `collection_name: str = COLLECTION_NAME`; `scripts/ingest.py --collection`. The default
  behaviour is unchanged.
- **U6 rewrites are cached and committed.** `data/eval/rewrites.jsonl`, keyed by
  `(question id, sha256(question), rewrite model, sha256(rewrite prompt))`. A miss calls the real
  `make_rewrite_node` (`app/workflow/rewrite.py`) through a small wrapper that passes
  `temperature=0, seed=0` (OllamaClient turns kwargs into `options`) and records failures.
  `rewrite_node` silently falls back to the original question on error, which would make
  rewrite-on equal rewrite-off, so the eval treats any fallback as a **run failure**. All sweeps
  reuse the same rewrites, so they are deterministic and reproducible without Ollama.
- **One retrieval per config, cut-offs after the fact.** A "run" = (embedder, prefix, rewrite):
  top-20 at threshold 0, scores kept. Any (k, threshold) = keep hits with score ≥ t, then the
  first k. This is exact because a collection this small is searched exhaustively (Qdrant's
  full-scan threshold). T9 checks it once against a direct top-5 query. Runs: `baseline`
  (bge-small, no prefix, rewrite on), `no_rewrite`, `prefix`, `bge_base`. Four runs × ~50 questions.
- **Bge prefix (owner decision).** The eval target puts
  `"Represent this sentence for searching relevant passages: "` in front of the (rewritten)
  query text. No `EmbeddingProvider` change during the experiment. If the prefix wins, the app
  gets `SentenceTransformerProvider(model, query_prefix=...)` + `EMBEDDING_QUERY_PREFIX`,
  recorded in the results ADR. The Discuss file's separate split ADR is dropped.
- **Metrics (pure, `app/evaluation/metrics.py`).** Ranked list L = hits with score ≥ t, cut to k
  chunks, then collapsed to unique `(paper_id, section)` at first rank; rank = position in L.
  - Recall@k = |L ∩ R| / |R|.
  - MRR@k = 1/rank of the first labelled section (any grade), else 0.
  - nDCG@k: gain = grade (2/1), discount log2(rank+1); IDCG from the labels sorted by grade,
    cut to k.
  - Off-corpus entries are left out of these three. They are reported as the no-result rate
    (share with L empty at t). The on-corpus empty-result rate is reported next to it.
  - Measurements only: crowding = mean distinct papers in the top-k chunks; table hit rate =
    share of `table` entries whose top-k has a `table` chunk in a labelled section.
- **Decision rule (from Discuss, made concrete).** Primary metric **nDCG@5** at t = 0.6 (owner).
  Paired bootstrap of mean(B − A): 1,000 resamples, numpy `default_rng(0)` (stdlib `random`
  trips bandit B311), indices resampled within each type, 95 % percentile range. A type is
  "clearly worse" if its mean delta ≤ −0.05 **and** it has more losses than wins. A per-question
  tie = |delta| < 1e-9. B replaces A only if lo > 0 and no type is clearly worse.
  - **top_k (owner decision): measure only.** Recall@k always rises with k, so the rule can't
    judge it. Report k = 3/5/10 (Recall, nDCG, crowding, evidence tokens); the default stays 5.
  - **Threshold:** a curve t = 0.40…0.80 in 0.02 steps (on-corpus Recall@5/nDCG@5, on-corpus
    empty rate, off-corpus no-result rate, max off-corpus top-1 score). A lower t replaces 0.6
    only if it passes the rule on nDCG@5 **and** the off-corpus no-result rate doesn't drop.
    Raising t can only lose recall, so it is reported, not chosen. With ~5 off-corpus
    questions this is coarse; reports say so.
- **Golden set.** 50 entries, quotas fixed now: `factual` 10, `paraphrased` 10, `cross_paper` 8,
  `table` 8, `multi_section` 9, `off_corpus` 5. Coverage: every one of the 10 papers is
  labelled in ≥ 3 entries. Constraints checked by the schema test:
  - `cross_paper` labels span ≥ 2 papers; `multi_section` has ≥ 2 sections of one paper;
  - `off_corpus` has empty `relevant` and `source_chunks`, every other type has ≥ 1 of each;
  - a `paraphrased` question contains no manifest title and no model name from a test alias
    list (MAIRA-2, LLaVA-Med, CONCH, RadFM, Med-PaLM M, PathAlign, RoentGen, …);
  - labels never use the Phase 8 odd labels (`Input`, `Target`, `Caption:`, `Example input:`,
    `Visual input example…`, `maira_2` appendix labels).

  Entry: `{"id": "q001", "question", "type", "relevant": [{"paper_id", "section",
  "grade"}], "source_chunks": ["maira_2:12"]}`. Our own text only. U3 probe: the runner reports
  per entry the longest word n-gram shared with its source chunks' text (read from Qdrant).
  T8 rewrites any `paraphrased` entry with an overlap ≥ 4 words.
- **Where code lives.** Core logic in `app/evaluation/` so mypy and bandit cover it; thin CLIs in
  `scripts/`. No new dependency (`langsmith`, `numpy` already installed).
- **Files on disk.** Committed, no paper text in any of them:
  - `data/eval/golden.jsonl`, `data/eval/rewrites.jsonl`, `data/eval/CHANGELOG.md` (pooling log);
  - `data/eval/runs/<name>.json`: config, golden sha256 + git hash, chunk count, per-question
    ranked `(paper_id, section, chunk_index, chunk_type, score)`, n-gram overlap;
  - `data/eval/reports/*.md`: run summaries, comparisons, the threshold curve.

  Every report header states "golden set LLM-labelled (Claude), not human-verified" (R9).
- **LangSmith (ADR-0024).** Dataset `paperlens-retrieval-golden`; example id =
  `uuid5(ns, entry id)`; inputs `{id, question}`, reference outputs `{type, relevant}`.
  - The sync script hashes the file. If the dataset's example ids equal the file's and every
    example's metadata `golden_sha256` equals the hash, it does nothing (changed in Execute:
    `langsmith` 0.12.5 cannot update dataset metadata, so the hash lives on each example).
    Otherwise it creates/updates the examples, deletes the ones no longer in the file, and
    tags the version with the short git hash (`update_dataset_tag`). It also refuses a
    `golden.jsonl` that fails `validate_golden`.
  - It refuses to run if `golden.jsonl` has uncommitted changes.
  - The runner's `--langsmith` uses `aevaluate` with the same retrieval target and thin
    evaluator adapters over the same metric functions; the experiment metadata = run config.
    Experiments run at k = 5, t = 0.6; curves are local only.

## Task graph
G1: T1 ‖ T2 ‖ T3 ‖ T4 → G2: T5 ‖ T6 → G3: T7 ‖ T8 → G4: T9 → G5: T10 → G6: T11

T3 touches `tests/test_retrieval.py`; no other G1 task does. T8 is authoring (Claude), not code.

## Tasks

### [x] T1: Metric functions
- Files: `app/evaluation/__init__.py`, `app/evaluation/metrics.py`, `tests/test_eval_metrics.py`
- Depends on: — · Group: G1
- Acceptance criteria:
  - [x] `RankedHit` (frozen: paper_id, section, chunk_index, chunk_type, score); `cut(hits, k, threshold)`, `collapse_sections`, `recall_at_k`, `mrr_at_k`, `ndcg_at_k`, `is_empty`, `distinct_papers`, `table_hit`, as defined in Decisions
  - [x] Hand-computed fixtures: duplicate sections count once, a grade-1-only hit, IDCG cut at k, a relevant hit below the threshold, empty labels raise (off-corpus is excluded upstream) (R2)
- Note: done; 35 tests. Added: empty labels also raise in `table_hit`; grade outside {1, 2} raises.

### [x] T2: Golden-set schema and loader
- Files: `app/evaluation/golden.py`, `tests/test_eval_golden.py`
- Depends on: — · Group: G1
- Acceptance criteria:
  - [x] Pydantic `Label`, `GoldenEntry`, `QuestionType` (StrEnum); `QUOTAS`; `load_golden(path)` (JSONL, line numbers in errors); `golden_sha256(path)`
  - [x] `validate_golden(entries, manifests)` checks the rules in Decisions (unique ids, grades ∈ {1,2}, quotas, coverage, per-type constraints, paraphrased alias check, banned labels) and returns all violations
  - [x] Tested on small in-code fixtures, one per violation
- Note: done; 46 tests. `Label.grade` is a plain int so `validate_golden` reports bad grades with the rest; added `load_manifests`, `derive_aliases`; `maira_2` appendix banned-label rule is a heuristic, T8 must check it against real section names.

### [x] T3: `collection_name` and scored hits in retrieval
- Files: `app/retrieval/search.py`, `app/retrieval/ingestion.py`, `scripts/ingest.py`, `tests/test_retrieval.py`
- Depends on: — · Group: G1
- Acceptance criteria:
  - [x] `search_hits(vector, client, *, top_k, score_threshold, collection_name=COLLECTION_NAME) -> list[RankedHit]` (score + chunk_type kept); `search()` is built on it and returns exactly what it did before
  - [x] `ingest(..., collection_name=COLLECTION_NAME)` threads the name through prepare/stale/delete/upsert; `scripts/ingest.py --collection`
  - [x] Existing tests unchanged and green; new tests: a non-default collection leaves `papers` untouched (FakeQdrant); `search_hits` returns scores and chunk_type; `/query` contract unchanged (R7)
- Note: done; 46 tests in file. Deviation (accepted): `search()` is not built on `search_hits()` (RankedHit lacks title/authors/page); both share `_query_points`, parsing is duplicated. `search_hits` skips points without paper_id/section/chunk_index, missing chunk_type -> "text". `--collection` not run against real Qdrant.

### [x] T4: LangSmith dataset sync
- Files: `app/evaluation/langsmith_sync.py`, `scripts/sync_golden.py`, `tests/test_eval_sync.py`
- Depends on: T2 · Group: G1 (after T2's models exist; small)
- Acceptance criteria:
  - [x] No-op when the hash matches; on a change: create the dataset if missing, upsert by stable ids, delete removed examples, store the hash, tag with the git hash
  - [x] Refuses a dirty `golden.jsonl`; a missing `LANGSMITH_API_KEY` gives a clear exit 1
  - [x] Tested with a fake `Client`, no network (R6); the `langsmith` 0.12.5 method names are checked against the installed package
- Note: done; 19 tests, fake client. Deviations (accepted): hash stored on each example (0.12.5 cannot update dataset metadata); create/update split; sync refuses a golden file failing `validate_golden`. Never run against real LangSmith.

### [x] T5: Retrieval target, rewrite cache, pre-flight
- Files: `app/evaluation/target.py`, `tests/test_eval_target.py`
- Depends on: T1, T2, T3 · Group: G2
- Acceptance criteria:
  - [x] `RetrievalConfig` (embedder, collection, prefix, rewrite, rewrite_model, depth=20); `retrieve_ranked(entry, config, ...)` → query used + top-20 `RankedHit`s at t = 0 via `search_hits`
  - [x] Rewrite cache as in U6; a miss with no LLM, or any rewrite fallback, raises
  - [x] `preflight(...)` as in U2: returns the corpus index (sections, chunk count, chunk types) or raises one error listing all problems (R5)
  - [x] n-gram overlap helper; tests with a fake async Qdrant, a fake embedder and `FakeLlm` (no Ollama/Qdrant)
- Note: done; 48 tests, fakes only. `preflight(entries, client, config)` (signature chosen in Execute); extra pre-flight problems: empty collection, points missing paper_id/section/chunk_index/embedding_model; `RewriteCacheMissError` (ruff N818); imports the private `_PROMPT` from `app/workflow/rewrite.py` for the prompt hash (accepted). Never run against real Qdrant or Ollama.

### [x] T6: Bootstrap, comparison, reports
- Files: `app/evaluation/compare.py`, `app/evaluation/report.py`, `tests/test_eval_compare.py`
- Depends on: T1, T2 · Group: G2
- Acceptance criteria:
  - [x] Stratified paired bootstrap, per-type means, W/L/T, `verdict()` exactly as in Decisions; seeded, so results are deterministic
  - [x] Tests: A = B → range [0, 0] and all ties; B better on every question → lo > 0 and adopted; one type clearly worse → rejected despite lo > 0
  - [x] Markdown renderers: run summary (incl. k = 3/5/10 table, crowding, table hit rate, n-gram overlap per type), comparison, threshold curve; each states the config (R4) and the R9 sentence
- Note: done; 29 tests, in-code fixtures. Run-result model (`RunMeta`, `QuestionRun`, `RunRecord`, `save_run`/`load_run`) lives in `compare.py`. Added `threshold_verdict` and separate cut-offs for A and B; `verdict()` does not enforce nDCG@5 at 0.6 (reports state the cut-offs; accepted). `numpy` now declared in `pyproject.toml`. Evidence tokens need `QuestionRun.chunk_tokens`, filled by T7.

### [x] T7: Runner CLI (local + LangSmith)
- Files: `scripts/eval_retrieval.py`, `app/evaluation/langsmith_eval.py`, `tests/test_eval_runner.py`
- Depends on: T4, T5, T6 · Group: G3
- Acceptance criteria:
  - [x] Subcommands: `run` (pre-flight → retrieve → `runs/<name>.json` + report), `compare A B`, `curve RUN`, `sections` (per-paper section list with chunk ids, for authoring), `pool RUN...` (unlabelled sections in each run's top-10, for adjudication)
  - [x] `run --langsmith`: `aevaluate` with evaluator adapters over the T1 functions; a test with a mocked `aevaluate` shows the adapters give the same numbers as the local path and the metadata holds the config (R3)
  - [x] `run` fills `QuestionRun.chunk_tokens` (token estimate per hit, from the chunk text already read for the n-gram probe, no text stored) so the k = 3/5/10 report shows evidence tokens instead of `n/a`; also fills `ngram_overlap` and `query`
  - [x] Loads `.env` like `scripts/ingest.py`; runs as `python -m scripts.eval_retrieval`
- Note: done; 58 tests, fakes only. Logic is in `app/evaluation/runner.py` and `authoring.py` (script is a thin CLI), so mypy/bandit cover it. Imports the private `_estimated_tokens` from `app/workflow/generate.py` (pending owner sign-off). `sections` ran live against Qdrant (595 chunks); `run`, Ollama rewrites and `aevaluate` not yet run live.

### [x] T8: Golden set v0 (Claude: draft → verify)
- Files: `data/eval/golden.jsonl`, `tests/test_golden_set.py`
- Depends on: T2, T7 (`sections`) · Group: G3
- Acceptance criteria:
  - [x] Draft from chunk text dumped to the scratchpad (never committed). For each label, scan the paper's whole section list (all papers for cross-paper questions)
  - [x] Verify pass: reread each labelled section; fix or drop failures; record counts (drafted / fixed / dropped) in Note
  - [x] `tests/test_golden_set.py` runs `validate_golden` on the real file and passes (R1, R6); `run` pre-flight passes against `papers`
- Note: done; 50 entries, 142 labels (83 grade 2, 59 grade 1). Verify pass: 10 entries got extra labels, 0 rewritten, 0 dropped. Live pre-flight passes (595 chunks). Worst paraphrased overlap 3 words. The `maira_2` banned-label rule matches no real section name (too narrow); owner decision pending (see 3-execute-T8.md).

### [x] T9: Runs, pooling, freeze v1
- Files: `data/eval/rewrites.jsonl`, `data/eval/runs/*.json`, `data/eval/reports/*.md`, `data/eval/CHANGELOG.md`, `data/eval/golden.jsonl`
- Depends on: T7, T8 · Group: G4
- Acceptance criteria:
  - [x] Ingest bge-base into `eval_bge-base-en-v1.5` (`papers` untouched); generate and commit rewrites
  - [x] v0 runs `baseline`, `no_rewrite`, `prefix`, `bge_base`; the truncation-exactness check recorded
  - [x] Pooling over the union of all four runs' top-10 (not baseline only, so baseline isn't favoured); additions logged in `CHANGELOG.md`; v0 → v1 baseline Recall@5 change recorded (estimate of missing labels)
  - [x] v1 committed, synced to LangSmith (if a key is present, else noted), all four runs re-run on v1; `baseline` also as a LangSmith experiment with matching numbers
  - [x] Optional owner spot-check file of 10 entries offered
- Note: done in three agent parts (T9a v0 runs, T9b pooling, T9c v1 runs + LangSmith); reports 3-execute-T9a/T9b/T9c.md. Pooling read all 383 candidates: 8 labels added (142 -> 150), baseline nDCG@5 0.521 -> 0.525, estimate matched. Truncation check: 0 mismatches. Dataset synced (50 examples, tag 11318b7), second sync a no-op; first `aevaluate` experiment passed the equality check (one duplicate experiment exists). Findings for T10: 0.6 threshold inert (off-corpus no-result 0.00 in all runs); 6/50 rewrites start with 'Rewritten question:' and 13/50 are unchanged (app rewrite_node does not strip the prefix). The owner spot-check list is in 3-execute-T8.md.

### [ ] T10: Comparisons and results ADR
- Files: `data/eval/reports/*.md`, `docs/adr/0025-retrieval-defaults-after-eval.md`, possibly `app/models/query.py`, `ui/app.py`, `app/retrieval/embedding.py`, `.env.example`
- Depends on: T9 · Group: G5
- Acceptance criteria:
  - [ ] Comparisons vs baseline: rewrite, prefix, embedder (rule on nDCG@5); threshold curve (threshold rule); k = 3/5/10 table (measure only); U3 paraphrased vs factual
  - [ ] ADR-0025 records every outcome, including "kept", with the numbers and the R9 caveat
  - [ ] Allowed code changes: the threshold constant + UI slider, or the prefix setting (`query_prefix`, `EMBEDDING_QUERY_PREFIX`, default as decided). **If rewrite-off or bge-base wins → stop and re-plan with the owner** (graph/SSE or Compose/latency impact)
- Note:

### [ ] T11: Docs
- Files: `README.md`, `CLAUDE.md`, `docs/roadmap.md`, `docs/architecture.md`, ADR-0023/0024
- Depends on: all · Group: G6
- Acceptance criteria:
  - [ ] README: how to run the eval, the sync, and that the golden set is LLM-labelled (R9)
  - [ ] CLAUDE.md: `app/evaluation/`, the scripts, `--collection`, phase status; roadmap row 9 ✓ with ADRs and "Claude-authored" wording
  - [ ] ADR-0023/0024 updated with anything Execute changed
- Note:

## ADRs
- ADR-0023: golden-set methodology: types, quotas, authoring passes, labels, metrics, decision rule incl. the top_k/threshold handling; roadmap row 9 wording (written in Plan)
- ADR-0024: dataset versioning and eval runner: git source of truth, LangSmith sync, local reports, collection per embedder; extends ADR-0019 (written in Plan)
- ADR-0025: retrieval defaults after the Phase 9 eval: threshold, top_k, rewrite, embedder, prefix; replaces the Discuss file's separate prefix-split ADR (written in T10)
