# T10: Comparisons and results ADR

**Status:** done. The bge query prefix is now the app default (for BGE models). Everything else stays as it was. ADR-0025 records all five outcomes. Not committed. Golden set is still LLM-labelled (Claude), not human-verified.

| Check | Result |
|---|---|
| Outcomes | prefix adopted; rewrite, embedder, threshold, top_k kept |
| Stop condition (rewrite-off or bge-base wins) | Not triggered. No graph, Compose or UI change |
| Code change | `app/retrieval/embedding.py`, `app/evaluation/runner.py`, `.env.example` |
| Tests | 14 new. `pytest` 442 passed (was 428) |
| Live check | 45 of 45 on-corpus questions: same top 5 as the stored `prefix` run |
| Gates | pytest, ruff, mypy, bandit all pass. `pip-audit` not run (no dependency change) |
| `.env` | Never read or printed |

Words used below:
- **nDCG@5:** a score from 0 to 1 for how well the labelled sections are ranked in the top 5. Higher is better.
- **Query prefix:** a fixed sentence put in front of the question text before it is turned into a vector. BGE models are trained to expect it on queries, not on documents.
- **Provider:** the app class that turns a query into a vector (`SentenceTransformerProvider`).

## 1. Outcome of each question

**What it is:** each setting judged by the plan's rule (nDCG@5, 45 questions, 95 % range, no type clearly worse). I applied the rule. I did not re-decide anything.

**Example:** prefix vs baseline: nDCG@5 0.525 to 0.566, range [+0.012, +0.074]. The lower end is above 0, so B replaces A.

**Result:**

| Setting | Result | Decision | Why |
|---|---|---|---|
| Query prefix | +0.041, range [+0.012, +0.074], 11 wins / 4 losses | Adopted | Lower end above 0, no type clearly worse |
| Rewrite | +0.020 for no_rewrite, range [-0.006, +0.046] | Kept (on) | Gain not shown. Clean questions, weak 0.6B rewriter |
| Embedder | bge_base +0.026, range [-0.035, +0.084], clearly worse on table | Kept bge-small | Lower end below 0, and the table flag |
| Threshold | Flat from 0.40 to 0.68, off-corpus no-result 0.00 | Kept 0.6 | Inert, no lower t passes the rule |
| top_k | k=3/5/10 measured | Kept 5 | Measure only (owner decision) |

**Not checked:** nothing beyond the T9c numbers. They were copied, not recomputed.

## 2. The code change

**What it is:** the app embedder can now put a prefix in front of queries. Documents are not touched.

**Example:** with no setting and `BAAI/bge-small-en-v1.5`, the question `What is a CNN?` is embedded as `Represent this sentence for searching relevant passages: What is a CNN?`.

**Result:**

| File | Change |
|---|---|
| `app/retrieval/embedding.py` | `BGE_QUERY_PREFIX` constant. `SentenceTransformerProvider(model, query_prefix="")`: `embed()` prepends it. New `get_query_prefix(model_name=None)`. `get_embedding_provider()` passes it in |
| `.env.example` | Commented `EMBEDDING_QUERY_PREFIX=` with the rule |
| `app/evaluation/runner.py` | Eval embedder built with an explicit empty prefix (part 3) |

Rule for `EMBEDDING_QUERY_PREFIX`:
- Set (even to an empty string): used as is. Empty switches the prefix off.
- Unset, model name starts with `BAAI/bge-`: the BGE sentence above.
- Unset, any other model: no prefix.

How queries and documents are kept apart: the provider's `embed()` is only called for queries (`retrieve_node`, the eval). Ingestion never uses the provider. It calls `model.encode()` on the chunk text directly. So I did not add a second method. The `EmbeddingProvider` Protocol is unchanged. A test checks that no ingested text contains the prefix.

**Not checked:** a real Docker `api` run. The image is the old one, and I did not rebuild it.

## 3. The eval-side fix against a double prefix

**What it is:** the eval's `prefix` config adds the prefix itself. If the eval embedder also took the new app default, `prefix` would get it twice and `baseline`, `no_rewrite` and `bge_base` would get it by accident.

**Example:** `runner._default_embedder("BAAI/bge-small-en-v1.5")` now builds the provider with `query_prefix=""`. Only `RetrievalConfig.prefix` adds text, exactly once. Run files still record `config.prefix`, unchanged.

**Result:** 6 new regression tests in `tests/test_eval_runner.py`:
- `baseline`, `no_rewrite`, `bge_base` embed the bare query (3 tests).
- `prefix` embeds the prefix exactly once.
- Setting `EMBEDDING_QUERY_PREFIX` in the environment does not reach the eval embedder.
- The eval's own frozen copy of the prefix equals the app constant (so a later change to one must be a conscious change to both).

I kept a frozen copy of the string in `runner.py` on purpose: it defines the experiment that is recorded in the run files.

**Not checked:** re-running the four evals after the change. The eval path is covered by tests only. The runs are in the committed files.

## 4. The live check

**What it is:** a one-off script (not committed): `C:\Users\sebas\AppData\Local\Temp\claude\8ef53501dc8e147d\15ab05fa-077f-41d8-af17-1d964322eac2\scratchpad\live_check.py`. It uses the stored rewritten query of each on-corpus golden question, embeds it with the app's default provider, and searches the real `papers` collection in Qdrant.

**Example:** for q001 the app returned the same five chunks, in the same order, as `data/eval/runs/prefix.json`.

**Result:**

| Check | Result |
|---|---|
| Default provider prefix (env var unset) | the BGE sentence |
| `search_hits` top 5 at t=0 vs stored `prefix` top 5 | 45 / 45 identical (ids and order) |
| App `search()` top 5 at t=0.6 vs stored top 5 (score at least 0.6) | 45 / 45 identical |
| Mismatches | none |
| `/query` in-process | `test_query.py`, `test_query_stream.py` pass (mock embedder, no live LLM) |

**Not checked:** `/query` with the real embedder and a live Ollama in one call. The two halves are covered: retrieval by the live check, the endpoint by the tests. The running compose `api` container still has the old code.

## 5. ADR-0025

**What it is:** `docs/adr/0025-retrieval-defaults-after-eval.md`, plus a row in `docs/adr/README.md` (Amends 0022, 0012).

**Result:** it says:
- Context: golden set LLM-labelled (Claude), not human-verified. 5 off-corpus questions. 8 to 9 per type. One labeller. Clean-question caveat. Weak rewriter caveat.
- A decision table for all five settings, with numbers and 95 % ranges.
- Prefix: how it works, the default rule, the per-type gain table, the 45 of 45 check.
- Rewrite kept. The `Rewritten question:` leak (6 of 50 rewrites; 13 of 50 unchanged) is a follow-up bug, not fixed.
- Embedder kept (table flag). Threshold kept and plainly called inert; off-corpus top-1 up to 0.829; the app answers off-corpus questions from unrelated chunks. The full curve and the k = 3/5/10 table.
- The paraphrased vs factual gap: 0.224 nDCG@5.
- Consequences: no re-ingest. The prefix applies from the next API start. The Docker image must be rebuilt to pick it up.
- A "Deliberately not decided" list and the follow-ups.

**Not checked:** I did not edit ADR-0022. Its line "No bge query instruction prefix yet" is now out of date. The index marks ADR-0025 as amending it.

## 6. Gates

| Gate | Result |
|---|---|
| `pytest tests/ -q` | 442 passed |
| `ruff check .` | All checks passed |
| `ruff format --check .` | 77 files already formatted |
| `mypy app/` | Success, no issues in 42 source files |
| `bandit -r app/` | 0 issues |
| `pip-audit` | Not run (no dependency change) |

Git status: modified `.env.example`, `app/evaluation/runner.py`, `app/retrieval/embedding.py`, `docs/adr/README.md`, `tests/test_eval_runner.py`, `tests/test_retrieval.py`; new `docs/adr/0025-retrieval-defaults-after-eval.md` and this file. Untouched: `DEFAULT_SCORE_THRESHOLD`, `top_k`, the rewrite node, the graph, `docker-compose.yml`, the UI, all run files and reports.

## 7. Open points for the owner

| # | What is decided | Options in practice | Recommendation |
|---|---|---|---|
| 1 | ADR-0022 still says "no bge prefix yet" | A: add one line "Amended by 0025". B: leave it, the index already links the two | A, in T11 with the other doc edits |
| 2 | The `Rewritten question:` leak in `rewrite_node` | A: fix now (small code change, then rebuild the cache and re-run all four evals). B: leave it as a follow-up | B for Phase 9. It is in the ADR. Fix it before Phase 10 so reranking is judged on clean rewrites |
| 3 | The app now answers off-corpus questions from unrelated chunks, and the threshold cannot stop it | A: accept for now. B: plan a better refusal signal (reranker score, margin, LLM check) as a later task | B. It is outside Phase 9. Phase 10 (rerank) may give a better signal |
| 4 | Users with a non-BGE model and their own query instruction | A: nothing to do. B: document it in README | A in code, B in T11 (one line) |
| 5 | The running Docker `api` has the old code | A: rebuild when you next run the stack. B: nothing | A |

## 8. What T11 needs

- README: how to run the eval and the sync; golden set LLM-labelled (Claude), not human-verified; the `EMBEDDING_QUERY_PREFIX` setting and the default rule.
- CLAUDE.md: add `EMBEDDING_QUERY_PREFIX` to the env-var list; update the `embedding.py` line (`query_prefix`, `get_query_prefix`); `app/evaluation/`, scripts, `--collection`; phase status; roadmap row 9 with ADRs 0023 to 0025 and the "Claude-authored" wording.
- `docs/architecture.md`: query embedding applies a prefix.
- ADR-0023 / 0024 updates from Execute; optional one line on ADR-0022.
- Commit this task's files (list in part 6). The run files and reports from T9c may still need committing.
- Update the T10 box and note in `2-plan.md`.

## Files

| File | State |
|---|---|
| `app/retrieval/embedding.py` | changed |
| `app/evaluation/runner.py` | changed |
| `.env.example` | changed |
| `tests/test_retrieval.py` | changed (8 new tests) |
| `tests/test_eval_runner.py` | changed (6 new tests) |
| `docs/adr/0025-retrieval-defaults-after-eval.md` | new |
| `docs/adr/README.md` | changed (index row) |
| `docs/phases/09-retrieval-evaluation/3-execute-T10.md` | new (this file) |
