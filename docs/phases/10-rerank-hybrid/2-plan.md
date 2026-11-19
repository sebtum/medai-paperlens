# Phase 10: Rerank and hybrid search — Plan

**Status:** signed off by the owner on 2026-11-18. Next: Execute, one task block at a time. All numbers come from a golden set that is LLM-labelled (Claude), not human-verified.

## Context

Phase 9 showed that search usually finds the answer but ranks it too low: 82 % of labelled sections are in the top 20, only 53 % in the top 5. Also, no off-topic question ever gets refused. Phase 10 adds a reranker, a second model that re-scores the top 20 hits and keeps the best 5. It also tests whether that model's score can refuse off-topic questions, and tests hybrid (keyword plus vector) search as an eval-only experiment. The Discuss file (`1-discuss.md`) set the scope. This plan sets the order and the checks.

## Decisions taken in Plan

- **MedCPT stays in.** Its model card says "public domain". The card's note that it is not meant for diagnostic use matches our non-clinical scope.
  - Details: `ncbi/MedCPT-Cross-Encoder` ships only `pytorch_model.bin` (438 MB, no safetensors). torch 2.14 loads it safely. Usage on the card: `[query, article]` pairs, `max_length=512`. CrossEncoder loading is checked in the reranker task (provider task T3).
- **The refusal cut-off is chosen after you see the results.** This is your decision, and it changes Discuss decision 4, which wanted a fixed rule first. Execute stops at a review point with the curve. There you choose one of three: a hard cut-off, a "the papers may not cover this" label below a looser cut-off, or nothing (numbers in the ADR only).
  - Details: risk: choosing the cut-off on the same 20 questions it is judged on flatters it. The ADR states this and reports the cut-off with its uncertainty. The cosine threshold of 0.6 stays as a loose pre-filter either way (inert, and retiring it would change the meaning of the request field).
- **The stored rewrites are cleaned, not regenerated.** The eval keeps the 50 rewritten questions in a file so every run uses the same queries. The 6 leaked rewrites are cleaned in that file without calling Ollama, so the re-run measures only the fix.
  - Details: rejected: regenerating all 50 with Ollama (the other 44 might drift, mixing two effects). The cache key gets a cleanup version, so a later change to the cleanup invalidates the file. The 15 new questions get fresh rewrites from Ollama (`run` without `--no-llm`).
- **Reranker scores are squeezed to 0..1.** The wrapper applies a sigmoid, so all three models give comparable 0..1 scores, and `confidence` stays in 0..1 if it ever carries this score.
  - Details: `CrossEncoder.predict(..., activation_fn=torch.nn.Sigmoid())`. Each run records `score_kind` (`cosine` or `rerank`). By default `compare` cuts cosine runs at 0.6 and reranked runs at 0 (no cut).
- **Both the app and the eval rerank the same text.** One shared function builds what the reranker reads: title, section and chunk text.
  - Details: `rerank_passage(title, section, text) -> "{title}\n{section}\n{text}"` in `app/retrieval/rerank.py`. The app takes it from `Citation` (the excerpt is the full chunk text). The eval reads `title` and `text` from the Qdrant payload, extending `fetch_chunk_texts` in `app/evaluation/runner.py`.
- **All three rerankers also run at depth 40 and on the original question.** The labelling pass has to cover every run that is later compared, so these runs come before the pass, not after the winner is known. Only the winner's variants enter a comparison.
- **The rerank step always exists in the workflow.** With `RERANKER_MODEL` empty or `none` it only keeps the first 5 hits. The graph shape and the stream events then stay the same in every setup.
  - Details: retrieve always fetches 20 (or 40 if adopted). Cutting dense top 20 to 5 equals dense top 5 (the Phase 9 truncation check found 0 mismatches).

## Task graph

G1: T1 ‖ T3 → G2: T2 ‖ T5 ‖ T8 → G3: T4 → G4: T6 → G5: T7 (owner review) → G6: T9 ‖ T10 → G7: T11 → G8: T12

- Details: T4 waits for T2, because new golden entries change the golden hash and T2 must run on v1. T4 also comes before any reranker result is seen, so the new questions are written blind. T8 (app) touches no eval files, and T5 touches no app files.

## Tasks

### [x] T1: Fix the rewriter leak
The rewriter's output loses a leading "Rewritten question:", and the 6 stored rewrites that have it are cleaned.
- Files: `app/workflow/rewrite.py`, `app/evaluation/target.py`, `data/eval/rewrites.jsonl`, `data/eval/CHANGELOG.md`, `tests/test_workflow.py`, `tests/test_eval_target.py`
- Depends on: — · Group: G1
- Acceptance criteria:
  - [x] A pure `clean_rewrite(text)` removes a leading `Rewritten question:` (any case, repeated), surrounding quotes and whitespace. `rewrite_node` uses it before the fallback.
  - [x] `_PROMPT` is unchanged. The cache key hashes the prompt plus a `REWRITE_CLEANUP_VERSION`.
  - [x] `_RecordingRewriteLlm` counts an output that is empty *after* cleaning as a failure.
  - [x] `rewrites.jsonl` is rewritten once, with new keys and cleaned text. Exactly 6 rewrite texts change, which is logged in `CHANGELOG.md`. A test checks that no stored rewrite starts with the prefix.
- Note: Done 2026-11-18. Exactly 6 changed (q009, q015, q029, q031, q035, q042), no Ollama call. Report: `3-execute-T1.md`.

### [x] T2: Clean baseline re-run
Re-run the four Phase 9 setups with the clean rewrites and check that the Phase 9 outcomes still hold.
- Files: `data/eval/runs/{baseline,no_rewrite,prefix,bge_base}.json`, `data/eval/reports/*.md`
- Depends on: T1 · Group: G2
- Acceptance criteria:
  - [x] Four runs with `--no-llm`. Reports regenerated (adds the Precision and Recall-versus-ceiling columns).
  - [x] Comparisons `prefix`, `no_rewrite` and `bge_base` against `baseline` re-run. The result is noted against ADR-0025.
  - [x] **Stop and ask the owner** if an ADR-0025 outcome flips (the prefix stops winning, or no-rewrite now wins).
- Note: Done 2026-11-18. No ADR-0025 outcome flipped. nDCG@5: baseline 0.532, prefix 0.562 (+0.030, range +0.005 to +0.056, margin thinner than Phase 9), no_rewrite 0.545 (not better), bge_base 0.544 (not better). Report: `3-execute-T2.md`.

### [x] T3: Reranker provider
A small reranker interface plus a sentence-transformers version, built like the embedder, with no new dependency.
- Files: `app/retrieval/rerank.py`, `tests/test_rerank.py`, `tests/conftest.py`
- Depends on: — · Group: G1
- Acceptance criteria:
  - [x] `Reranker` Protocol: `async score(query, passages) -> list[float]` and `model_name`. `CrossEncoderReranker` runs in a thread (`asyncio.to_thread`) with sigmoid output.
  - [x] `get_reranker_model_name()` reads `RERANKER_MODEL` (empty or `none` = off). `get_reranker() -> Reranker | None` is an `lru_cache` singleton and the FastAPI dependency.
  - [x] `rerank_passage()` as in Decisions.
  - [x] `FakeReranker` in conftest. An autouse fixture ensures no test loads a real model.
  - [x] One manual check (not in CI) that all three model IDs load with `CrossEncoder` and score a pair. Exact IDs and revisions are recorded in Note.
  - Details: MiniLM may now be `cross-encoder/ms-marco-MiniLM-L6-v2` (renamed, old ID redirects). `max_length=512` for all three.
- Note: Done 2026-11-18. All three model IDs load and score (MiniLM `cross-encoder/ms-marco-MiniLM-L-6-v2` rev 233902d, `BAAI/bge-reranker-base` rev 2cfc18c, `ncbi/MedCPT-Cross-Encoder` rev 71caf65). Explicit sigmoid needed for MiniLM only. Report: `3-execute-T3.md`.

### [x] T4: Golden set v2: 15 near-miss questions
Claude writes 15 more off-topic questions that are medical AI but not covered by the 10 papers. They are written before any reranker result is seen.
- Files: `data/eval/golden.jsonl`, `app/evaluation/golden.py` (`QUOTAS`), `tests/test_golden_set.py`, `tests/test_eval_golden.py`, `data/eval/rewrites.jsonl`, `data/eval/CHANGELOG.md`
- Depends on: T2 · Group: G3
- Acceptance criteria:
  - [x] Entries `q051`–`q065`, type `off_corpus`. Quota 5 → 20, total 50 → 65. Tests updated.
  - [x] Each checked against the corpus: the dense top-10 sections (`pool --include-off-corpus`) are read and contain no answer. Dropped or rewritten questions are counted in Note.
  - [x] None is flagged by the safety filter. The 15 new rewrites come from Ollama.
  - [x] Logged in `CHANGELOG.md` as v2 (draft until T6 freezes it).
  - Details: example: "How does GPT-4 perform on dermatology images?" Mix of near misses: same modality but a different task, a model the corpus doesn't cover, a dataset the corpus doesn't use.
- Note: Done 2026-11-18. 15 questions q051-q065, 0 dropped, 0 rewritten; 0 of 65 flagged by the safety filter; 15 new rewrites from Ollama, first 50 lines of both files byte-identical. v2 is a draft (hash af6ce07a...), T6 freezes it. Report: `3-execute-T4.md`.

### [x] T5: Eval support for reranking
The eval runner can rerank, time the reranker, and report the "right paper, wrong section" share.
- Files: `app/evaluation/target.py`, `runner.py`, `compare.py`, `report.py`, `metrics.py`, `authoring.py`, `langsmith_eval.py`, `scripts/eval_retrieval.py`, `tests/test_eval_*.py`
- Depends on: T3 · Group: G2
- Acceptance criteria:
  - [x] `RetrievalConfig` gets `reranker: str | None`, `rerank_query: "rewritten" | "original"`, `prefilter_threshold = 0.6`. `retrieve_ranked` pre-filters at 0.6, reranks with `rerank_passage` text, sorts by score (ties keep dense order) and returns the time taken.
  - [x] Named configs `rerank_minilm`, `rerank_bge`, `rerank_medcpt` (built on `prefix`). CLI flags `--depth`, `--rerank-query` and `--reranker` (the name is required when used).
  - [x] Run meta holds `score_kind`, reranker model, load time and disk size. Each question holds `rerank_ms`, timed after one warm-up call. The report shows the median and 90th percentile.
  - [x] `compare` picks its default cut-off from `score_kind`. `curve --gate` sweeps refusal on the top-1 score only (0 to 1). Off-corpus refused and on-corpus lost are reported per t.
  - [x] New metric `wrong_section_share(hits, labels, k)`, shown in the run report.
  - [x] `pool` records rejected candidates in `data/eval/adjudicated.jsonl` and skips them in later passes.
  - [x] The LangSmith experiment threshold follows `score_kind`. Tests use fakes only. The `NAMED_CONFIGS` test is updated.
- Note: Done 2026-11-18. `--depth`, `--rerank-query` and `--reranker` require `--name` (read as the run name, so a variant never overwrites a config file). `fetch_chunk_payloads` lives in `target.py` (circular import otherwise). Report: `3-execute-T5.md`.

### [x] T6: Reranker runs, labelling pass, freeze v2
Run all reranker setups, label the hits they lift into the top 5 the same way as in Phase 9, and freeze golden set v2.
- Files: `data/eval/runs/*.json`, `reports/*.md`, `golden.jsonl`, `adjudicated.jsonl`, `CHANGELOG.md`
- Depends on: T4, T5 · Group: G4
- Acceptance criteria:
  - [x] 9 reranked runs: 3 models × (rewritten at depth 20, original at depth 20, rewritten at depth 40). The four dense configs are re-run on the v2 draft.
  - [x] Pool over the union of the reranked runs' top 5 (`--k 5 --threshold 0`). Every candidate is read. Additions and rejections are logged.
  - [x] v2 frozen. All runs re-run on v2 (`--no-llm`). Dataset re-synced to LangSmith if a key is present, else noted.
- Note: Done 2026-11-18. 13 runs on frozen v2 (65 questions, 159 labels, hash cc08395a...). 241 candidates read, 9 labels added, 232 rejected. Open for owner: q044 label blocked by a banned heading, q037 label is chunk-based, q064 judged off-corpus. Report: `3-execute-T6.md`.

### [x] T7: Comparisons and owner review → reranker ADR
Compare each reranker with today's default. Then Execute stops so the owner can see times and scores and decide.
- Files: `data/eval/reports/*.md`, `docs/adr/0026-reranker.md`, `docs/phases/10-rerank-hybrid/3-execute-T7.md`
- Depends on: T6 · Group: G5
- Acceptance criteria:
  - [x] Each reranker against `prefix` with the decision rule (nDCG@5). Original versus rewritten query and depth 40 versus 20 for the leading model. A latency table. `curve --gate` for the leading model.
  - [x] **Owner review (planned stop):** the owner sets the time limit, then picks the model (the rule winner within the limit), the depth and the query text. For refusal the owner picks a hard cut-off, a low-confidence label, or nothing.
  - [x] ADR-0026 records every outcome with numbers and the LLM-labelled caveat, including "nothing adopted" if no model passes.
- Note: Done 2026-11-19. Owner decisions: MiniLM, depth 20, rewritten query, no refusal option (time limit under 2 s). ADR-0026 Accepted. MiniLM is the app default (unset = MiniLM; empty or `none` = off). No startup warm-up (open, listed in the ADR). Report: `3-execute-T7.md`.

### [x] T8: Rerank step in the app
A new workflow step between retrieve and generate. The UI shows it, and the citations event comes after it.
- Files: `app/workflow/rerank.py`, `graph.py`, `state.py`, `retrieve.py`, `app/api/routes/query.py`, `ui/app.py`, `docker-compose.yml`, `.env.example`, `tests/test_workflow.py`, `test_query.py`, `test_query_stream.py`, `test_retrieval.py`, `test_model_status.py`
- Depends on: T3 · Group: G2 (its default model is set after T7)
- Acceptance criteria:
  - [x] `make_rerank_node(reranker, *, keep=5)`. Without a reranker it cuts to 5. On an error it logs, keeps the vector order cut to 5, and sets `rerank_fallback`.
  - [x] `retrieve_node` fetches `RETRIEVE_DEPTH` (20). `build_workflow(client, provider, llm, reranker)`. `WorkflowState` gets `rerank_top_score: float | None`. `_initial_state` updated.
  - [x] SSE `citations` is sent when `rerank` ends (it was `retrieve`). Debug gets `rerank_model` and `rerank_top_score`. `confidence` is unchanged (cosine).
  - [x] UI label "Reranking passages". `RERANKER_MODEL` added to the Compose `api` service and `.env.example`. The model goes into the shared Hugging Face cache volume.
  - [x] Tests: node unit tests (order, cut, fallback, off), stream order with 6 steps, and `get_reranker` overridden in every API test.
- Note: Done 2026-11-18. Beyond the plan: `WorkflowState.rerank_fallback` and a `rerank_fallback` debug field; `build_workflow` takes the reranker as a required argument (`None` = off). Default model stays off until T7. Model loads lazily on the first request (no warm-up yet; decide after T7). Docker path not checked (Verify). Report: `3-execute-T8.md`.

### [-] T9 (skipped): Refusal in the app (only if T7 picks a cut-off or a label)
With a hard cut-off, the app answers "No relevant papers found." below the chosen reranker score, and `confidence` becomes the reranker score. With the label option, the app still answers, and the UI adds "the papers may not cover this".
- Details, label option: `RERANK_LOW_SCORE` setting, a `low_confidence` debug flag, a UI note. No API contract change, so no new ADR; recorded in ADR-0026. The criteria below are for the hard cut-off.
- Files: `app/workflow/rerank.py`, `app/models/query.py`, `ui/app.py`, `README.md`, `docs/adr/0027-refusal-signal.md`, tests
- Depends on: T7, T8 · Group: G6
- Acceptance criteria:
  - [ ] `RERANK_MIN_SCORE` (server setting). The rerank node empties the citations when the top-1 score is below it, and generate's existing empty path answers.
  - [ ] `confidence` = reranker top-1. UI text and README API section updated. ADR-0027 amends ADR-0025 and the API contract.
  - [ ] Skipped with a one-line note if T7 picks nothing.
- Note: Skipped 2026-11-19. The owner chose no refusal option at the T7 review. No ADR-0027. Refusal moves to the Phase 11 grounding check.

### [ ] T10: Hybrid search, eval only
Score chunks by keywords as well, merge both lists, and test whether that plus the reranker beats the reranker alone.
- Files: `app/retrieval/ingestion.py`, `search.py`, `scripts/ingest.py`, `app/evaluation/target.py`, `runner.py`, tests, `docs/adr/0028-hybrid-search.md`
- Depends on: T7 · Group: G6
- Acceptance criteria:
  - [ ] Spike: BM25 computed by Qdrant itself (`models.Document(model="qdrant/bm25")`, sparse vector with IDF), with no new dependency. **If it fails, stop and ask the owner** about `fastembed`.
  - [ ] `ingest(..., hybrid=True)` writes collection `eval_hybrid` (dense plus sparse `bm25` over `embed_text`). `papers` is untouched.
  - [ ] `search_hits_hybrid`: dense prefetch (cosine ≥ 0.6) plus BM25 prefetch, fused by RRF (reciprocal rank fusion), limit 20.
  - [ ] Runs `hybrid` (measured only) and `hybrid_rerank` (adopted reranker). Pool their top 5 (rejected candidates skipped). Rule against the adopted reranker config.
  - [ ] ADR-0028 records adopted or not. **If adopted, stop and re-plan app integration** (re-ingest `papers`, search change, Compose).
- Note:

### [ ] T11: Chunk-size trigger
Check whether "right paper, wrong section" is still above 25 % of top-5 hits after the reranker.
- Files: `data/eval/reports/*.md`, a Note here
- Depends on: T7, T10 · Group: G7
- Acceptance criteria:
  - [ ] `wrong_section_share` at k = 5 for the adopted config is reported (it is 36 % today).
  - [ ] At or below 25 %: recorded, done. Above 25 %: **stop and detail the experiment with the owner** (token budget 256, `merge_peers` on and off, two collections, a pre-flight mode that skips the source-chunk checks because re-chunking changes chunk numbers).
- Note:

### [ ] T12: Docs
- Files: `README.md`, `CLAUDE.md`, `docs/roadmap.md`, `docs/architecture.md`, `docs/adr/0023-golden-set-methodology.md`, ADR index
- Depends on: all · Group: G8
- Acceptance criteria:
  - [ ] ADR-0023 amended: v2 near-miss questions, the top-5 labelling pass, `adjudicated.jsonl`, score kinds and the gate curve.
  - [ ] README: the rerank step, `RERANKER_MODEL`, first-start download size, the stream event order. CLAUDE.md layers and commands. Roadmap row 10 done.
- Note:

## ADRs
- ADR-0023 amendment: golden set v2 and the labelling pass (T12; the method is fixed by this plan)
- ADR-0026: reranker model, depth, query text, placement as its own step, measured gain and time (T7)
- ADR-0027: refusal signal and the meaning of `confidence` (T9, only if a cut-off is adopted)
- ADR-0028: hybrid search, adopted or not (T10). The chunk-size outcome goes here too, unless it changes a default.

## Verification
- Each task: its tests plus `ruff check` on the touched files.
- Phase end: `.\.venv\Scripts\pytest.exe tests/`, `ruff.exe check .`, `mypy.exe app/`, `bandit.exe -r app/`, `pip-audit.exe`.
- Eval reproducibility: re-running the adopted config with `--no-llm` gives identical hits.
- Live check: the app's top 5 for the 45 on-corpus questions match the adopted eval run (as in Phase 9 T10).
- `docker compose up --build`: the stream shows 6 steps with `rerank`, and `citations` comes after `rerank`. The research question is grounded, the personal-advice question is refused, and the pasta question gets no citations. The rerank time is measured once in the container against the owner's limit.
