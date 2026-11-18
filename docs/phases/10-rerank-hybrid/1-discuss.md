# Phase 10: Rerank and hybrid search — Discuss

**Status:** decisions taken with the owner on 2026-11-18, one by one. Signed off 2026-11-18. Plan changed decision 4: the refusal cut-off is chosen after the results are seen (see `2-plan.md`). Every number below comes from the Phase 9 golden set, which is LLM-labelled (written by Claude), not human-verified.

## Goal

Put the right sections into the top 5 more often. Phase 9 showed that search usually finds the answer (82 % of labelled sections are in the top 20) but ranks it too low (53 % in the top 5).

A reranker fixes the order. It is a second model that reads the question and each of the top 20 hits together, gives each pair a relevance score, and keeps the best 5. Its score may also tell us when the papers do not cover a question at all.

- Details: baseline numbers from `data/eval/runs/prefix.json`: Recall@20 0.82, Recall@5 0.53, nDCG@5 0.566. 36 % of top-5 chunks are "right paper, wrong section". The 0.6 score threshold refuses no off-corpus question (no-result rate 0.00). Sources: ADR-0025, `docs/phases/09-retrieval-evaluation/3-execute-diagnosis.md`.

## Decisions

### 1. Fix the rewriter leak first

The query rewriter sometimes leaves the text "Rewritten question:" in its output. This is the first task of the phase, so the reranker is judged against a clean baseline. The fix, the new baseline and the reranker comparison go into one PR.

- Details: 6 of 50 cached rewrites start with `Rewritten question:` (the `_PROMPT` in `app/workflow/rewrite.py` ends with that phrase and the node does not strip it). Steps: strip the prefix in `rewrite_node`, rebuild `data/eval/rewrites.jsonl`, re-run `baseline`, `no_rewrite`, `prefix`, `bge_base`, regenerate their reports (which also adds the Precision and Recall-versus-ceiling columns missing from the old reports). Rejected: a separate PR first (one more PR, and the baseline changes before Phase 10 starts); dropping the rewriter (changes a default, so it needs its own decision rule run and an ADR).

### 2. Measure three rerankers

We compare three models and adopt the winner by the Phase 9 decision rule. All three run through `sentence-transformers`, which is already installed, so no new dependency.

| Model | Size | Strength | Open point |
|---|---|---|---|
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | 22M | Very fast on CPU | Trained on web search, not medicine |
| `BAAI/bge-reranker-base` | 278M | Strong general model, MIT licence | Slowest of the three |
| `ncbi/MedCPT-Cross-Encoder` | about 110M | Trained on PubMed searches | Licence to check before use |

- Details: Jina rerankers are excluded (non-commercial licence). `bge-reranker-v2-m3` (568M) is likely too slow on CPU. A side measurement: the reranker scores against the original question versus the rewritten one (a cross-encoder reads both texts together, so the original wording may work better). The decision rule is the stratified paired bootstrap on nDCG@5 from ADR-0024/0025.

### 3. Rerank 20 hits, test 40 once

The app reranks the top 20 hits. The eval also runs the winning reranker on the top 40 once. If 40 clearly wins and fits the time limit, it becomes the default.

- Details: the eval runner already retrieves `DEFAULT_DEPTH = 20` per question, so depth 20 costs nothing new. Depth 40 needs one run with a larger depth. At depth 20 the ceiling is Recall 0.82: no reranker can find what is not in the pool.

### 4. Test the reranker score as the refusal signal, with more test questions

Today no off-corpus question gets refused, and 5 test questions are too few to set any cut-off. Claude writes about 15 more "near-miss" questions (medical AI, but not covered by the 10 papers). A reranker cut-off is adopted only if it refuses most of them without losing on-corpus answers.

- Details: example near-miss question: "How does GPT-4 perform on dermatology images?" This makes golden set v2 (about 65 questions), logged in `data/eval/CHANGELOG.md`, re-synced to LangSmith, still marked LLM-labelled. If a cut-off is adopted, the API's `confidence` field changes meaning from vector similarity to reranker score. That is an API contract change and needs an ADR. The cosine threshold of 0.6 is then retired or kept as a loose pre-filter (decided in Plan from the curve).

### 5. Hybrid search: eval only, after the reranker

Hybrid search scores each chunk twice, by keywords and by vector similarity, and merges both lists. We measure it in a separate eval collection, as "hybrid plus reranker" against "reranker alone". The app changes only if it passes the decision rule.

- Details: keyword scoring = BM25 as a sparse vector in Qdrant, merged by reciprocal rank fusion (RRF) through the Query API (`prefetch` + `fusion`). Needs a second vector per chunk, so an eval collection such as `eval_hybrid`; the live `papers` collection stays untouched unless adopted. Probably needs `fastembed` for the BM25 encoding (a new dependency, eval-only unless adopted). Hope: exact terms (model names, datasets, table numbers); table questions have the most wrong-paper hits (38 %).

### 6. The reranker gets its own workflow step

A new rerank step sits between retrieve and generate. The UI step list and LangSmith then show "rerank" with its own timing. Retrieve fetches 20 hits; rerank keeps 5.

- Details: new `rerank_node` in `app/workflow/`; `WorkflowState` and the SSE `step_start` / `step_end` node names change (shared interfaces). The UI must label the new step. Rejected: reranking inside `retrieve_node` (fewer interface changes, but rerank time is hidden).

### 7. Time limit: measure first, then set it

We measure the extra time each reranker adds on the owner's machine before fixing a limit. The owner sets the limit after seeing the numbers, before a model is adopted.

- Details: report median and 90th percentile per query over the golden questions, CPU only, model already loaded. Rough expectations for 20 hits: MiniLM about 0.2 s, MedCPT about 0.5 to 1 s, bge-reranker-base about 1 to 3 s. Also report model load time and download size (it affects Docker image or first start).

### 8. Chunk-size experiment only if a trigger fires

After the reranker is adopted, we measure "right paper, wrong section" again. If it is still above 25 % of top-5 hits, we run the smaller chunk-size experiment in Phase 10. The biomedical embedder stays on the roadmap for later.

- Details: today 36 %. Experiment as in roadmap follow-up 2: token budget 256, `merge_peers` on and off, collection `eval_chunk256`, bootstrap against the reranked baseline.

## Scope

### In

- Rewriter leak fix and a clean re-run of the four Phase 9 configs.
- Three rerankers measured; one adopted into the app as its own step.
- Golden set v2 with about 15 more off-corpus questions; reranker score tested as the refusal signal.
- Hybrid search as an eval-only experiment.
- Latency measurement per reranker.
- Chunk-size experiment only if the 25 % trigger fires.

### Out

- Biomedical embedder (MedCPT or PubMedBERT as embedder): roadmap, later.
- Cap of chunks per section: Phase 11.
- Generation quality, the real grounding check and the safety-filter gap ("should i stop"): Phase 11.
- A larger corpus: after Phase 10.

## Requirements

- **The reranker is fair to judge.** Phase 9 labels were checked only for hits that dense search put in its top 10. A reranker lifts hits from ranks 11 to 20 that nobody labelled. Before comparing, the new top 5 of each reranker gets the same labelling pass as in Phase 9.
  - Details: reuse the `pool` authoring helper; log every added label in `CHANGELOG.md`; bump the golden hash. Without this, the reranker is penalised for good unlabelled hits.
- **Every adopted change has a measured gain.** Same decision rule as Phase 9.
- **The app still works offline.** The reranker model is downloaded once and cached, like the embedder. Tests use a fake reranker, never a real model download.
- **Five quality gates and the Docker Compose demo pass,** as in every phase.

## Assumptions

- The CPU on the owner's machine is the deployment target for the latency numbers.
- `sentence-transformers` `CrossEncoder` loads all three candidates.
- Qdrant's sparse vectors and Query API fusion work with the pinned `qdrant/qdrant:latest` image and `qdrant-client>=1.14`.

## Unknowns

- **MedCPT licence.** Resolved in Plan by reading the model card. If unclear, drop MedCPT and keep two models.
- **Latency per model.** Resolved by measurement in Execute (decision 7). The owner then sets the limit.
- **BM25 encoding route.** `fastembed` versus Qdrant server-side BM25. Resolved by a short spike at the start of the hybrid task.
- **How the refusal cut-off is chosen.** Fixed rule (for example "refuse most near-miss questions, lose at most N on-corpus") set in Plan, before the curve is seen.

## Risks

- **Too few off-corpus questions even after v2.** 20 questions is still small. Mitigation: report the cut-off with its uncertainty, and keep "show low confidence" as a fallback instead of a hard refusal.
- **Same author for questions and labels.** Claude writes the new questions and labels the new hits. Mitigation: mark it clearly, as in Phase 9; the human spot-check list stays open.
- **Image size and first start.** A 278M reranker adds about 1 GB to the download. Mitigation: measured and reported in decision 7.
- **Scope creep from hybrid.** Mitigation: hybrid stays eval-only unless it passes the rule.

## ADRs to write

- **Reranker model and placement:** which model, depth 20 or 40, its own step, measured gain and time.
- **Refusal signal and the meaning of `confidence`:** only if a reranker cut-off is adopted (amends ADR-0025 and the API contract).
- **Hybrid search:** adopted or not, with the numbers.
- **Golden set v2:** amendment to ADR-0023 (near-miss off-corpus questions, the new labelling pass).
