# Roadmap: Phases 7–11

This document fixes the **order and goals** of the post-MVP phases. It is not a
decision record: each phase's ADRs are written just in time, during that phase's
own planning, and committed in the same PR as the code. Decisions that depend on
measurements (parser, embedder, reranker) are recorded *after* measuring, following
the pattern of [ADR-0015](adr/0015-default-model-qwen3-06b-no-thinking.md).

## Baseline (Phase 6 complete)

The full stack (FastAPI → LangGraph → Qdrant/Ollama → Streamlit, Docker Compose)
works end to end, but it is not yet a meaningful RAG system:

- **Abstract-only corpus.** `data/papers/*.json` holds 10 abstracts, indexed as one
  vector per paper. Retrieval is effectively "pick 1 of 10".
- **Weak embeddings for long text.** `all-MiniLM-L6-v2` is tuned for short inputs.
- **No reranking**, no keyword/hybrid search (added or measured in Phase 10).
- **Blocking UI.** `POST /query` returns only after generation finishes, and the
  UI shows a spinner for the whole duration.
- **Stub grounding check.** `validate_node` sets `grounded = True` whenever
  citations exist and the answer is non-empty.
- **No retrieval eval.** `scripts/eval_llm.py` produces a human-scored sheet over
  hardcoded (faked) retrieval. There are no retrieval metrics.
- **Small default model.** `qwen3:0.6b` is fast enough for query rewriting but too
  weak for multi-chunk synthesis with correct citations (to be quantified in Phase 11).
- **Open:** `pip-audit` CVE drift in transitive dependencies.

## Cross-phase decisions (fixed now)

These shape more than one phase, so they are settled up front.

- **Corpus.** Keep the same 10 papers but use their full arXiv PDFs. PDFs stay
  gitignored. A committed manifest (arXiv IDs + URLs) and a download script make
  the corpus reproducible. Public papers only, no PHI
  ([ADR-0003](adr/0003-avoid-patient-data-and-clinical-decision-support.md)).
- **Chunk identity model.** Every chunk carries `paper_id`, `section`, `page`, and
  `chunk_index`. Citations display paper + section + page. Evaluation labels are
  at **paper + section** level, so they survive re-chunking experiments.
- **LLM roles.** Ollama remains the default and only required provider
  ([ADR-0007](adr/0007-use-ollama-as-default-local-llm-provider.md)). Gemini via a
  Google AI Studio API key stays opt-in
  ([ADR-0016](adr/0016-optional-hosted-llm-provider.md)) and is used as the demo
  generator and the eval judge. (The plan to use it as the drafter of golden-set
  questions was dropped in Phase 9: Claude wrote and labelled the golden set, see
  [ADR-0023](adr/0023-golden-set-methodology.md).)
- **Observability and evaluation via LangSmith (opt-in).** Tracing is enabled
  with `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY` (optionally
  `LANGSMITH_PROJECT`); LangGraph nodes are traced without code changes.
  Evaluation runs as LangSmith experiments over versioned LangSmith datasets.
  Off by default: the app runs fully offline without it, evaluation requires an
  account, and the test suite never depends on it. The `langsmith` SDK is already
  an indirect dependency via langgraph.
- **Measure before improving.** Retrieval improvements (reranking, hybrid search)
  come only after the retrieval eval exists, so every change ships with a
  measured delta.

## Phases

| # | Phase | Goal | Expected ADR(s) |
|---|---|---|---|
| 7 ✓ | Streaming | Tokens and step status (rewriting, retrieving, generating) are visible in the UI as they happen. `POST /query` stays unchanged for tests and eval. Opt-in LangSmith tracing for per-node latency and inputs/outputs. | SSE `POST /query/stream` endpoint; `stream()` on the LLM provider interface; LangSmith for observability and evaluation |
| 8 ✓ | Full-text ingestion | Parse PDFs (Docling), chunk by document structure, upgrade the embedding model, and show section/page citations. | Docling as PDF parser (ADR-0020); chunking with Docling's HybridChunker and chunk/citation identity (ADR-0021); default embedder `bge-small-en-v1.5` (ADR-0022) |
| 9 ✓ | Retrieval evaluation | Golden set of 50 questions, **Claude-authored and LLM-labelled, not human-verified** (the owner had no time for the human filter; changed from the Gemini-drafted plan), stored in git and synced to a LangSmith dataset. Each retrieval configuration is a run (local files, optionally a LangSmith experiment). Recall@k, MRR and nDCG@k are custom functions over paper + section labels. Outcome: the BGE query prefix is adopted; threshold 0.6, `top_k` 5, rewrite on and `bge-small-en-v1.5` are kept. | Golden-set methodology (ADR-0023); dataset versioning and eval runner (ADR-0024); retrieval defaults after the eval (ADR-0025) |
| 10 ✓ | Rerank and hybrid search | Retrieve top 20, rerank with a cross-encoder, keep top 5. BM25 + dense fusion (RRF) and a smaller chunk size were measured as eval-only experiments. Each change was judged with the Phase 9 decision rule on a golden set grown to 65 questions (20 off-corpus) and 164 labels, still LLM-labelled and not human-verified. Outcome: the MiniLM reranker is adopted (nDCG@5 0.683 against 0.565 for dense search alone, on the final golden set); no refusal rule; hybrid search and 256-token chunks are not adopted (no safe gain on top of the reranker). Smaller chunks halve the evidence text and the reranker time, to be judged in Phase 11. | Reranker (ADR-0026); hybrid search and the chunk-size addendum (ADR-0028, Proposed) |
| 11 | Generation and safety eval | LangSmith LLM-as-judge evaluators (Gemini as judge model) scoring faithfulness, citation correctness, and relevance. Human ratings collected through LangSmith annotation queues, replacing the CSV sheet from [`scripts/eval_llm.py`](../scripts/eval_llm.py) as the calibration source; judge-human agreement reported as Cohen's κ. Precision/recall of the safety filter. Replace the stub `validate_node` based on findings. | Judge model, rubric and calibration method (amends ADR-0016); grounding validator design |

### Phase 9 findings that feed Phase 10

Input candidates only. These are findings from [ADR-0025](adr/0025-retrieval-defaults-after-eval.md)
and the Phase 9 reports, not decisions. The owner decides the scope of Phase 10 in its
Discuss stage. All numbers come from a golden set that is LLM-labelled (Claude), not
human-verified, with only 5 off-corpus questions.

- **The 0.6 score threshold is inert.** The curve is flat from 0.40 to 0.68 and no
  off-corpus question is refused (no-result rate 0.00). The highest off-corpus top-1
  score is 0.829, and the first threshold that removes an off-corpus question (0.78)
  empties 31 % of on-corpus ones. A better refusal signal than a score cut-off (a
  reranker score, a margin between top hits, an LLM check) is a candidate. More than 5
  off-corpus golden questions would be needed to judge it.
  Options, cheapest first (owner's recommendation from the Phase 9 review, not decided):
  1. Margin or relative rule: refuse when the top score is not clearly above the rest.
     Cheap, but tuning on 5 off-corpus questions is weak evidence.
  2. LLM relevance check after retrieval: one extra call, and a 0.6B model is a weak judge.
  3. Reranker score as the signal (Phase 10): a cross-encoder separates on-topic from
     off-topic better than embedding similarity. Recommended first choice.
  4. Grounding rule in the generation prompt ("the papers do not cover this"): cheap
     safety net, to be checked in Phase 11.
  5. Show low confidence in the UI instead of refusing.
  Why a cut-off fails: the whole corpus is medical AI, so off-corpus questions stay close in
  topic to some chunk. Embedding similarity measures topic closeness, not whether the answer is present.
- **Rewriter quality.** 6 of 50 cached rewrites start with the literal text
  `Rewritten question:` (the prompt ends with that phrase and `rewrite_node` does not
  strip it), and 13 of 50 equal the question. The 0.6B rewriter showed no measured
  gain (nDCG@5 +0.020 without it, range [-0.006, +0.046]). Fixing the leak needs a
  rewrite-cache rebuild and a re-run of the four evals, ideally before Phase 10 so a
  reranker is judged on clean rewrites.
- **Paraphrased versus factual gap.** Baseline nDCG@5 0.452 against 0.676 (0.224
  gap). The query prefix closes about a third of it.
- **`cross_paper` is the weakest type** in every run (baseline nDCG@5 0.292; the top 5
  holds 1.8 distinct papers on average).
- **`top_k`:** k = 10 lifts Recall from 0.509 to 0.687 at about twice the evidence
  tokens, above the 4000-token evidence budget of `generate_node`. Its effect on
  answers belongs to Phase 11.

### Phase 9 diagnosis follow-ups (decide later)

Source: [`3-execute-diagnosis.md`](phases/09-retrieval-evaluation/3-execute-diagnosis.md),
part 8. Same caveat: LLM-labelled golden set, 45 on-corpus questions. Not decisions.

What the checks showed: crowding is minor (a cap of 1 chunk per section gives about +0.03
nDCG@5, caps of 2 or 3 give nothing), a second reader found 0 answers in 24 unlabelled
"right paper, wrong section" hits, and Recall@20 (0.82) is far above Recall@5 (0.53). The
main loss is ranking. Pick these up in the stage shown:

| # | Follow-up | Pick up in | Trigger or condition |
|---|---|---|---|
| 1 | ~~Reranker (retrieve 20, rerank, keep 5), judged with the Phase 9 runs and bootstrap rule~~ Done in Phase 10: MiniLM adopted (ADR-0026), no refusal rule | Phase 10 Discuss | First candidate. Targets the 0.82 to 0.53 gap, and doubles as the off-corpus refusal signal (option 3 above) |
| 2 | ~~Chunk-size experiment~~ Done in Phase 10, not adopted (ADR-0028 addendum): token budget 256, `merge_peers` on and off, new collection `eval_chunk256`, bootstrap against baseline | Phase 10 Discuss (after the reranker is measured) | Only if a reranker leaves a large "right paper, wrong section" share. Decision rule applies because it may change a default |
| 3 | Cap of chunks per section in `retrieve_node` (cap 1 or 2) | Phase 11 | Our metrics cannot see its cost (a needed second chunk of a section). Adopt only if the generation eval shows no drop in answer quality |
| 4 | Biomedical embedder (MedCPT, PubMedBERT; see Open questions) | Phase 10 Discuss, after 1 and 2 | `bge_base` was not clearly better and wrong-paper hits are about 20 %, so low priority |
| 5 | Automatic pooling with an LLM judge (recurring job behind `LlmProvider`, human calibration on about 50 pairs) | Phase 11 | Only if a later second-reader pass finds many missing labels. The first pass found 0 of 24 |
| 6 | Larger second-reader sample, with a non-Claude reader or a human, and ranks 6 to 20 included | Phase 11 (with the judge calibration) | The first pass used Claude, so it is not independent. The calibration step needs a human anyway |
| 7 | ~~Rebuild the live `papers` collection~~ Done: the live `papers` collection holds 595 chunks and the eval preflight passes (checked in the Phase 9 verify) | - | - |
| 8 | Fix the rewriter `Rewritten question:` leak and re-run the four evals | Before Phase 10 Execute | See the rewriter bullet above |

### Exit criteria (every phase)

- All five quality gates pass (pytest, ruff, mypy, bandit, pip-audit).
- Tests use mock LLMs only. No live Ollama or Gemini in the suite.
- The README Docker Compose demo path still works.

## Open questions

Decided in each phase's own discussion, not now:

- Chunk size and overlap (Phase 8)
- `bge-small-en-v1.5` vs `bge-base-en-v1.5` (Phase 8; settled in Phase 9: bge-small kept, bge-base was not better and clearly worse on table questions, ADR-0025)
- **Domain-specific embedding model (Phase 8; not tested in Phase 9, still open).** The Phase 8
  default is a general model (`bge-*`). When evaluation starts, also test a model
  trained on biomedical text or literature and compare it on the same golden set
  instead of assuming it wins. Candidates to check (confirm they still exist and
  check their licences first):
  - `MedCPT`: trained on PubMed search logs. It is asymmetric, with separate query
    and article encoders, so `embed()` would need a query/document split.
  - PubMedBERT-based sentence embeddings (for example `NeuML/pubmedbert-base-embeddings`).
  Things the choice depends on: the input unit (abstracts versus full-text chunks,
  mind the 512-token limit), the vector dimension (`VECTOR_DIM` and the Qdrant
  collection change, so a re-ingest with `--overwrite` is needed), and CPU latency
  in the local-first setup. The corpus is only 10 papers, so differences may be
  within noise until it grows.
- Whether query rewriting helps retrieval (measured in Phase 9: no gain shown for the 0.6B rewriter on clean questions, so it stays on; messy queries and a larger rewriter are untested, ADR-0025)
- Reranker candidates: `ms-marco-MiniLM-L-6-v2`, `bge-reranker-base`, optionally a
  Jina reranker (non-commercial license) (Phase 10)
- Which Gemini model name is current at implementation time (`GEMINI_MODEL` is
  configurable)

## Out of scope / parked

- **JEPA / VL-JEPA.** A vision-language research architecture with no off-the-shelf
  text reranker. A cross-encoder already gives a single-score relevance judgment
  without token generation.
- **Langfuse.** Open-source, self-hostable alternative to LangSmith. Considered
  but not chosen: it adds Postgres, ClickHouse, Redis and its own server to the
  Compose stack. Revisit if offline evaluation becomes a requirement.
- **Kubernetes.** Parked per [ADR-0002](adr/0002-docker-before-k8.md).
- **Expanding the corpus beyond 10 papers.** Revisit after Phase 10.
