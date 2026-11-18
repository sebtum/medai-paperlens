# Phase 9: Retrieval evaluation — Discuss

## Goal
Make retrieval quality measurable. A golden set of 40–60 questions, written and verified by
Claude and labelled at paper + section level, is scored with Recall@k, MRR and nDCG@k. The
scoring runs offline or as LangSmith experiments. The phase records a baseline and uses cheap
sweeps to confirm or replace the provisional Phase 8 defaults: threshold 0.6, top_k 5,
rewrite on, bge-small, no query prefix.

Owner decisions:
- Scope = harness + cheap sweeps.
- Runner = offline + LangSmith over shared metric functions.
- Golden set source of truth = a git file synced to LangSmith.
- **Claude writes the golden set; the owner does not review it manually** (no time). This
  differs from the roadmap's "Gemini-drafted, human-filtered".

## Scope
### In
- **Golden set** `data/eval/golden.jsonl`, the source of truth in git. Each entry has `id`,
  `question`, `type`, `relevant: [{paper_id, section, grade}]` (grade 2 = core, 1 = partial)
  and `source_chunks` (`paper_id:chunk_index` references for audit, no text). Types:
  - single-paper factual;
  - paraphrased (no title, model name or source wording);
  - cross-paper (2+ papers);
  - table-targeted;
  - multi-section within one paper (answer spread over distant sections);
  - off-corpus, about 10 %, with empty `relevant`.

  Entries hold our own text only and no long verbatim excerpts.
- **Authoring by Claude** in the Execute session (no runtime LLM, no Gemini key). It works in
  three passes, with per-type quotas fixed before drafting:
  1. *Draft:* write each question from the parsed chunks. For each label, scan the whole
     paper's section list (all papers for cross-paper questions), not only the source chunk.
  2. *Verify:* a separate pass rereads each labelled section and confirms it supports the
     question. Entries that fail are fixed or dropped.
  3. *Pooling, after the baseline run:* adjudicate retrieved but unlabelled top-k sections and
     add the relevant ones, with a change log.

  An optional owner spot-check of about 10 entries is offered but not required.
- **Metrics module.** Pure functions Recall@k, MRR, nDCG@k (graded) over
  `(paper_id, section)`. The ranked chunk list is first collapsed to unique sections at their
  first rank, so one section repeated across the top-k counts once (owner decision). Recall@k
  is the share of labelled sections found in the top-k. Off-corpus questions are left out of
  Recall, MRR and nDCG and reported separately as the no-result rate at the threshold.
- **Retrieval-only target.** Optional rewrite, then embed, then `search()`, reusing
  `app/retrieval/search.py`, `make_rewrite_node` and the `EmbeddingProvider`. No generation.
- **Eval runner** `scripts/eval_retrieval.py`. Local mode writes a committed JSON + markdown
  report. LangSmith mode runs `evaluate()` with the same metrics as custom evaluators, and the
  config goes into the experiment metadata.
- **Sync script.** Pushes `golden.jsonl` to a LangSmith dataset only when its hash changed,
  and tags the version with the git hash. The dataset is never edited in the LangSmith UI.
- **Label validation.** A run fails if a labelled `(paper_id, section)` doesn't exist in the
  indexed corpus (stale labels).
- **Sweeps:**
  - baseline;
  - rewrite on/off;
  - top_k ∈ {3, 5, 10};
  - the threshold curve, computed post hoc from a single top-20 run at threshold 0;
  - bge-small vs bge-base (re-ingest);
  - bge query instruction prefix. This needs a query/document split of `EmbeddingProvider`,
    which is a shared interface change.
- **Measurements only:** per-paper crowding (distinct papers in the top-k) and table-chunk hit
  rate.
- **Decision rule** (owner decision). Each comparison of a tested setting B against the
  current default A reports three things:
  - the paired bootstrap range (95 %, 1,000 resamples) of B − A, with resampling done within
    question types so the type mix stays fixed;
  - the scores per question type;
  - a win/loss/tie count per question.

  A default changes only if the whole range is above zero **and** no question type gets
  clearly worse. The change is recorded in an ADR. The bootstrap covers chance in which
  questions were picked, not bias in the golden set. Reports state that the results hold for
  questions like these LLM-written ones.

### Out
- Fixing the odd Phase 8 labels (`med_palm_m` `Input`/`Target`, `llava_med`
  `Caption:`/`Example input:`, `maira_2` appendix labels). Golden labels never point at them
  (owner decision). A later fix will trip the stale-label check.
- Domain embedders (MedCPT, PubMedBERT): listed as future experiments.
- Chunk-size and overlap sweeps.
- Implementing a per-paper cap. The decision moves to Phase 10, using these numbers.
- Reranking and hybrid search (Phase 10). Generation, faithfulness and safety eval (Phase 11).
- Corpus expansion.
- Eval in CI. Tests cover the harness, not live numbers.

## Requirements
- R1 40–60 questions in git covering all six types. Every entry has an id, a type and graded
  labels; off-corpus entries have empty labels.
- R2 Metric functions are pure and unit-tested against hand-computed fixtures.
- R3 The eval runs fully offline with no LangSmith. With a key, the LangSmith run gives the
  same numbers from the same functions.
- R4 Each report records its config: embedder, prefix, rewrite, top_k, threshold, chunk count
  and the golden-set git hash. Reports are committed, because LangSmith's free tier keeps
  experiment traces for only about 14 days.
- R5 Unknown labels fail the run loudly instead of counting as misses.
- R6 The test suite needs no Gemini, Ollama, Qdrant, LangSmith or parsed cache. The sync
  script is tested with mocks. A schema test checks `golden.jsonl`: ids are unique, types are
  valid, grades ∈ {1, 2}, off-corpus entries have empty labels, and the type quotas are met.
- R7 The `/query` and `/query/stream` contracts are unchanged. App defaults change only
  through an ADR.
- R8 The five quality gates pass. No new heavy dependency: `langsmith` is already a dep.
- R9 Docs and reports call the golden set **LLM-labelled (Claude), not human-verified**, so
  the numbers aren't overstated.

## Assumptions
- A1 Phase 9 needs no Gemini; Gemini stays reserved for the Phase 11 judge. The LangSmith free
  Developer plan is enough (5k traces/month, about 500 per sweep).
- A2 Phase 8 section labels are stable enough to label against.
- A3 Rewrite runs need a local Ollama (`qwen3:0.6b`). Sweeps run on the host, not in CI.

## Unknowns
- U2 Label validation without the parsed cache in CI: a committed section index vs a
  pre-flight check against Qdrant payloads at run time. → Plan.
- U3 Drafting bias: questions drafted from a chunk share its wording, which inflates dense
  scores. Mitigation: authoring rules and the paraphrased type. Check by comparing the
  paraphrased subgroup with the factual one. → measured in Execute.
- U4 Swapping embedders without clobbering the dev collection: a collection name per config vs
  re-ingest with `--overwrite`. → Plan.
- U6 Rewrite nondeterminism: temperature 0 vs caching rewritten queries per run. → Plan.

## Risks
- A small set gives noisy deltas (handled by the decision rule), and tuning the threshold on the
  same set it is evaluated on overfits. Numbers are presented as indicative.
- The bootstrap assumes the golden set is a fair sample. A systematic skew (wording close to the
  chunks, too many easy types) passes undetected. The per-type view and the paraphrased type
  are the only probes for it.
- **No human in the loop.** Claude built the chunker and also labels against it, so any
  systematic labelling error goes unnoticed. Mitigations: the separate verify pass, the
  stale-label check, type quotas fixed before drafting, and pooling.
- **Incomplete labels.** A relevant section that is missing from the labels makes a correct hit
  count as a miss, so recall is understated. Mitigation: the whole-paper section scan in the
  draft pass and pooling after the baseline.
- Section-level labels are coarse. A hit on a long section (e.g. the `maira_2` appendix)
  doesn't guarantee that the chunk holds the answer. Grades soften this.
- LangSmith retention and vendor lock-in. Mitigated by committed local reports and the git
  source of truth.
- Redistribution: no paper excerpts in committed eval files.

## ADRs to write
- ADR-0023: golden-set methodology (types and quotas; Claude-authored draft → verify → pooling
  instead of Gemini draft + human filter; graded paper + section labels; metric definitions; decision rule).
  Includes a roadmap row 9 update.
- ADR-0024: dataset versioning and eval runner (git source of truth, LangSmith sync, local
  reports; extends ADR-0019's Phase 9 scope).
- ADR-0025: query/document embedding split + bge query prefix (amends ADR-0012/0022); written
  if the prefix experiment goes ahead.
- Default changes (threshold, top_k, rewrite, embedder) are recorded after measuring, as
  ADR-0022 amendments or one results ADR, following the ADR-0015 pattern.
