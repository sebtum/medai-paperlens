# ADR-0023: Golden-set methodology for retrieval evaluation

## Status

Accepted (Phase 9 Plan). Amended in Phase 10 (golden set v2, see the amendment below).

## Context

Phase 8 left every retrieval default provisional: threshold 0.6, top_k 5, query rewrite
on, `bge-small-en-v1.5`, no query prefix (ADR-0022). There are no retrieval metrics, so
none of these can be confirmed or replaced, and Phase 10 (rerank, hybrid) needs a measured
baseline. The roadmap planned a Gemini-drafted, human-filtered golden set. The owner has no
time to review labels, so Claude writes and checks the set in the Execute session, with no
runtime LLM.

## Decision

**Golden set.** `data/eval/golden.jsonl`, 50 entries. Each entry has `id`, `question`,
`type`, `relevant: [{paper_id, section, grade}]` and `source_chunks`
(`paper_id:chunk_index`, for audit only). It holds our own text only and no paper excerpts.
Grade 2 = the section holds the core of the answer, grade 1 = partial or supporting.

**Types and quotas, fixed before drafting:**

| Type | Count | Constraint |
|---|---|---|
| `factual` | 10 | single paper |
| `paraphrased` | 10 | no paper title or model name, no source wording |
| `cross_paper` | 8 | labels span ≥ 2 papers |
| `table` | 8 | ≥ 1 source chunk is a table chunk |
| `multi_section` | 9 | ≥ 2 distant sections of one paper |
| `off_corpus` | 5 | empty `relevant` and `source_chunks` |

Every paper is labelled in ≥ 3 entries. Labels never use the odd Phase 8 labels
(`med_palm_m` `Input`/`Target`, `llava_med` `Caption:`/`Example input:`/`Visual input
example…`, `maira_2` appendix labels).

**Authoring in three passes:**
1. *Draft* from the parsed chunks. For each label, Claude scans the whole paper's section
   list (all papers for `cross_paper`), not only the source chunk.
2. *Verify:* a separate pass rereads each labelled section and confirms it supports the
   question. Entries that fail are fixed or dropped.
3. *Pooling* after the first runs. Unlabelled sections in the top 10 of **all** sweep runs
   (not the baseline only, so the baseline isn't favoured) are adjudicated, and the
   relevant ones are added with a change log (`data/eval/CHANGELOG.md`). The v0 → v1
   recall change is recorded as an estimate of how many labels the draft missed.

As a check on drafting bias (U3), the runner reports for each entry the longest run of words
it shares with its source chunks. `paraphrased` entries with a shared run of 4 or more words
are rewritten.

**Metrics** (pure functions over `(paper_id, section)`). The ranked list is the hits with
score ≥ threshold, cut to k chunks, then collapsed to unique sections at their first rank.
- Recall@k = share of labelled sections in that list.
- MRR@k = 1 / rank of the first labelled section, else 0.
- nDCG@k: gain = grade, discount log2(rank + 1), ideal DCG from the labels sorted by grade
  and cut to k.

Off-corpus entries are left out of these and reported as the no-result rate at the
threshold. Crowding (distinct papers in the top k) and the table-chunk hit rate are
reported as measurements only.

**Decision rule.** The primary metric is nDCG@5 at threshold 0.6. A tested setting B is
compared with the default A by a paired bootstrap of mean(B − A): 1,000 resamples, seeded,
resampled within each question type so the type mix stays fixed, 95 % percentile range.
Alongside it come per-type means and a per-question win/loss/tie count. B replaces A only
if the whole range is above zero **and** no type is clearly worse (mean delta ≤ −0.05 and
more losses than wins in that type). Any change is recorded in an ADR.
- **top_k** is measured only (k = 3, 5, 10). Recall@k rises with k by construction, so the
  rule can't judge it; the default stays 5 until generation-side evidence exists (Phase 11).
- **Threshold:** a curve from 0.40 to 0.80. A lower value replaces 0.6 only if it passes
  the rule on nDCG@5 and doesn't lower the off-corpus no-result rate. A higher value can
  only lose recall, so it is reported, not chosen.

All reports state that the golden set is **LLM-labelled (Claude), not human-verified**, and
that results hold for questions like these.

## Changes during Execute

The decision above stands. These details were settled while building and running it
(Phase 9 Execute, 2026-11-17).

- **Golden set as built.** v0 has 50 entries and 142 labels (83 at grade 2, 59 at
  grade 1), with the planned quotas. The verify pass gave 10 entries extra labels and
  rewrote or dropped none. The worst paraphrased overlap with a source chunk was 3 words
  (limit 4), so no entry was rewritten.
- **Pooling ran over all 383 candidates.** The union of the top 10 of all four v0 runs
  (`baseline`, `no_rewrite`, `prefix`, `bge_base`) at threshold 0.40 gave 383
  unlabelled sections. All were read: 8 labels were added (1 at grade 2, 7 at grade 1),
  375 were rejected. No question text, id, type or `source_chunks` entry changed, and no
  label was removed or re-graded. The off-corpus questions were checked separately and
  nothing answers them.
- **v0 to v1.** The result is v1: 150 labels (84 at grade 2, 66 at grade 1), sha256
  prefix `4a4733e5fa76`. Baseline Recall@5 went from 0.510 to 0.509 and nDCG@5 from 0.521
  to 0.525, matching the estimate made by rescoring the old hits. So the draft missed few
  labels, at least among sections that these four setups retrieve; sections that none of
  them retrieves cannot be found by pooling. The method and every addition are in
  `data/eval/CHANGELOG.md`.
- **Banned-label rule.** The first `maira_2` rule (appendix-style headings) matched no
  real section name, so it banned nothing. The Phase 9 verify found this (an independent
  Codex review added `Additional qualitative examples > Current Study` to the real set
  and the validator accepted it). Fixed: `validate_golden` now also bans an explicit list
  of 13 real `maira_2` false headings (the example-figure labels under `Additional
  qualitative examples` and the annotation-protocol sections under `Qualitative
  evaluation of twenty random ...`, from the T8 notes), matched on the whole section
  label. No existing golden label is on the list, so `golden.jsonl` and its hash did not
  change. In the pool those sections were 1.6 % of the top-5 slots (14 of 900) and none
  answers a question.
- **Factual means one paper.** The `factual` constraint "single paper" was in this ADR
  but not in the validator. The Phase 9 verify added it; every real factual entry
  already satisfied it.
- **Report wording.** Every report states "golden set LLM-labelled (Claude), not
  human-verified". The optional owner spot-check list (10 entries) is in
  `docs/phases/09-retrieval-evaluation/3-execute-T8.md`.
- **Outcome.** The decision rule was applied in Phase 9 and its results are recorded in
  ADR-0025 (the BGE query prefix was adopted; threshold, `top_k`, rewrite and embedder
  were kept).

## Amendment: golden set v2 (Phase 10, 2026-11-18 to 2026-11-21)

The decision above stands. Phase 10 added a reranker and two experiments (hybrid search,
chunk size), and these details were settled while measuring them. The numbers are in
ADR-0026 and ADR-0028, the full record is in `data/eval/CHANGELOG.md`.

- **Near-miss off-corpus questions.** v2 adds 15 off-corpus questions (q051 to q065), so
  `off_corpus` goes from 5 to 20 and the set from 50 to 65 questions. They are medical-AI
  questions that look like they belong to the 10 papers but are not answered by them (same
  modality but another task, a model or dataset the corpus does not cover, a topic next to a
  covered paper). They were written before any reranker result existed and checked against the
  corpus text. Reason: with 5 off-corpus questions a refusal rule could not be judged, and a
  cosine threshold was already known to be inert. `QUOTAS` and the two golden tests changed.
  q001 to q050 were not touched.
- **Labelling pass on the top 5 of every new run.** An unlabelled section counts as
  irrelevant, so a run that retrieves a new section can look worse for a reason that is a
  missing label. After each new group of runs, the unlabelled sections in their top 5 (not
  the top 10 of Phase 9; reranked and fused scores are not cut at a threshold) are read
  against the question text only, never by score or rank, and judged by the Phase 9
  standard. Three passes ran: the reranker runs (241 candidates, 9 added), the hybrid runs
  (48 candidates, 2 added) and the chunk-size runs (20 candidates, 3 added). The set is now
  164 labels (84 at grade 2, 80 at grade 1). No label was removed or re-graded. The same
  author wrote the questions and judged the sections, so this is not a blind pass.
- **`data/eval/adjudicated.jsonl`.** Every section that was read and rejected is stored with
  its question and a one-line reason, so later passes skip it and the judgement can be
  audited. It holds our own text only.
- **Hash changes and re-runs.** Each pass changes the golden hash (`cc08395a87ca`, then
  `7f236dd41db4`, now `0c62886fd489`). Run records carry the hash they were scored on, and
  numbers are compared only within one hash. After a pass, the runs that are still used are
  re-run on the new file. The bge-reranker and MedCPT runs stay on `cc08395a87ca` (owner
  decision: those models are not used), and the numbers quoted in ADR-0025 and ADR-0026 come
  from earlier hashes.
- **Score kinds.** A run records its `score_kind`: `cosine` (dense search), `rerank`
  (reranker, 0 to 1 by sigmoid) or `rrf` (hybrid fusion). Only cosine runs are cut at the
  0.6 threshold; the other two are not cosine values and are not cut. Runs saved before
  Phase 10 count as cosine.
- **Gate curve.** For a refusal rule on the reranker's top-1 score, a curve shows, per
  cut-off t, how many of the 20 off-corpus questions are refused (the gain) and how many of
  the 45 on-corpus questions would be refused too (the cost). It is measured only; no rule
  was adopted (ADR-0026).
- **Decision rule unchanged.** nDCG@5, paired bootstrap, stratified by type, no type clearly
  worse. Phase 10 applied it to the reranker (adopted), hybrid search and chunk size (neither
  adopted).

## Alternatives Considered

### Gemini-drafted, human-filtered (roadmap plan)

Pros:
- A second model plus a human catches labelling errors that one author can't.

Cons:
- Needs owner review time that isn't available, and a Gemini key that Phase 9 otherwise
  doesn't need (A1).

### Chunk-level labels

Pros:
- Exact: a hit means the chunk holds the answer.

Cons:
- Labels break on any re-chunking, and several chunks of one section are often equally
  relevant. Section labels survive Phase 10 changes; grades soften their coarseness.

### Recall@5 as the primary metric

Pros:
- Easier to read.

Cons:
- Ignores grades and rank order within the top 5 (owner chose nDCG@5).

## Consequences

Positive:
- Every Phase 9 and Phase 10 default change comes with a measured, reproducible delta.
- Per-type scores and the paraphrased type make drafting bias visible.

Negative:
- No human in the loop: Claude built the chunker and labels against it, so a systematic
  labelling error passes undetected. Mitigated by the verify pass, quotas, pooling and the
  stale-label check (ADR-0024), not removed.
- 50 questions give noisy deltas, and tuning the threshold on the same set it is scored on
  overfits. Numbers are indicative.
- Section labels are coarse: a hit on a long section doesn't guarantee that the chunk holds
  the answer.

## Follow-up

- An optional owner spot-check of about 10 entries.
- Grow the set when the corpus grows; add domain embedders (MedCPT, PubMedBERT) as
  experiments on the same set.
