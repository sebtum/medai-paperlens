# ADR-0023: Golden-set methodology for retrieval evaluation

## Status

Accepted (Phase 9 Plan).

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
