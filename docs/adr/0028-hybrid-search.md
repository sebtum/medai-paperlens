# ADR-0028: Hybrid search (BM25 plus dense, merged by RRF) is not adopted

## Status

Proposed (2026-11-20, Phase 10 Execute, T10), pending owner review. Outcome of the decision
rule of ADR-0023 applied to the runner of ADR-0024. Follows ADR-0026 (the adopted reranker).
No application change follows from this ADR. An addendum records the chunk-size
experiment (Phase 10, T11), also not adopted.

## Context

Dense search finds sections by meaning. Keyword search (BM25) finds them by exact words:
model names, dataset names, table numbers. **Hybrid search** runs both and merges the two
ranked lists. The merge used here is reciprocal rank fusion (RRF): each list gives a chunk
`1 / (k + rank)` points and the points are added, so only positions count, never raw
scores. Both lists weigh the same, with Qdrant's default `k`.

The Phase 10 question (discuss decision 5): does hybrid search plus the adopted reranker
beat the adopted reranker alone? It was measured as an eval-only experiment. The live
`papers` collection was not touched.

How it was built (T10, spike report `3-execute-T10-spike.md`):

- Qdrant 1.18.2 computes BM25 itself: `models.Document(text=..., model="qdrant/bm25")` as a
  sparse vector named `bm25` with the IDF modifier and default options. No new dependency
  (`fastembed` was not needed).
- Collection `eval_hybrid` holds the same 595 chunks as `papers`, with a named dense vector
  `dense` (same embedder, bge-small) and the sparse vector `bm25`.
- Query: one Qdrant call with a dense prefetch (cosine at or above 0.6, the existing
  filter, which applies to the dense list only) and a BM25 prefetch, fused by RRF, limit 20.
  The BM25 query is the rewritten question without the dense instruction prefix.
- Two runs: `hybrid` (no reranker, measured only) and `hybrid_rerank` (the 20 merged hits
  go to the adopted MiniLM reranker, depth 20, rewritten question).

Labelling. Hybrid put 48 top-5 sections in front of the judge that no earlier pass had
seen. They were read and judged by text only: 2 were added as labels (grade 1, q022 and
q043), 46 were rejected with a reason (`data/eval/adjudicated.jsonl`). This changed the
golden set hash to `7f236dd41db4` (65 questions, 161 labels). The chunk-size pass of the
addendum added 3 more labels and moved the hash to `0c62886fd489` (164 labels). Every run
used below was re-run on it. See `data/eval/CHANGELOG.md`.

## Results

Metric: nDCG@5 on the 45 on-corpus questions. Golden set LLM-labelled (Claude), not
human-verified, so the same author wrote the questions and the labels. Code at `0d961f8`.
Numbers are on golden hash `0c62886fd489` (re-run after the chunk-size labelling pass; the
hybrid conclusions did not change).

| Run | nDCG@5 | Recall@5 | Pool recall at 20 | Wrong-section share at 5 | Table hit rate |
|---|---|---|---|---|---|
| `prefix` (dense, ADR-0025) | 0.565 | 0.518 | 0.824 | 0.347 | 0.625 |
| `hybrid` | 0.697 | 0.643 | 0.864 | 0.302 | 0.875 |
| `rerank_minilm` (adopted, ADR-0026) | 0.683 | 0.628 | 0.824 | 0.284 | 0.625 |
| `hybrid_rerank` | 0.706 | 0.674 | 0.864 | 0.311 | 0.625 |

Pool recall at 20 is the share of labelled sections inside the 20 hits the reranker sees.
The pool of `rerank_minilm` is the dense pool, so its value equals `prefix`.

| Comparison (B against A) | Mean delta | 95 % range | Verdict |
|---|---|---|---|
| `hybrid_rerank` against `rerank_minilm` (the deciding one) | +0.023 | -0.0237 to +0.0840 | A stays |
| `hybrid` against `prefix` (measured only) | +0.131 | +0.0690 to +0.2003 | B replaces A |
| `rerank_minilm` against `prefix` (for scale) | +0.118 | +0.0447 to +0.1974 | B replaces A |
| `hybrid` against `rerank_minilm` (not in the plan, for information) | +0.014 | -0.0343 to +0.0639 | A stays |

For the deciding comparison: 7 questions better, 8 worse, 30 tied. No question type is
clearly worse. Off-corpus no-result rate is 0.00 for all runs, as before.

## Decision

1. **Hybrid search is not adopted.** On top of the adopted reranker its gain is +0.023 and
   the lower end of the range is below 0, so the rule keeps the reranker alone. The live
   app, the `papers` collection and Compose stay as they are.
2. **The eval support stays in the code**: `ingest --hybrid --collection eval_hybrid`, the
   `hybrid` and `hybrid_rerank` configs, `search_hits_hybrid`. It is eval only and refuses
   to write the app collection.

## Consequences and what we learned

- **Hybrid roughly equals the reranker here, with a different cost.** Alone, hybrid scores
  about the same as the reranker (+0.014, not a safe difference). BM25 adds a few
  milliseconds in the vector database, while the reranker adds about 1.7 s per question.
  The reranker only re-orders a pool, the hybrid query changes the pool. The pool grew
  only a little (0.824 to 0.864 at 20), which fits the small gain on top of the reranker.
- **Hybrid alone is a possible cheaper alternative, not a result of this ADR.** The plan
  asked a different question (on top of the reranker), so this is not adopted. If the owner
  wants it, it needs its own comparison and ADR, a larger question set, and the app work
  listed below.
- **Table questions are better with hybrid alone** (table hit rate 0.875 against 0.625).
  The expected weakness of BM25 on table numbers did not show up. Only 8 table questions.
- **If hybrid is adopted later**: re-ingest `papers` into the hybrid layout (named dense
  vector plus `bm25`), change the app search, and pin the Qdrant image version in Compose
  (it is `latest` today). Weights and `k` were not tuned: with 65 questions tuning would
  fit noise.

## Addendum: chunk size (T11, 2026-11-20)

**Smaller chunks were tried and not adopted.** They gave about the same ranking quality
with half the text per answer and half the reranker time, but the gain is not safe, so the
live collection stays as it is.

Trigger. After the reranker, 28.4 % of the top-5 chunks come from the right paper but a
wrong section. The limit set in the discuss stage was 25 %. The owner approved the full
experiment.

Experiment. The papers were cut again with a cap of 256 tokens (the whole text that is
embedded, title line included; today 512), in two ways, into eval-only collections:
`eval_chunk256_merge` (neighbouring pieces under one heading glued together up to the cap,
1,258 chunks) and `eval_chunk256_nomerge` (pieces kept apart, 1,961 chunks). Today the
collection `papers` has 595 chunks. The pipeline is the adopted one (dense search, MiniLM,
depth 20, rewritten question). `papers` was not touched.

| Run | Chunks | nDCG@5 | Recall@5 | Wrong-section share | Table hit rate | Evidence tokens at 5 | Reranker time, median |
|---|---|---|---|---|---|---|---|
| `rerank_minilm` (adopted) | 595 | 0.683 | 0.628 | 0.284 | 0.625 | 2556 | 1592 ms |
| `chunk256_merge` | 1,258 | 0.703 | 0.639 | 0.258 | 0.750 | 1282 | 795 ms |
| `chunk256_nomerge` | 1,961 | 0.697 | 0.632 | 0.258 | 0.875 | 1105 | 780 ms |

| Comparison against `rerank_minilm` | Mean delta | 95 % range | Wins / losses / ties | Verdict |
|---|---|---|---|---|
| `chunk256_merge` | +0.020 | -0.0323 to +0.0840 | 12 / 13 / 20 | A stays |
| `chunk256_nomerge` | +0.014 | -0.0402 to +0.0772 | 12 / 15 / 18 | A stays; multi_section clearly worse (-0.052, 2 wins, 5 losses) |

Labelling. The two runs put 20 sections into the top 5 that no earlier pass had seen. They
were read and judged by text only: 3 were added as labels (grade 1: q023 twice, q038), 17
were rejected with a reason. See `data/eval/CHANGELOG.md`.

Decision.

1. **Smaller chunks are not adopted.** Neither variant has a safe gain, and the no-merge
   variant is clearly worse on multi-section questions. `papers` and the app stay as they
   are.
2. **The wrong-section share fell from 28.4 % to 25.8 %** and is still above the 25 % limit.
   Cutting smaller does not remove the problem; it moves it a little.
3. **The eval support stays in the code**: `ingest --chunk-tokens N --no-merge-peers` and
   `run --rechunked` (skips the source-chunk-number checks, since re-chunking changes
   the numbers; the labelled-section check stays). Both are eval only and refuse `papers`.

What we learned.

- **The same score with half the text.** Evidence tokens drop from 2556 to 1282 and the
  reranker time from 1592 ms to 795 ms, with no loss in nDCG@5 (+0.020, not safe). The
  decision rule asks for a gain, so it keeps today's chunks. A cheaper answer step is a
  different argument, and it can only be judged with the generation evaluation (Phase 11).
- **Adoption is a bigger change than the test.** The live collection would be rebuilt, chunk
  numbers and citation numbers would change, and the answer model would read shorter
  pieces. That needs its own plan.
- **If Phase 11 shows shorter context helps** (faster, or no worse answers), re-test the
  merge variant on a larger question set. The no-merge variant is not worth it.
- **Idea for later (owner, 2026-11-21): smaller chunks to cut time or tokens.** With top 5
  the evidence is about half as long. With top 10 it is about as long as today (2 x 1282
  tokens), so the token saving is gone but the answer sees more sections. The reranker time
  stays about half as long only while the depth stays 20; if the depth must double to keep
  the coverage, that saving is gone too. Untested at top 10 and at depth 40.

Limits of this addendum. 45 on-corpus questions, per type 8 to 10. Labels come from the
same author as the questions. New labels were added only from the top 5 of the two
chunk-size runs, 3 of 20 candidates, so a small bias towards them is possible. The effect on
generated answers was not measured. `source_chunks` entries refer to today's 595 chunks and
were not checked on the re-cut collections.

## Limits

- Golden set LLM-labelled by the same author as the questions; 45 on-corpus questions, per
  type 8 to 10, so per-type rows are rough.
- Labels for hybrid were added from the top 5 of the two hybrid runs only; sections that
  appear only in lower ranks stay unlabelled. Two labels were added from 48 candidates,
  so this bias is small, and it can only favour the hybrid runs.
- The BM25 query cost in the app was not measured. The numbers above are quality only.
- The runs `rerank_bge*` and `rerank_medcpt*` from ADR-0026 stay on the earlier hash
  `cc08395a87ca`. They were not re-run on the new hash because those models are not used.
