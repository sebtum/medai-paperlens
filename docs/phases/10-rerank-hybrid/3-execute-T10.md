# Phase 10, T10: Hybrid search (eval only) - result

**Result in one line.** Keyword search added to vector search helps a lot on its own, but on top of the reranker the gain is small and not safe, so hybrid is not adopted. The app is unchanged.

## What was tested

- **Hybrid search against the reranker alone.** Vector and keyword (BM25) lists are merged by reciprocal rank fusion (RRF), then the MiniLM reranker keeps 5. This was compared with the adopted reranker setup.
  - Details: collection `eval_hybrid` (595 chunks, dense plus sparse `bm25` built by Qdrant 1.18.2, no new dependency), runs `hybrid` and `hybrid_rerank`, code in commit `0d961f8`, decision in ADR-0028 (Proposed). The spike is in `3-execute-T10-spike.md`.

## Example

- Question q001 asks for the beam size of one paper. Keyword search matches the paper's name directly. Vector search sometimes ranks other report-generation papers first.

## Result

| Run | nDCG@5 (0 to 1) |
|---|---|
| dense only (`prefix`) | 0.565 |
| hybrid only | 0.697 |
| dense plus reranker (adopted) | 0.683 |
| hybrid plus reranker | 0.706 |

- **The deciding comparison:** hybrid plus reranker against the adopted reranker is +0.023, range -0.0237 to +0.0840. The lower end is below 0, so the reranker alone stays.
- **Hybrid alone** is +0.132 over dense and +0.014 over the reranker (not a safe difference). It costs milliseconds, the reranker about 1.7 s.
- **The pool grew little:** labelled sections inside the 20 hits go from 0.826 to 0.857.
- **Labels:** 48 new top-5 sections were judged. 2 became labels, 46 were rejected. The golden set is now 65 questions, 161 labels, hash `7f236dd41db4`.
  - Details: reports `3-execute-T10-labels.md`, `data/eval/reports/compare-hybrid_rerank-vs-rerank_minilm.md`, `compare-hybrid-vs-prefix.md`. Wins, losses, ties for the deciding comparison: 7 / 8 / 30. No type clearly worse. Off-corpus no-result rate 0.00 everywhere.

## Not checked

- The cost of the BM25 query inside the app (quality only was measured).
- Tuning of `k` and list weights (65 questions would fit noise).
- Hybrid alone as a cheaper alternative to the reranker. It needs its own comparison and ADR.
- The golden set is LLM-labelled by Claude, not human-verified.
- Runs for bge and MedCPT stay on the earlier hash (not used).
