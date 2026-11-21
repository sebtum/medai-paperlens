# Retrieval run: hybrid_rerank

Note: golden set LLM-labelled (Claude), not human-verified.

- Run: `hybrid_rerank`, config `{"collection": "eval_hybrid", "config_name": "hybrid_rerank", "depth": 20, "embedder": "BAAI/bge-small-en-v1.5", "fusion": "rrf", "golden_validated": true, "hybrid": true, "prefilter_threshold": 0.6, "prefix": "Represent this sentence for searching relevant passages: ", "rerank_query": "rewritten", "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2", "reranker_disk_bytes": 890152505, "reranker_load_s": 4.21, "retrieval_threshold": 0.0, "rewrite": true, "rewrite_model": "qwen3:0.6b", "rewrite_prompt_sha256": "945eb0733345db9575c97f3d9224c9583cb474ea503318a19ef86bf7b06f5243", "score_kind": "rerank"}`, golden sha256 `0c62886fd489`, git `2fe32a5-dirty`, 595 chunks in the collection
- Score kind: rerank; threshold: 0.00; 45 on-corpus and 20 off-corpus questions

## By k (on-corpus questions)

Precision is the share of the top-k chunks that sit in a labelled section (how much of the context is on target). Recall vs ceiling is recall divided by the best the k slots allow, min(k, number of labels); 1.00 means as many labelled sections as k slots can hold. Both are reported, not used by the decision rule. Wrong section is the share of the top-k chunks that come from a paper with a label for the question but from a section without one (right paper, wrong section). Crowding is the mean number of distinct papers in the top-k chunks. Table hit rate is the share of `table` questions with a table chunk in a labelled section in the top-k. Evidence tokens are the mean token estimate of the top-k chunks (n/a when the run holds no token counts). The default k stays 5; this table only measures.

| k | Recall | nDCG | MRR | Precision | Recall vs ceiling | Wrong section | Crowding | Table hit rate | Evidence tokens |
|---|---|---|---|---|---|---|---|---|---|
| 3 | 0.586 | 0.709 | 0.867 | 0.615 | 0.726 | 0.252 | 1.44 | 0.500 | 1525 |
| 5 | 0.674 | 0.706 | 0.867 | 0.542 | 0.711 | 0.311 | 1.69 | 0.625 | 2550 |
| 10 | 0.786 | 0.745 | 0.867 | 0.431 | 0.786 | 0.376 | 2.20 | 0.875 | 4950 |

## Hybrid search

Hits come from one Qdrant query: a dense list (cosine at or above 0.60) and a BM25 keyword list, merged by reciprocal rank fusion (RRF). The keyword query is the same rewritten question without the dense instruction prefix.

## Reranker

Scores are reranker output (sigmoid, 0 to 1), not cosine. The time is one reranker call per question, measured after one warm-up call, on this machine; it excludes the vector search and the rewrite.

| Measure | Value |
|---|---|
| Reranker model | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Query text | rewritten |
| Cosine cut of the dense list (inside the hybrid query) | 0.60 |
| Load time | 4.2 s |
| Disk size | 890 MB |
| Time per question, median | 1750 ms (n=65) |
| Time per question, 90th percentile | 1951 ms |

## By question type (k=5)

The n-gram overlap is the longest run of words a question shares with its source chunks (U3 probe: a paraphrased question should have a low overlap).

| Type | n | Recall@5 | nDCG@5 | MRR@5 | Precision@5 | Recall@5 vs ceiling | Empty rate | Mean overlap | Max overlap |
|---|---|---|---|---|---|---|---|---|---|
| factual | 10 | 0.850 | 0.839 | 0.850 | 0.340 | 0.850 | 0.00 | 2.9 | 4 |
| paraphrased | 10 | 0.662 | 0.641 | 0.950 | 0.560 | 0.662 | 0.00 | 2.0 | 3 |
| cross_paper | 8 | 0.284 | 0.427 | 0.729 | 0.475 | 0.406 | 0.00 | 2.6 | 4 |
| table | 8 | 0.958 | 0.843 | 0.833 | 0.575 | 0.958 | 0.00 | 2.4 | 5 |
| multi_section | 9 | 0.588 | 0.757 | 0.944 | 0.778 | 0.663 | 0.00 | 2.6 | 3 |
| off_corpus | 20 | n/a | n/a | n/a | n/a | n/a | 0.00 | n/a | n/a |

## Thresholded results

| Measure | Value |
|---|---|
| On-corpus empty-result rate (lower is better) | 0.00 |
| Off-corpus no-result rate (higher is better) | 0.00 |
| Max off-corpus top-1 score | 0.995 |

Few questions: cross_paper (n=8), table (n=8), multi_section (n=9). Treat these rows as rough.
