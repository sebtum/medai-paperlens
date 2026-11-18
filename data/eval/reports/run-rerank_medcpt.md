# Retrieval run: rerank_medcpt

Note: golden set LLM-labelled (Claude), not human-verified.

- Run: `rerank_medcpt`, config `{"collection": "papers", "config_name": "rerank_medcpt", "depth": 20, "embedder": "BAAI/bge-small-en-v1.5", "golden_validated": true, "prefilter_threshold": 0.6, "prefix": "Represent this sentence for searching relevant passages: ", "rerank_query": "rewritten", "reranker": "ncbi/MedCPT-Cross-Encoder", "reranker_disk_bytes": 876888660, "reranker_load_s": 2.85, "retrieval_threshold": 0.0, "rewrite": true, "rewrite_model": "qwen3:0.6b", "rewrite_prompt_sha256": "945eb0733345db9575c97f3d9224c9583cb474ea503318a19ef86bf7b06f5243", "score_kind": "rerank"}`, golden sha256 `cc08395a87ca`, git `3d61760-dirty`, 595 chunks in the collection
- Score kind: rerank; threshold: 0.00; 45 on-corpus and 20 off-corpus questions

## By k (on-corpus questions)

Precision is the share of the top-k chunks that sit in a labelled section (how much of the context is on target). Recall vs ceiling is recall divided by the best the k slots allow, min(k, number of labels); 1.00 means as many labelled sections as k slots can hold. Both are reported, not used by the decision rule. Wrong section is the share of the top-k chunks that come from a paper with a label for the question but from a section without one (right paper, wrong section). Crowding is the mean number of distinct papers in the top-k chunks. Table hit rate is the share of `table` questions with a table chunk in a labelled section in the top-k. Evidence tokens are the mean token estimate of the top-k chunks (n/a when the run holds no token counts). The default k stays 5; this table only measures.

| k | Recall | nDCG | MRR | Precision | Recall vs ceiling | Wrong section | Crowding | Table hit rate | Evidence tokens |
|---|---|---|---|---|---|---|---|---|---|
| 3 | 0.547 | 0.656 | 0.844 | 0.593 | 0.678 | 0.244 | 1.38 | 0.500 | 1431 |
| 5 | 0.599 | 0.639 | 0.844 | 0.502 | 0.626 | 0.320 | 1.64 | 0.625 | 2353 |
| 10 | 0.771 | 0.709 | 0.854 | 0.406 | 0.771 | 0.401 | 2.07 | 0.875 | 4538 |

## Reranker

Scores are reranker output (sigmoid, 0 to 1), not cosine. The time is one reranker call per question, measured after one warm-up call, on this machine; it excludes the vector search and the rewrite.

| Measure | Value |
|---|---|
| Reranker model | ncbi/MedCPT-Cross-Encoder |
| Query text | rewritten |
| Cosine pre-filter | 0.60 |
| Load time | 2.9 s |
| Disk size | 877 MB |
| Time per question, median | 8884 ms (n=65) |
| Time per question, 90th percentile | 13055 ms |

## By question type (k=5)

The n-gram overlap is the longest run of words a question shares with its source chunks (U3 probe: a paraphrased question should have a low overlap).

| Type | n | Recall@5 | nDCG@5 | MRR@5 | Precision@5 | Recall@5 vs ceiling | Empty rate | Mean overlap | Max overlap |
|---|---|---|---|---|---|---|---|---|---|
| factual | 10 | 0.750 | 0.702 | 0.700 | 0.260 | 0.750 | 0.00 | 2.9 | 4 |
| paraphrased | 10 | 0.615 | 0.606 | 0.850 | 0.560 | 0.615 | 0.00 | 2.0 | 3 |
| cross_paper | 8 | 0.279 | 0.403 | 0.875 | 0.500 | 0.381 | 0.00 | 2.6 | 4 |
| table | 8 | 0.708 | 0.716 | 0.812 | 0.475 | 0.708 | 0.00 | 2.4 | 5 |
| multi_section | 9 | 0.602 | 0.747 | 1.000 | 0.733 | 0.646 | 0.00 | 2.6 | 3 |
| off_corpus | 20 | n/a | n/a | n/a | n/a | n/a | 0.00 | n/a | n/a |

## Thresholded results

| Measure | Value |
|---|---|
| On-corpus empty-result rate (lower is better) | 0.00 |
| Off-corpus no-result rate (higher is better) | 0.00 |
| Max off-corpus top-1 score | 1.000 |

Few questions: cross_paper (n=8), table (n=8), multi_section (n=9). Treat these rows as rough.
