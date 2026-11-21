# Retrieval run: chunk256_merge

Note: golden set LLM-labelled (Claude), not human-verified.

- Run: `chunk256_merge`, config `{"collection": "eval_chunk256_merge", "config_name": "rerank_minilm", "depth": 20, "embedder": "BAAI/bge-small-en-v1.5", "golden_validated": true, "prefilter_threshold": 0.6, "prefix": "Represent this sentence for searching relevant passages: ", "rechunked": true, "rerank_query": "rewritten", "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2", "reranker_disk_bytes": 890152505, "reranker_load_s": 4.22, "retrieval_threshold": 0.0, "rewrite": true, "rewrite_model": "qwen3:0.6b", "rewrite_prompt_sha256": "945eb0733345db9575c97f3d9224c9583cb474ea503318a19ef86bf7b06f5243", "score_kind": "rerank"}`, golden sha256 `0c62886fd489`, git `2fe32a5-dirty`, 1258 chunks in the collection
- Score kind: rerank; threshold: 0.00; 45 on-corpus and 20 off-corpus questions

## By k (on-corpus questions)

Precision is the share of the top-k chunks that sit in a labelled section (how much of the context is on target). Recall vs ceiling is recall divided by the best the k slots allow, min(k, number of labels); 1.00 means as many labelled sections as k slots can hold. Both are reported, not used by the decision rule. Wrong section is the share of the top-k chunks that come from a paper with a label for the question but from a section without one (right paper, wrong section). Crowding is the mean number of distinct papers in the top-k chunks. Table hit rate is the share of `table` questions with a table chunk in a labelled section in the top-k. Evidence tokens are the mean token estimate of the top-k chunks (n/a when the run holds no token counts). The default k stays 5; this table only measures.

| k | Recall | nDCG | MRR | Precision | Recall vs ceiling | Wrong section | Crowding | Table hit rate | Evidence tokens |
|---|---|---|---|---|---|---|---|---|---|
| 3 | 0.568 | 0.708 | 0.859 | 0.652 | 0.704 | 0.207 | 1.42 | 0.500 | 783 |
| 5 | 0.639 | 0.703 | 0.869 | 0.573 | 0.674 | 0.258 | 1.67 | 0.750 | 1282 |
| 10 | 0.750 | 0.741 | 0.869 | 0.464 | 0.750 | 0.356 | 2.00 | 0.750 | 2488 |

## Reranker

Scores are reranker output (sigmoid, 0 to 1), not cosine. The time is one reranker call per question, measured after one warm-up call, on this machine; it excludes the vector search and the rewrite.

| Measure | Value |
|---|---|
| Reranker model | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Query text | rewritten |
| Cosine pre-filter | 0.60 |
| Load time | 4.2 s |
| Disk size | 890 MB |
| Time per question, median | 795 ms (n=65) |
| Time per question, 90th percentile | 859 ms |

## By question type (k=5)

The n-gram overlap is the longest run of words a question shares with its source chunks (U3 probe: a paraphrased question should have a low overlap).

| Type | n | Recall@5 | nDCG@5 | MRR@5 | Precision@5 | Recall@5 vs ceiling | Empty rate | Mean overlap | Max overlap |
|---|---|---|---|---|---|---|---|---|---|
| factual | 10 | 0.850 | 0.876 | 0.900 | 0.400 | 0.850 | 0.00 | 1.1 | 2 |
| paraphrased | 10 | 0.558 | 0.617 | 0.900 | 0.500 | 0.558 | 0.00 | 1.6 | 3 |
| cross_paper | 8 | 0.283 | 0.410 | 0.723 | 0.525 | 0.406 | 0.00 | 2.0 | 3 |
| table | 8 | 0.896 | 0.842 | 0.875 | 0.625 | 0.896 | 0.00 | 1.8 | 3 |
| multi_section | 9 | 0.584 | 0.746 | 0.926 | 0.844 | 0.650 | 0.00 | 2.3 | 4 |
| off_corpus | 20 | n/a | n/a | n/a | n/a | n/a | 0.00 | n/a | n/a |

## Thresholded results

| Measure | Value |
|---|---|
| On-corpus empty-result rate (lower is better) | 0.00 |
| Off-corpus no-result rate (higher is better) | 0.00 |
| Max off-corpus top-1 score | 0.998 |

Few questions: cross_paper (n=8), table (n=8), multi_section (n=9). Treat these rows as rough.
