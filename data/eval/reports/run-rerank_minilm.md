# Retrieval run: rerank_minilm

Note: golden set LLM-labelled (Claude), not human-verified.

- Run: `rerank_minilm`, config `{"collection": "papers", "config_name": "rerank_minilm", "depth": 20, "embedder": "BAAI/bge-small-en-v1.5", "golden_validated": true, "prefilter_threshold": 0.6, "prefix": "Represent this sentence for searching relevant passages: ", "rerank_query": "rewritten", "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2", "reranker_disk_bytes": 890152505, "reranker_load_s": 5.72, "retrieval_threshold": 0.0, "rewrite": true, "rewrite_model": "qwen3:0.6b", "rewrite_prompt_sha256": "945eb0733345db9575c97f3d9224c9583cb474ea503318a19ef86bf7b06f5243", "score_kind": "rerank"}`, golden sha256 `7f236dd41db4`, git `0d961f8-dirty`, 595 chunks in the collection
- Score kind: rerank; threshold: 0.00; 45 on-corpus and 20 off-corpus questions

## By k (on-corpus questions)

Precision is the share of the top-k chunks that sit in a labelled section (how much of the context is on target). Recall vs ceiling is recall divided by the best the k slots allow, min(k, number of labels); 1.00 means as many labelled sections as k slots can hold. Both are reported, not used by the decision rule. Wrong section is the share of the top-k chunks that come from a paper with a label for the question but from a section without one (right paper, wrong section). Crowding is the mean number of distinct papers in the top-k chunks. Table hit rate is the share of `table` questions with a table chunk in a labelled section in the top-k. Evidence tokens are the mean token estimate of the top-k chunks (n/a when the run holds no token counts). The default k stays 5; this table only measures.

| k | Recall | nDCG | MRR | Precision | Recall vs ceiling | Wrong section | Crowding | Table hit rate | Evidence tokens |
|---|---|---|---|---|---|---|---|---|---|
| 3 | 0.553 | 0.691 | 0.852 | 0.607 | 0.685 | 0.230 | 1.51 | 0.500 | 1560 |
| 5 | 0.632 | 0.683 | 0.852 | 0.520 | 0.662 | 0.284 | 1.76 | 0.625 | 2556 |
| 10 | 0.799 | 0.748 | 0.867 | 0.413 | 0.799 | 0.365 | 2.27 | 0.875 | 4875 |

## Reranker

Scores are reranker output (sigmoid, 0 to 1), not cosine. The time is one reranker call per question, measured after one warm-up call, on this machine; it excludes the vector search and the rewrite.

| Measure | Value |
|---|---|
| Reranker model | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Query text | rewritten |
| Cosine pre-filter | 0.60 |
| Load time | 5.7 s |
| Disk size | 890 MB |
| Time per question, median | 3570 ms (n=65) |
| Time per question, 90th percentile | 3695 ms |

## By question type (k=5)

The n-gram overlap is the longest run of words a question shares with its source chunks (U3 probe: a paraphrased question should have a low overlap).

| Type | n | Recall@5 | nDCG@5 | MRR@5 | Precision@5 | Recall@5 vs ceiling | Empty rate | Mean overlap | Max overlap |
|---|---|---|---|---|---|---|---|---|---|
| factual | 10 | 0.750 | 0.776 | 0.800 | 0.320 | 0.750 | 0.00 | 2.9 | 4 |
| paraphrased | 10 | 0.662 | 0.642 | 0.933 | 0.520 | 0.662 | 0.00 | 2.0 | 3 |
| cross_paper | 8 | 0.301 | 0.401 | 0.688 | 0.475 | 0.406 | 0.00 | 2.6 | 4 |
| table | 8 | 0.833 | 0.796 | 0.812 | 0.550 | 0.833 | 0.00 | 2.4 | 5 |
| multi_section | 9 | 0.584 | 0.775 | 1.000 | 0.756 | 0.641 | 0.00 | 2.6 | 3 |
| off_corpus | 20 | n/a | n/a | n/a | n/a | n/a | 0.00 | n/a | n/a |

## Thresholded results

| Measure | Value |
|---|---|
| On-corpus empty-result rate (lower is better) | 0.00 |
| Off-corpus no-result rate (higher is better) | 0.00 |
| Max off-corpus top-1 score | 0.995 |

Few questions: cross_paper (n=8), table (n=8), multi_section (n=9). Treat these rows as rough.
