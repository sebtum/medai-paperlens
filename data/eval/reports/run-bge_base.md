# Retrieval run: bge_base

Note: golden set LLM-labelled (Claude), not human-verified.

- Run: `bge_base`, config `{"collection": "eval_bge-base-en-v1.5", "config_name": "bge_base", "depth": 20, "embedder": "BAAI/bge-base-en-v1.5", "golden_validated": true, "prefix": "", "reranker": null, "retrieval_threshold": 0.0, "rewrite": true, "rewrite_model": "qwen3:0.6b", "rewrite_prompt_sha256": "945eb0733345db9575c97f3d9224c9583cb474ea503318a19ef86bf7b06f5243", "score_kind": "cosine"}`, golden sha256 `cc08395a87ca`, git `3d61760-dirty`, 595 chunks in the collection
- Score kind: cosine; threshold: 0.60; 45 on-corpus and 20 off-corpus questions

## By k (on-corpus questions)

Precision is the share of the top-k chunks that sit in a labelled section (how much of the context is on target). Recall vs ceiling is recall divided by the best the k slots allow, min(k, number of labels); 1.00 means as many labelled sections as k slots can hold. Both are reported, not used by the decision rule. Wrong section is the share of the top-k chunks that come from a paper with a label for the question but from a section without one (right paper, wrong section). Crowding is the mean number of distinct papers in the top-k chunks. Table hit rate is the share of `table` questions with a table chunk in a labelled section in the top-k. Evidence tokens are the mean token estimate of the top-k chunks (n/a when the run holds no token counts). The default k stays 5; this table only measures.

| k | Recall | nDCG | MRR | Precision | Recall vs ceiling | Wrong section | Crowding | Table hit rate | Evidence tokens |
|---|---|---|---|---|---|---|---|---|---|
| 3 | 0.435 | 0.544 | 0.693 | 0.533 | 0.552 | 0.267 | 1.56 | 0.625 | 1418 |
| 5 | 0.531 | 0.543 | 0.698 | 0.467 | 0.555 | 0.320 | 1.71 | 0.625 | 2321 |
| 10 | 0.668 | 0.600 | 0.714 | 0.360 | 0.668 | 0.398 | 2.07 | 0.625 | 4456 |

## By question type (k=5)

The n-gram overlap is the longest run of words a question shares with its source chunks (U3 probe: a paraphrased question should have a low overlap).

| Type | n | Recall@5 | nDCG@5 | MRR@5 | Precision@5 | Recall@5 vs ceiling | Empty rate | Mean overlap | Max overlap |
|---|---|---|---|---|---|---|---|---|---|
| factual | 10 | 0.750 | 0.645 | 0.625 | 0.300 | 0.750 | 0.00 | 2.9 | 4 |
| paraphrased | 10 | 0.565 | 0.500 | 0.683 | 0.560 | 0.565 | 0.00 | 2.0 | 3 |
| cross_paper | 8 | 0.256 | 0.379 | 0.667 | 0.375 | 0.338 | 0.00 | 2.6 | 4 |
| table | 8 | 0.521 | 0.490 | 0.500 | 0.400 | 0.521 | 0.00 | 2.4 | 5 |
| multi_section | 9 | 0.501 | 0.670 | 1.000 | 0.689 | 0.552 | 0.00 | 2.6 | 3 |
| off_corpus | 20 | n/a | n/a | n/a | n/a | n/a | 0.00 | n/a | n/a |

## Thresholded results

| Measure | Value |
|---|---|
| On-corpus empty-result rate (lower is better) | 0.00 |
| Off-corpus no-result rate (higher is better) | 0.00 |
| Max off-corpus top-1 score | 0.826 |

Few questions: cross_paper (n=8), table (n=8), multi_section (n=9). Treat these rows as rough.
