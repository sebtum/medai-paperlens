# Retrieval run: no_rewrite

Note: golden set LLM-labelled (Claude), not human-verified.

- Run: `no_rewrite`, config `{"collection": "papers", "config_name": "no_rewrite", "depth": 20, "embedder": "BAAI/bge-small-en-v1.5", "golden_validated": true, "prefix": "", "reranker": null, "retrieval_threshold": 0.0, "rewrite": false, "rewrite_model": "", "rewrite_prompt_sha256": null, "score_kind": "cosine"}`, golden sha256 `cc08395a87ca`, git `3d61760-dirty`, 595 chunks in the collection
- Score kind: cosine; threshold: 0.60; 45 on-corpus and 20 off-corpus questions

## By k (on-corpus questions)

Precision is the share of the top-k chunks that sit in a labelled section (how much of the context is on target). Recall vs ceiling is recall divided by the best the k slots allow, min(k, number of labels); 1.00 means as many labelled sections as k slots can hold. Both are reported, not used by the decision rule. Wrong section is the share of the top-k chunks that come from a paper with a label for the question but from a section without one (right paper, wrong section). Crowding is the mean number of distinct papers in the top-k chunks. Table hit rate is the share of `table` questions with a table chunk in a labelled section in the top-k. Evidence tokens are the mean token estimate of the top-k chunks (n/a when the run holds no token counts). The default k stays 5; this table only measures.

| k | Recall | nDCG | MRR | Precision | Recall vs ceiling | Wrong section | Crowding | Table hit rate | Evidence tokens |
|---|---|---|---|---|---|---|---|---|---|
| 3 | 0.456 | 0.553 | 0.722 | 0.533 | 0.567 | 0.289 | 1.38 | 0.625 | 1440 |
| 5 | 0.506 | 0.546 | 0.727 | 0.458 | 0.529 | 0.347 | 1.67 | 0.750 | 2368 |
| 10 | 0.713 | 0.626 | 0.739 | 0.369 | 0.713 | 0.411 | 2.36 | 0.875 | 4690 |

## By question type (k=5)

The n-gram overlap is the longest run of words a question shares with its source chunks (U3 probe: a paraphrased question should have a low overlap).

| Type | n | Recall@5 | nDCG@5 | MRR@5 | Precision@5 | Recall@5 vs ceiling | Empty rate | Mean overlap | Max overlap |
|---|---|---|---|---|---|---|---|---|---|
| factual | 10 | 0.650 | 0.676 | 0.700 | 0.260 | 0.650 | 0.00 | 2.9 | 4 |
| paraphrased | 10 | 0.485 | 0.452 | 0.703 | 0.480 | 0.485 | 0.00 | 2.0 | 3 |
| cross_paper | 8 | 0.263 | 0.321 | 0.604 | 0.350 | 0.338 | 0.00 | 2.6 | 4 |
| table | 8 | 0.521 | 0.569 | 0.688 | 0.450 | 0.521 | 0.00 | 2.4 | 5 |
| multi_section | 9 | 0.570 | 0.683 | 0.926 | 0.756 | 0.620 | 0.00 | 2.6 | 3 |
| off_corpus | 20 | n/a | n/a | n/a | n/a | n/a | 0.00 | n/a | n/a |

## Thresholded results

| Measure | Value |
|---|---|
| On-corpus empty-result rate (lower is better) | 0.00 |
| Off-corpus no-result rate (higher is better) | 0.00 |
| Max off-corpus top-1 score | 0.870 |

Few questions: cross_paper (n=8), table (n=8), multi_section (n=9). Treat these rows as rough.
