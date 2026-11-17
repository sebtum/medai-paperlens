# Retrieval run: bge_base_v0

Note: golden set LLM-labelled (Claude), not human-verified.

- Run: `bge_base_v0`, config `{"collection": "eval_bge-base-en-v1.5", "config_name": "bge_base", "depth": 20, "embedder": "BAAI/bge-base-en-v1.5", "golden_validated": true, "prefix": "", "retrieval_threshold": 0.0, "rewrite": true, "rewrite_model": "qwen3:0.6b", "rewrite_prompt_sha256": "d34bb79357e76c0f34c059b398cb3178ff6dd2dde15654f5bbff495f2bbf4277"}`, golden sha256 `889f6d00b903`, git `a756d78`, 595 chunks in the collection
- Threshold: 0.60; 45 on-corpus and 5 off-corpus questions

## By k (on-corpus questions)

Crowding is the mean number of distinct papers in the top-k chunks. Table hit rate is the share of `table` questions with a table chunk in a labelled section in the top-k. Evidence tokens are the mean token estimate of the top-k chunks (n/a when the run holds no token counts). The default k stays 5; this table only measures.

| k | Recall | nDCG | MRR | Crowding | Table hit rate | Evidence tokens |
|---|---|---|---|---|---|---|
| 3 | 0.438 | 0.528 | 0.681 | 1.53 | 0.625 | 1403 |
| 5 | 0.549 | 0.544 | 0.694 | 1.71 | 0.625 | 2319 |
| 10 | 0.679 | 0.606 | 0.711 | 2.07 | 0.625 | 4429 |

## By question type (k=5)

The n-gram overlap is the longest run of words a question shares with its source chunks (U3 probe: a paraphrased question should have a low overlap).

| Type | n | Recall@5 | nDCG@5 | MRR@5 | Empty rate | Mean overlap | Max overlap |
|---|---|---|---|---|---|---|---|
| factual | 10 | 0.750 | 0.682 | 0.675 | 0.00 | 2.9 | 4 |
| paraphrased | 10 | 0.580 | 0.493 | 0.667 | 0.00 | 2.0 | 3 |
| cross_paper | 8 | 0.284 | 0.343 | 0.604 | 0.00 | 2.6 | 4 |
| table | 8 | 0.521 | 0.496 | 0.500 | 0.00 | 2.4 | 5 |
| multi_section | 9 | 0.552 | 0.670 | 1.000 | 0.00 | 2.6 | 3 |
| off_corpus | 5 | n/a | n/a | n/a | 0.00 | n/a | n/a |

## Thresholded results

| Measure | Value |
|---|---|
| On-corpus empty-result rate (lower is better) | 0.00 |
| Off-corpus no-result rate (higher is better) | 0.00 |
| Max off-corpus top-1 score | 0.788 |

Few questions: cross_paper (n=8), table (n=8), multi_section (n=9), off_corpus (n=5). Treat these rows as rough.
