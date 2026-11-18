# Retrieval run: baseline_v0

Note: golden set LLM-labelled (Claude), not human-verified.

- Run: `baseline_v0`, config `{"collection": "papers", "config_name": "baseline", "depth": 20, "embedder": "BAAI/bge-small-en-v1.5", "golden_validated": true, "prefix": "", "retrieval_threshold": 0.0, "rewrite": true, "rewrite_model": "qwen3:0.6b", "rewrite_prompt_sha256": "d34bb79357e76c0f34c059b398cb3178ff6dd2dde15654f5bbff495f2bbf4277"}`, golden sha256 `889f6d00b903`, git `a756d78`, 595 chunks in the collection
- Threshold: 0.60; 45 on-corpus and 5 off-corpus questions

## By k (on-corpus questions)

Crowding is the mean number of distinct papers in the top-k chunks. Table hit rate is the share of `table` questions with a table chunk in a labelled section in the top-k. Evidence tokens are the mean token estimate of the top-k chunks (n/a when the run holds no token counts). The default k stays 5; this table only measures.

| k | Recall | nDCG | MRR | Crowding | Table hit rate | Evidence tokens |
|---|---|---|---|---|---|---|
| 3 | 0.474 | 0.530 | 0.678 | 1.40 | 0.625 | 1465 |
| 5 | 0.510 | 0.521 | 0.688 | 1.80 | 0.625 | 2430 |
| 10 | 0.681 | 0.592 | 0.700 | 2.47 | 0.875 | 4709 |

## By question type (k=5)

The n-gram overlap is the longest run of words a question shares with its source chunks (U3 probe: a paraphrased question should have a low overlap).

| Type | n | Recall@5 | nDCG@5 | MRR@5 | Empty rate | Mean overlap | Max overlap |
|---|---|---|---|---|---|---|---|
| factual | 10 | 0.650 | 0.676 | 0.700 | 0.00 | 2.9 | 4 |
| paraphrased | 10 | 0.465 | 0.459 | 0.745 | 0.00 | 2.0 | 3 |
| cross_paper | 8 | 0.243 | 0.295 | 0.625 | 0.00 | 2.6 | 4 |
| table | 8 | 0.583 | 0.565 | 0.625 | 0.00 | 2.4 | 5 |
| multi_section | 9 | 0.576 | 0.578 | 0.722 | 0.00 | 2.6 | 3 |
| off_corpus | 5 | n/a | n/a | n/a | 0.00 | n/a | n/a |

## Thresholded results

| Measure | Value |
|---|---|
| On-corpus empty-result rate (lower is better) | 0.00 |
| Off-corpus no-result rate (higher is better) | 0.00 |
| Max off-corpus top-1 score | 0.829 |

Few questions: cross_paper (n=8), table (n=8), multi_section (n=9), off_corpus (n=5). Treat these rows as rough.
