# Threshold curve: baseline

Note: golden set LLM-labelled (Claude), not human-verified.

- Run: `baseline`, config `{"collection": "papers", "config_name": "baseline", "depth": 20, "embedder": "BAAI/bge-small-en-v1.5", "golden_validated": true, "prefix": "", "retrieval_threshold": 0.0, "rewrite": true, "rewrite_model": "qwen3:0.6b", "rewrite_prompt_sha256": "d34bb79357e76c0f34c059b398cb3178ff6dd2dde15654f5bbff495f2bbf4277"}`, golden sha256 `4a4733e5fa76`, git `11318b7`, 595 chunks in the collection
- Cut-off k=5; 45 on-corpus and 5 off-corpus questions

| t | Recall@5 | nDCG@5 | On-corpus empty rate | Off-corpus no-result rate | Max off-corpus top-1 |
|---|---|---|---|---|---|
| 0.40 | 0.509 | 0.525 | 0.00 | 0.00 | 0.829 |
| 0.42 | 0.509 | 0.525 | 0.00 | 0.00 | 0.829 |
| 0.44 | 0.509 | 0.525 | 0.00 | 0.00 | 0.829 |
| 0.46 | 0.509 | 0.525 | 0.00 | 0.00 | 0.829 |
| 0.48 | 0.509 | 0.525 | 0.00 | 0.00 | 0.829 |
| 0.50 | 0.509 | 0.525 | 0.00 | 0.00 | 0.829 |
| 0.52 | 0.509 | 0.525 | 0.00 | 0.00 | 0.829 |
| 0.54 | 0.509 | 0.525 | 0.00 | 0.00 | 0.829 |
| 0.56 | 0.509 | 0.525 | 0.00 | 0.00 | 0.829 |
| 0.58 | 0.509 | 0.525 | 0.00 | 0.00 | 0.829 |
| 0.60 | 0.509 | 0.525 | 0.00 | 0.00 | 0.829 |
| 0.62 | 0.509 | 0.525 | 0.00 | 0.00 | 0.829 |
| 0.64 | 0.509 | 0.525 | 0.00 | 0.00 | 0.829 |
| 0.66 | 0.509 | 0.525 | 0.00 | 0.00 | 0.829 |
| 0.68 | 0.509 | 0.525 | 0.00 | 0.00 | 0.829 |
| 0.70 | 0.497 | 0.514 | 0.00 | 0.00 | 0.829 |
| 0.72 | 0.464 | 0.488 | 0.04 | 0.00 | 0.829 |
| 0.74 | 0.437 | 0.461 | 0.09 | 0.00 | 0.829 |
| 0.76 | 0.378 | 0.397 | 0.27 | 0.00 | 0.829 |
| 0.78 | 0.355 | 0.372 | 0.31 | 0.20 | 0.829 |
| 0.80 | 0.258 | 0.281 | 0.44 | 0.20 | 0.829 |

A lower threshold replaces 0.60 only if it passes the nDCG@5 rule and the off-corpus no-result rate does not drop. Raising the threshold can only lose recall, so it is reported, not chosen.
Only 5 off-corpus questions: the no-result rate moves in steps of 0.20, so the threshold part is coarse.
