# Comparison: bge_base against baseline

Note: golden set LLM-labelled (Claude), not human-verified.

- A (baseline side): `baseline`, config `{"collection": "papers", "config_name": "baseline", "depth": 20, "embedder": "BAAI/bge-small-en-v1.5", "golden_validated": true, "prefix": "", "retrieval_threshold": 0.0, "rewrite": true, "rewrite_model": "qwen3:0.6b", "rewrite_prompt_sha256": "d34bb79357e76c0f34c059b398cb3178ff6dd2dde15654f5bbff495f2bbf4277"}`, golden sha256 `4a4733e5fa76`, git `11318b7`, 595 chunks in the collection
- B (candidate side): `bge_base`, config `{"collection": "eval_bge-base-en-v1.5", "config_name": "bge_base", "depth": 20, "embedder": "BAAI/bge-base-en-v1.5", "golden_validated": true, "prefix": "", "retrieval_threshold": 0.0, "rewrite": true, "rewrite_model": "qwen3:0.6b", "rewrite_prompt_sha256": "d34bb79357e76c0f34c059b398cb3178ff6dd2dde15654f5bbff495f2bbf4277"}`, golden sha256 `4a4733e5fa76`, git `11318b7`, 595 chunks in the collection
- Metric: ndcg@5 on 45 on-corpus questions; A at k=5, t=0.60, B at k=5, t=0.60
- Bootstrap: 1000 resamples within each type, numpy default_rng(0), 95 % percentile range

## Decision

**A stays**: lower end of the range is -0.0347, not above 0; clearly worse on: table.

| Mean A | Mean B | Mean delta (B - A) | 95 % range | Wins / losses / ties |
|---|---|---|---|---|
| 0.525 | 0.550 | +0.026 | [-0.0347, +0.0835] | 17 / 14 / 14 |

Rule: B replaces A only if the lower end of the range is above 0 and no type is clearly worse (mean delta <= -0.05 and more losses than wins). A tie is a per-question difference below 1e-9.

## By question type

| Type | n | Mean A | Mean B | Mean delta | W / L / T | Clearly worse |
|---|---|---|---|---|---|---|
| factual | 10 | 0.676 | 0.682 | +0.006 | 1 / 1 / 8 | no |
| paraphrased | 10 | 0.452 | 0.487 | +0.035 | 5 / 4 / 1 | no |
| cross_paper | 8 | 0.292 | 0.379 | +0.087 | 5 / 2 / 1 | no |
| table | 8 | 0.565 | 0.496 | -0.069 | 1 / 3 / 4 | yes |
| multi_section | 9 | 0.610 | 0.675 | +0.065 | 5 / 4 / 0 | no |

## Off-corpus no-result rate

| A | B | Off-corpus questions |
|---|---|---|
| 0.00 | 0.00 | 5 |

Few questions in: cross_paper (n=8), table (n=8), multi_section (n=9). Per-type deltas are rough.
Only 5 off-corpus questions, so the no-result rate moves in steps of 0.20. The threshold part is coarse.
