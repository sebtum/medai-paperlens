# Comparison: prefix_v0 against baseline_v0

Note: golden set LLM-labelled (Claude), not human-verified.

- A (baseline side): `baseline_v0`, config `{"collection": "papers", "config_name": "baseline", "depth": 20, "embedder": "BAAI/bge-small-en-v1.5", "golden_validated": true, "prefix": "", "retrieval_threshold": 0.0, "rewrite": true, "rewrite_model": "qwen3:0.6b", "rewrite_prompt_sha256": "d34bb79357e76c0f34c059b398cb3178ff6dd2dde15654f5bbff495f2bbf4277"}`, golden sha256 `889f6d00b903`, git `a756d78`, 595 chunks in the collection
- B (candidate side): `prefix_v0`, config `{"collection": "papers", "config_name": "prefix", "depth": 20, "embedder": "BAAI/bge-small-en-v1.5", "golden_validated": true, "prefix": "Represent this sentence for searching relevant passages: ", "retrieval_threshold": 0.0, "rewrite": true, "rewrite_model": "qwen3:0.6b", "rewrite_prompt_sha256": "d34bb79357e76c0f34c059b398cb3178ff6dd2dde15654f5bbff495f2bbf4277"}`, golden sha256 `889f6d00b903`, git `a756d78`, 595 chunks in the collection
- Metric: ndcg@5 on 45 on-corpus questions; A at k=5, t=0.60, B at k=5, t=0.60
- Bootstrap: 1000 resamples within each type, numpy default_rng(0), 95 % percentile range

## Decision

**B replaces A**: lower end of the range is above 0 and no type is clearly worse.

| Mean A | Mean B | Mean delta (B - A) | 95 % range | Wins / losses / ties |
|---|---|---|---|---|
| 0.521 | 0.561 | +0.040 | [+0.0122, +0.0734] | 11 / 4 / 30 |

Rule: B replaces A only if the lower end of the range is above 0 and no type is clearly worse (mean delta <= -0.05 and more losses than wins). A tie is a per-question difference below 1e-9.

## By question type

| Type | n | Mean A | Mean B | Mean delta | W / L / T | Clearly worse |
|---|---|---|---|---|---|---|
| factual | 10 | 0.676 | 0.676 | +0.000 | 0 / 0 / 10 | no |
| paraphrased | 10 | 0.459 | 0.532 | +0.074 | 5 / 1 / 4 | no |
| cross_paper | 8 | 0.295 | 0.295 | +0.000 | 0 / 0 / 8 | no |
| table | 8 | 0.565 | 0.626 | +0.061 | 3 / 0 / 5 | no |
| multi_section | 9 | 0.578 | 0.643 | +0.065 | 3 / 3 / 3 | no |

## Off-corpus no-result rate

| A | B | Off-corpus questions |
|---|---|---|
| 0.00 | 0.00 | 5 |

Few questions in: cross_paper (n=8), table (n=8), multi_section (n=9). Per-type deltas are rough.
Only 5 off-corpus questions, so the no-result rate moves in steps of 0.20. The threshold part is coarse.
