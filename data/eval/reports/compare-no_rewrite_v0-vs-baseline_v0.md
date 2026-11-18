# Comparison: no_rewrite_v0 against baseline_v0

Note: golden set LLM-labelled (Claude), not human-verified.

- A (baseline side): `baseline_v0`, config `{"collection": "papers", "config_name": "baseline", "depth": 20, "embedder": "BAAI/bge-small-en-v1.5", "golden_validated": true, "prefix": "", "retrieval_threshold": 0.0, "rewrite": true, "rewrite_model": "qwen3:0.6b", "rewrite_prompt_sha256": "d34bb79357e76c0f34c059b398cb3178ff6dd2dde15654f5bbff495f2bbf4277"}`, golden sha256 `889f6d00b903`, git `a756d78`, 595 chunks in the collection
- B (candidate side): `no_rewrite_v0`, config `{"collection": "papers", "config_name": "no_rewrite", "depth": 20, "embedder": "BAAI/bge-small-en-v1.5", "golden_validated": true, "prefix": "", "retrieval_threshold": 0.0, "rewrite": false, "rewrite_model": "", "rewrite_prompt_sha256": null}`, golden sha256 `889f6d00b903`, git `a756d78`, 595 chunks in the collection
- Metric: ndcg@5 on 45 on-corpus questions; A at k=5, t=0.60, B at k=5, t=0.60
- Bootstrap: 1000 resamples within each type, numpy default_rng(0), 95 % percentile range

## Decision

**A stays**: lower end of the range is -0.0057, not above 0.

| Mean A | Mean B | Mean delta (B - A) | 95 % range | Wins / losses / ties |
|---|---|---|---|---|
| 0.521 | 0.540 | +0.020 | [-0.0057, +0.0459] | 8 / 6 / 31 |

Rule: B replaces A only if the lower end of the range is above 0 and no type is clearly worse (mean delta <= -0.05 and more losses than wins). A tie is a per-question difference below 1e-9.

## By question type

| Type | n | Mean A | Mean B | Mean delta | W / L / T | Clearly worse |
|---|---|---|---|---|---|---|
| factual | 10 | 0.676 | 0.676 | +0.000 | 0 / 0 / 10 | no |
| paraphrased | 10 | 0.459 | 0.460 | +0.001 | 3 / 4 / 3 | no |
| cross_paper | 8 | 0.295 | 0.310 | +0.015 | 1 / 1 / 6 | no |
| table | 8 | 0.565 | 0.569 | +0.005 | 1 / 1 / 6 | no |
| multi_section | 9 | 0.578 | 0.659 | +0.081 | 3 / 0 / 6 | no |

## Off-corpus no-result rate

| A | B | Off-corpus questions |
|---|---|---|
| 0.00 | 0.00 | 5 |

Few questions in: cross_paper (n=8), table (n=8), multi_section (n=9). Per-type deltas are rough.
Only 5 off-corpus questions, so the no-result rate moves in steps of 0.20. The threshold part is coarse.
