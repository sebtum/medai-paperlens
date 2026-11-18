# Comparison: prefix against baseline

Note: golden set LLM-labelled (Claude), not human-verified.

- A (baseline side): `baseline`, config `{"collection": "papers", "config_name": "baseline", "depth": 20, "embedder": "BAAI/bge-small-en-v1.5", "golden_validated": true, "prefix": "", "reranker": null, "retrieval_threshold": 0.0, "rewrite": true, "rewrite_model": "qwen3:0.6b", "rewrite_prompt_sha256": "945eb0733345db9575c97f3d9224c9583cb474ea503318a19ef86bf7b06f5243", "score_kind": "cosine"}`, golden sha256 `4a4733e5fa76`, git `44b51cb-dirty`, 595 chunks in the collection
- B (candidate side): `prefix`, config `{"collection": "papers", "config_name": "prefix", "depth": 20, "embedder": "BAAI/bge-small-en-v1.5", "golden_validated": true, "prefix": "Represent this sentence for searching relevant passages: ", "reranker": null, "retrieval_threshold": 0.0, "rewrite": true, "rewrite_model": "qwen3:0.6b", "rewrite_prompt_sha256": "945eb0733345db9575c97f3d9224c9583cb474ea503318a19ef86bf7b06f5243", "score_kind": "cosine"}`, golden sha256 `4a4733e5fa76`, git `44b51cb-dirty`, 595 chunks in the collection
- Metric: ndcg@5 on 45 on-corpus questions; A at k=5, t=0.60, B at k=5, t=0.60
- Bootstrap: 1000 resamples within each type, numpy default_rng(0), 95 % percentile range

## Decision

**B replaces A**: lower end of the range is above 0 and no type is clearly worse.

| Mean A | Mean B | Mean delta (B - A) | 95 % range | Wins / losses / ties |
|---|---|---|---|---|
| 0.532 | 0.562 | +0.030 | [+0.0052, +0.0564] | 11 / 6 / 28 |

Rule: B replaces A only if the lower end of the range is above 0 and no type is clearly worse (mean delta <= -0.05 and more losses than wins). A tie is a per-question difference below 1e-9.

## By question type

| Type | n | Mean A | Mean B | Mean delta | W / L / T | Clearly worse |
|---|---|---|---|---|---|---|
| factual | 10 | 0.676 | 0.676 | +0.000 | 0 / 0 / 10 | no |
| paraphrased | 10 | 0.452 | 0.515 | +0.063 | 5 / 2 / 3 | no |
| cross_paper | 8 | 0.292 | 0.292 | +0.000 | 0 / 0 / 8 | no |
| table | 8 | 0.565 | 0.614 | +0.049 | 3 / 1 / 4 | no |
| multi_section | 9 | 0.644 | 0.682 | +0.038 | 3 / 3 / 3 | no |

## Off-corpus no-result rate

| A | B | Off-corpus questions |
|---|---|---|
| 0.00 | 0.00 | 5 |

Few questions in: cross_paper (n=8), table (n=8), multi_section (n=9). Per-type deltas are rough.
Only 5 off-corpus questions, so the no-result rate moves in steps of 0.20. The threshold part is coarse.
