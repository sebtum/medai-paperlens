# Comparison: rerank_bge_orig against rerank_bge

Note: golden set LLM-labelled (Claude), not human-verified.

- A (baseline side): `rerank_bge`, config `{"collection": "papers", "config_name": "rerank_bge", "depth": 20, "embedder": "BAAI/bge-small-en-v1.5", "golden_validated": true, "prefilter_threshold": 0.6, "prefix": "Represent this sentence for searching relevant passages: ", "rerank_query": "rewritten", "reranker": "BAAI/bge-reranker-base", "reranker_disk_bytes": 1134410492, "reranker_load_s": 5.28, "retrieval_threshold": 0.0, "rewrite": true, "rewrite_model": "qwen3:0.6b", "rewrite_prompt_sha256": "945eb0733345db9575c97f3d9224c9583cb474ea503318a19ef86bf7b06f5243", "score_kind": "rerank"}`, golden sha256 `cc08395a87ca`, git `3d61760-dirty`, 595 chunks in the collection
- B (candidate side): `rerank_bge_orig`, config `{"collection": "papers", "config_name": "rerank_bge", "depth": 20, "embedder": "BAAI/bge-small-en-v1.5", "golden_validated": true, "prefilter_threshold": 0.6, "prefix": "Represent this sentence for searching relevant passages: ", "rerank_query": "original", "reranker": "BAAI/bge-reranker-base", "reranker_disk_bytes": 1134410492, "reranker_load_s": 5.12, "retrieval_threshold": 0.0, "rewrite": true, "rewrite_model": "qwen3:0.6b", "rewrite_prompt_sha256": "945eb0733345db9575c97f3d9224c9583cb474ea503318a19ef86bf7b06f5243", "score_kind": "rerank"}`, golden sha256 `cc08395a87ca`, git `3d61760-dirty`, 595 chunks in the collection
- Metric: ndcg@5 on 45 on-corpus questions; A at k=5, t=0.00, B at k=5, t=0.00
- Bootstrap: 1000 resamples within each type, numpy default_rng(0), 95 % percentile range

## Decision

**A stays**: lower end of the range is -0.0419, not above 0; clearly worse on: paraphrased.

| Mean A | Mean B | Mean delta (B - A) | 95 % range | Wins / losses / ties |
|---|---|---|---|---|
| 0.704 | 0.685 | -0.019 | [-0.0419, +0.0038] | 7 / 7 / 31 |

Rule: B replaces A only if the lower end of the range is above 0 and no type is clearly worse (mean delta <= -0.05 and more losses than wins). A tie is a per-question difference below 1e-9.

## By question type

| Type | n | Mean A | Mean B | Mean delta | W / L / T | Clearly worse |
|---|---|---|---|---|---|---|
| factual | 10 | 0.876 | 0.876 | +0.000 | 0 / 0 / 10 | no |
| paraphrased | 10 | 0.622 | 0.553 | -0.069 | 3 / 4 / 3 | yes |
| cross_paper | 8 | 0.410 | 0.398 | -0.011 | 2 / 1 / 5 | no |
| table | 8 | 0.791 | 0.783 | -0.008 | 1 / 1 / 6 | no |
| multi_section | 9 | 0.787 | 0.787 | +0.000 | 1 / 1 / 7 | no |

## Off-corpus no-result rate

| A | B | Off-corpus questions |
|---|---|---|
| 0.00 | 0.00 | 20 |

Few questions in: cross_paper (n=8), table (n=8), multi_section (n=9). Per-type deltas are rough.
