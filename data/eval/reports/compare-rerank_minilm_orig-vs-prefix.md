# Comparison: rerank_minilm_orig against prefix

Note: golden set LLM-labelled (Claude), not human-verified.

- A (baseline side): `prefix`, config `{"collection": "papers", "config_name": "prefix", "depth": 20, "embedder": "BAAI/bge-small-en-v1.5", "golden_validated": true, "prefix": "Represent this sentence for searching relevant passages: ", "reranker": null, "retrieval_threshold": 0.0, "rewrite": true, "rewrite_model": "qwen3:0.6b", "rewrite_prompt_sha256": "945eb0733345db9575c97f3d9224c9583cb474ea503318a19ef86bf7b06f5243", "score_kind": "cosine"}`, golden sha256 `cc08395a87ca`, git `3d61760-dirty`, 595 chunks in the collection
- B (candidate side): `rerank_minilm_orig`, config `{"collection": "papers", "config_name": "rerank_minilm", "depth": 20, "embedder": "BAAI/bge-small-en-v1.5", "golden_validated": true, "prefilter_threshold": 0.6, "prefix": "Represent this sentence for searching relevant passages: ", "rerank_query": "original", "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2", "reranker_disk_bytes": 890152505, "reranker_load_s": 4.14, "retrieval_threshold": 0.0, "rewrite": true, "rewrite_model": "qwen3:0.6b", "rewrite_prompt_sha256": "945eb0733345db9575c97f3d9224c9583cb474ea503318a19ef86bf7b06f5243", "score_kind": "rerank"}`, golden sha256 `cc08395a87ca`, git `3d61760-dirty`, 595 chunks in the collection
- Metric: ndcg@5 on 45 on-corpus questions; A at k=5, t=0.60, B at k=5, t=0.00
- Bootstrap: 1000 resamples within each type, numpy default_rng(0), 95 % percentile range

## Decision

**B replaces A**: lower end of the range is above 0 and no type is clearly worse.

| Mean A | Mean B | Mean delta (B - A) | 95 % range | Wins / losses / ties |
|---|---|---|---|---|
| 0.564 | 0.699 | +0.135 | [+0.0593, +0.2163] | 21 / 7 / 17 |

Rule: B replaces A only if the lower end of the range is above 0 and no type is clearly worse (mean delta <= -0.05 and more losses than wins). A tie is a per-question difference below 1e-9.

## By question type

| Type | n | Mean A | Mean B | Mean delta | W / L / T | Clearly worse |
|---|---|---|---|---|---|---|
| factual | 10 | 0.676 | 0.776 | +0.100 | 1 / 0 / 9 | no |
| paraphrased | 10 | 0.515 | 0.676 | +0.162 | 6 / 4 / 0 | no |
| cross_paper | 8 | 0.306 | 0.376 | +0.070 | 3 / 0 / 5 | no |
| table | 8 | 0.614 | 0.834 | +0.220 | 4 / 1 / 3 | no |
| multi_section | 9 | 0.679 | 0.805 | +0.126 | 7 / 2 / 0 | no |

## Off-corpus no-result rate

| A | B | Off-corpus questions |
|---|---|---|
| 0.00 | 0.00 | 20 |

Few questions in: cross_paper (n=8), table (n=8), multi_section (n=9). Per-type deltas are rough.
