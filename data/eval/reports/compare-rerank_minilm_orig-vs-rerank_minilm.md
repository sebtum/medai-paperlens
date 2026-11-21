# Comparison: rerank_minilm_orig against rerank_minilm

Note: golden set LLM-labelled (Claude), not human-verified.

- A (baseline side): `rerank_minilm`, config `{"collection": "papers", "config_name": "rerank_minilm", "depth": 20, "embedder": "BAAI/bge-small-en-v1.5", "golden_validated": true, "prefilter_threshold": 0.6, "prefix": "Represent this sentence for searching relevant passages: ", "rerank_query": "rewritten", "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2", "reranker_disk_bytes": 890152505, "reranker_load_s": 4.08, "retrieval_threshold": 0.0, "rewrite": true, "rewrite_model": "qwen3:0.6b", "rewrite_prompt_sha256": "945eb0733345db9575c97f3d9224c9583cb474ea503318a19ef86bf7b06f5243", "score_kind": "rerank"}`, golden sha256 `cc08395a87ca`, git `3d61760-dirty`, 595 chunks in the collection
- B (candidate side): `rerank_minilm_orig`, config `{"collection": "papers", "config_name": "rerank_minilm", "depth": 20, "embedder": "BAAI/bge-small-en-v1.5", "golden_validated": true, "prefilter_threshold": 0.6, "prefix": "Represent this sentence for searching relevant passages: ", "rerank_query": "original", "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2", "reranker_disk_bytes": 890152505, "reranker_load_s": 4.14, "retrieval_threshold": 0.0, "rewrite": true, "rewrite_model": "qwen3:0.6b", "rewrite_prompt_sha256": "945eb0733345db9575c97f3d9224c9583cb474ea503318a19ef86bf7b06f5243", "score_kind": "rerank"}`, golden sha256 `cc08395a87ca`, git `3d61760-dirty`, 595 chunks in the collection
- Metric: ndcg@5 on 45 on-corpus questions; A at k=5, t=0.00, B at k=5, t=0.00
- Bootstrap: 1000 resamples within each type, numpy default_rng(0), 95 % percentile range

## Decision

**A stays**: lower end of the range is -0.0052, not above 0.

| Mean A | Mean B | Mean delta (B - A) | 95 % range | Wins / losses / ties |
|---|---|---|---|---|
| 0.684 | 0.699 | +0.015 | [-0.0052, +0.0363] | 9 / 4 / 32 |

Rule: B replaces A only if the lower end of the range is above 0 and no type is clearly worse (mean delta <= -0.05 and more losses than wins). A tie is a per-question difference below 1e-9.

## By question type

| Type | n | Mean A | Mean B | Mean delta | W / L / T | Clearly worse |
|---|---|---|---|---|---|---|
| factual | 10 | 0.776 | 0.776 | +0.000 | 0 / 0 / 10 | no |
| paraphrased | 10 | 0.642 | 0.676 | +0.034 | 4 / 1 / 5 | no |
| cross_paper | 8 | 0.401 | 0.376 | -0.025 | 2 / 3 / 3 | no |
| table | 8 | 0.796 | 0.834 | +0.038 | 1 / 0 / 7 | no |
| multi_section | 9 | 0.781 | 0.805 | +0.024 | 2 / 0 / 7 | no |

## Off-corpus no-result rate

| A | B | Off-corpus questions |
|---|---|---|
| 0.00 | 0.00 | 20 |

Few questions in: cross_paper (n=8), table (n=8), multi_section (n=9). Per-type deltas are rough.
