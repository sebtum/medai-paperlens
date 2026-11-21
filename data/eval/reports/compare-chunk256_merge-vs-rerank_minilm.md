# Comparison: chunk256_merge against rerank_minilm

Note: golden set LLM-labelled (Claude), not human-verified.

- A (baseline side): `rerank_minilm`, config `{"collection": "papers", "config_name": "rerank_minilm", "depth": 20, "embedder": "BAAI/bge-small-en-v1.5", "golden_validated": true, "prefilter_threshold": 0.6, "prefix": "Represent this sentence for searching relevant passages: ", "rerank_query": "rewritten", "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2", "reranker_disk_bytes": 890152505, "reranker_load_s": 4.05, "retrieval_threshold": 0.0, "rewrite": true, "rewrite_model": "qwen3:0.6b", "rewrite_prompt_sha256": "945eb0733345db9575c97f3d9224c9583cb474ea503318a19ef86bf7b06f5243", "score_kind": "rerank"}`, golden sha256 `0c62886fd489`, git `2fe32a5-dirty`, 595 chunks in the collection
- B (candidate side): `chunk256_merge`, config `{"collection": "eval_chunk256_merge", "config_name": "rerank_minilm", "depth": 20, "embedder": "BAAI/bge-small-en-v1.5", "golden_validated": true, "prefilter_threshold": 0.6, "prefix": "Represent this sentence for searching relevant passages: ", "rechunked": true, "rerank_query": "rewritten", "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2", "reranker_disk_bytes": 890152505, "reranker_load_s": 4.22, "retrieval_threshold": 0.0, "rewrite": true, "rewrite_model": "qwen3:0.6b", "rewrite_prompt_sha256": "945eb0733345db9575c97f3d9224c9583cb474ea503318a19ef86bf7b06f5243", "score_kind": "rerank"}`, golden sha256 `0c62886fd489`, git `2fe32a5-dirty`, 1258 chunks in the collection
- Metric: ndcg@5 on 45 on-corpus questions; A at k=5, t=0.00, B at k=5, t=0.00
- Bootstrap: 1000 resamples within each type, numpy default_rng(0), 95 % percentile range

## Decision

**A stays**: lower end of the range is -0.0323, not above 0.

| Mean A | Mean B | Mean delta (B - A) | 95 % range | Wins / losses / ties |
|---|---|---|---|---|
| 0.683 | 0.703 | +0.020 | [-0.0323, +0.0840] | 12 / 13 / 20 |

Rule: B replaces A only if the lower end of the range is above 0 and no type is clearly worse (mean delta <= -0.05 and more losses than wins). A tie is a per-question difference below 1e-9.

## By question type

| Type | n | Mean A | Mean B | Mean delta | W / L / T | Clearly worse |
|---|---|---|---|---|---|---|
| factual | 10 | 0.776 | 0.876 | +0.100 | 1 / 0 / 9 | no |
| paraphrased | 10 | 0.642 | 0.617 | -0.025 | 3 / 4 / 3 | no |
| cross_paper | 8 | 0.401 | 0.410 | +0.008 | 4 / 3 / 1 | no |
| table | 8 | 0.796 | 0.842 | +0.046 | 1 / 2 / 5 | no |
| multi_section | 9 | 0.775 | 0.746 | -0.030 | 3 / 4 / 2 | no |

## Off-corpus no-result rate

| A | B | Off-corpus questions |
|---|---|---|
| 0.00 | 0.00 | 20 |

Few questions in: cross_paper (n=8), table (n=8), multi_section (n=9). Per-type deltas are rough.
