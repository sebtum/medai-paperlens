# Comparison: rerank_medcpt_d40 against rerank_minilm_d40

Note: golden set LLM-labelled (Claude), not human-verified.

- A (baseline side): `rerank_minilm_d40`, config `{"collection": "papers", "config_name": "rerank_minilm", "depth": 40, "embedder": "BAAI/bge-small-en-v1.5", "golden_validated": true, "prefilter_threshold": 0.6, "prefix": "Represent this sentence for searching relevant passages: ", "rerank_query": "rewritten", "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2", "reranker_disk_bytes": 890152505, "reranker_load_s": 4.1, "retrieval_threshold": 0.0, "rewrite": true, "rewrite_model": "qwen3:0.6b", "rewrite_prompt_sha256": "945eb0733345db9575c97f3d9224c9583cb474ea503318a19ef86bf7b06f5243", "score_kind": "rerank"}`, golden sha256 `cc08395a87ca`, git `3d61760-dirty`, 595 chunks in the collection
- B (candidate side): `rerank_medcpt_d40`, config `{"collection": "papers", "config_name": "rerank_medcpt", "depth": 40, "embedder": "BAAI/bge-small-en-v1.5", "golden_validated": true, "prefilter_threshold": 0.6, "prefix": "Represent this sentence for searching relevant passages: ", "rerank_query": "rewritten", "reranker": "ncbi/MedCPT-Cross-Encoder", "reranker_disk_bytes": 876888660, "reranker_load_s": 2.56, "retrieval_threshold": 0.0, "rewrite": true, "rewrite_model": "qwen3:0.6b", "rewrite_prompt_sha256": "945eb0733345db9575c97f3d9224c9583cb474ea503318a19ef86bf7b06f5243", "score_kind": "rerank"}`, golden sha256 `cc08395a87ca`, git `3d61760-dirty`, 595 chunks in the collection
- Metric: ndcg@5 on 45 on-corpus questions; A at k=5, t=0.00, B at k=5, t=0.00
- Bootstrap: 1000 resamples within each type, numpy default_rng(0), 95 % percentile range

## Decision

**A stays**: lower end of the range is -0.0936, not above 0; clearly worse on: factual, cross_paper.

| Mean A | Mean B | Mean delta (B - A) | 95 % range | Wins / losses / ties |
|---|---|---|---|---|
| 0.725 | 0.688 | -0.037 | [-0.0936, +0.0178] | 12 / 19 / 14 |

Rule: B replaces A only if the lower end of the range is above 0 and no type is clearly worse (mean delta <= -0.05 and more losses than wins). A tie is a per-question difference below 1e-9.

## By question type

| Type | n | Mean A | Mean B | Mean delta | W / L / T | Clearly worse |
|---|---|---|---|---|---|---|
| factual | 10 | 0.876 | 0.802 | -0.074 | 0 / 2 / 8 | yes |
| paraphrased | 10 | 0.639 | 0.633 | -0.006 | 4 / 5 / 1 | no |
| cross_paper | 8 | 0.468 | 0.401 | -0.068 | 2 / 5 / 1 | yes |
| table | 8 | 0.858 | 0.841 | -0.017 | 1 / 3 / 4 | no |
| multi_section | 9 | 0.761 | 0.741 | -0.021 | 5 / 4 / 0 | no |

## Off-corpus no-result rate

| A | B | Off-corpus questions |
|---|---|---|
| 0.00 | 0.00 | 20 |

Few questions in: cross_paper (n=8), table (n=8), multi_section (n=9). Per-type deltas are rough.
