# Comparison: rerank_bge_d40 against prefix

Note: golden set LLM-labelled (Claude), not human-verified.

- A (baseline side): `prefix`, config `{"collection": "papers", "config_name": "prefix", "depth": 20, "embedder": "BAAI/bge-small-en-v1.5", "golden_validated": true, "prefix": "Represent this sentence for searching relevant passages: ", "reranker": null, "retrieval_threshold": 0.0, "rewrite": true, "rewrite_model": "qwen3:0.6b", "rewrite_prompt_sha256": "945eb0733345db9575c97f3d9224c9583cb474ea503318a19ef86bf7b06f5243", "score_kind": "cosine"}`, golden sha256 `cc08395a87ca`, git `3d61760-dirty`, 595 chunks in the collection
- B (candidate side): `rerank_bge_d40`, config `{"collection": "papers", "config_name": "rerank_bge", "depth": 40, "embedder": "BAAI/bge-small-en-v1.5", "golden_validated": true, "prefilter_threshold": 0.6, "prefix": "Represent this sentence for searching relevant passages: ", "rerank_query": "rewritten", "reranker": "BAAI/bge-reranker-base", "reranker_disk_bytes": 1134410492, "reranker_load_s": 7.33, "retrieval_threshold": 0.0, "rewrite": true, "rewrite_model": "qwen3:0.6b", "rewrite_prompt_sha256": "945eb0733345db9575c97f3d9224c9583cb474ea503318a19ef86bf7b06f5243", "score_kind": "rerank"}`, golden sha256 `cc08395a87ca`, git `3d61760-dirty`, 595 chunks in the collection
- Metric: ndcg@5 on 45 on-corpus questions; A at k=5, t=0.60, B at k=5, t=0.00
- Bootstrap: 1000 resamples within each type, numpy default_rng(0), 95 % percentile range

## Decision

**B replaces A**: lower end of the range is above 0 and no type is clearly worse.

| Mean A | Mean B | Mean delta (B - A) | 95 % range | Wins / losses / ties |
|---|---|---|---|---|
| 0.564 | 0.718 | +0.154 | [+0.0679, +0.2517] | 24 / 10 / 11 |

Rule: B replaces A only if the lower end of the range is above 0 and no type is clearly worse (mean delta <= -0.05 and more losses than wins). A tie is a per-question difference below 1e-9.

## By question type

| Type | n | Mean A | Mean B | Mean delta | W / L / T | Clearly worse |
|---|---|---|---|---|---|---|
| factual | 10 | 0.676 | 0.895 | +0.219 | 3 / 0 / 7 | no |
| paraphrased | 10 | 0.515 | 0.589 | +0.075 | 5 / 4 / 1 | no |
| cross_paper | 8 | 0.306 | 0.400 | +0.094 | 6 / 2 / 0 | no |
| table | 8 | 0.614 | 0.910 | +0.297 | 4 / 2 / 2 | no |
| multi_section | 9 | 0.679 | 0.774 | +0.095 | 6 / 2 / 1 | no |

## Off-corpus no-result rate

| A | B | Off-corpus questions |
|---|---|---|
| 0.00 | 0.00 | 20 |

Few questions in: cross_paper (n=8), table (n=8), multi_section (n=9). Per-type deltas are rough.
