# Refusal curve: rerank_bge

Note: golden set LLM-labelled (Claude), not human-verified.

- Run: `rerank_bge`, config `{"collection": "papers", "config_name": "rerank_bge", "depth": 20, "embedder": "BAAI/bge-small-en-v1.5", "golden_validated": true, "prefilter_threshold": 0.6, "prefix": "Represent this sentence for searching relevant passages: ", "rerank_query": "rewritten", "reranker": "BAAI/bge-reranker-base", "reranker_disk_bytes": 1134410492, "reranker_load_s": 5.28, "retrieval_threshold": 0.0, "rewrite": true, "rewrite_model": "qwen3:0.6b", "rewrite_prompt_sha256": "945eb0733345db9575c97f3d9224c9583cb474ea503318a19ef86bf7b06f5243", "score_kind": "rerank"}`, golden sha256 `cc08395a87ca`, git `3d61760-dirty`, 595 chunks in the collection
- Score kind: rerank; 45 on-corpus and 20 off-corpus questions

A question is refused at t when its top-1 score is below t, or when it has no hit at all (a reranked run drops hits below the cosine pre-filter first). Only the top-1 score is used. Off-corpus refused is good. On-corpus lost is the cost: questions the papers do cover that would be refused too.

| t | Off-corpus refused | Off-corpus refused rate | On-corpus lost | On-corpus lost rate |
|---|---|---|---|---|
| 0.00 | 0 / 20 | 0.00 | 0 / 45 | 0.00 |
| 0.01 | 2 / 20 | 0.10 | 0 / 45 | 0.00 |
| 0.02 | 3 / 20 | 0.15 | 1 / 45 | 0.02 |
| 0.05 | 6 / 20 | 0.30 | 2 / 45 | 0.04 |
| 0.10 | 8 / 20 | 0.40 | 5 / 45 | 0.11 |
| 0.15 | 9 / 20 | 0.45 | 5 / 45 | 0.11 |
| 0.20 | 12 / 20 | 0.60 | 6 / 45 | 0.13 |
| 0.25 | 13 / 20 | 0.65 | 6 / 45 | 0.13 |
| 0.30 | 13 / 20 | 0.65 | 7 / 45 | 0.16 |
| 0.35 | 14 / 20 | 0.70 | 9 / 45 | 0.20 |
| 0.40 | 15 / 20 | 0.75 | 11 / 45 | 0.24 |
| 0.45 | 15 / 20 | 0.75 | 11 / 45 | 0.24 |
| 0.50 | 16 / 20 | 0.80 | 11 / 45 | 0.24 |
| 0.55 | 16 / 20 | 0.80 | 11 / 45 | 0.24 |
| 0.60 | 17 / 20 | 0.85 | 12 / 45 | 0.27 |
| 0.65 | 17 / 20 | 0.85 | 12 / 45 | 0.27 |
| 0.70 | 17 / 20 | 0.85 | 15 / 45 | 0.33 |
| 0.75 | 18 / 20 | 0.90 | 15 / 45 | 0.33 |
| 0.80 | 18 / 20 | 0.90 | 15 / 45 | 0.33 |
| 0.85 | 18 / 20 | 0.90 | 16 / 45 | 0.36 |
| 0.90 | 19 / 20 | 0.95 | 18 / 45 | 0.40 |
| 0.95 | 19 / 20 | 0.95 | 21 / 45 | 0.47 |
| 0.98 | 19 / 20 | 0.95 | 28 / 45 | 0.62 |
| 0.99 | 20 / 20 | 1.00 | 33 / 45 | 0.73 |
| 1.00 | 20 / 20 | 1.00 | 45 / 45 | 1.00 |

The cut-off is chosen by the owner after seeing this curve. A cut-off picked on the same questions it is scored on looks better than it will be on new ones.
