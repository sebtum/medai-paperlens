# Refusal curve: rerank_minilm_d40

Note: golden set LLM-labelled (Claude), not human-verified.

- Run: `rerank_minilm_d40`, config `{"collection": "papers", "config_name": "rerank_minilm", "depth": 40, "embedder": "BAAI/bge-small-en-v1.5", "golden_validated": true, "prefilter_threshold": 0.6, "prefix": "Represent this sentence for searching relevant passages: ", "rerank_query": "rewritten", "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2", "reranker_disk_bytes": 890152505, "reranker_load_s": 4.1, "retrieval_threshold": 0.0, "rewrite": true, "rewrite_model": "qwen3:0.6b", "rewrite_prompt_sha256": "945eb0733345db9575c97f3d9224c9583cb474ea503318a19ef86bf7b06f5243", "score_kind": "rerank"}`, golden sha256 `cc08395a87ca`, git `3d61760-dirty`, 595 chunks in the collection
- Score kind: rerank; 45 on-corpus and 20 off-corpus questions

A question is refused at t when its top-1 score is below t, or when it has no hit at all (a reranked run drops hits below the cosine pre-filter first). Only the top-1 score is used. Off-corpus refused is good. On-corpus lost is the cost: questions the papers do cover that would be refused too.

| t | Off-corpus refused | Off-corpus refused rate | On-corpus lost | On-corpus lost rate |
|---|---|---|---|---|
| 0.00 | 0 / 20 | 0.00 | 0 / 45 | 0.00 |
| 0.01 | 1 / 20 | 0.05 | 0 / 45 | 0.00 |
| 0.02 | 2 / 20 | 0.10 | 0 / 45 | 0.00 |
| 0.05 | 3 / 20 | 0.15 | 0 / 45 | 0.00 |
| 0.10 | 5 / 20 | 0.25 | 1 / 45 | 0.02 |
| 0.15 | 5 / 20 | 0.25 | 1 / 45 | 0.02 |
| 0.20 | 5 / 20 | 0.25 | 3 / 45 | 0.07 |
| 0.25 | 5 / 20 | 0.25 | 3 / 45 | 0.07 |
| 0.30 | 5 / 20 | 0.25 | 4 / 45 | 0.09 |
| 0.35 | 5 / 20 | 0.25 | 4 / 45 | 0.09 |
| 0.40 | 6 / 20 | 0.30 | 4 / 45 | 0.09 |
| 0.45 | 6 / 20 | 0.30 | 4 / 45 | 0.09 |
| 0.50 | 7 / 20 | 0.35 | 4 / 45 | 0.09 |
| 0.55 | 8 / 20 | 0.40 | 5 / 45 | 0.11 |
| 0.60 | 8 / 20 | 0.40 | 5 / 45 | 0.11 |
| 0.65 | 9 / 20 | 0.45 | 5 / 45 | 0.11 |
| 0.70 | 9 / 20 | 0.45 | 5 / 45 | 0.11 |
| 0.75 | 11 / 20 | 0.55 | 8 / 45 | 0.18 |
| 0.80 | 12 / 20 | 0.60 | 9 / 45 | 0.20 |
| 0.85 | 14 / 20 | 0.70 | 12 / 45 | 0.27 |
| 0.90 | 15 / 20 | 0.75 | 15 / 45 | 0.33 |
| 0.95 | 17 / 20 | 0.85 | 19 / 45 | 0.42 |
| 0.98 | 19 / 20 | 0.95 | 31 / 45 | 0.69 |
| 0.99 | 19 / 20 | 0.95 | 36 / 45 | 0.80 |
| 1.00 | 20 / 20 | 1.00 | 45 / 45 | 1.00 |

The cut-off is chosen by the owner after seeing this curve. A cut-off picked on the same questions it is scored on looks better than it will be on new ones.
