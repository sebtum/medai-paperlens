# Refusal curve: rerank_medcpt

Note: golden set LLM-labelled (Claude), not human-verified.

- Run: `rerank_medcpt`, config `{"collection": "papers", "config_name": "rerank_medcpt", "depth": 20, "embedder": "BAAI/bge-small-en-v1.5", "golden_validated": true, "prefilter_threshold": 0.6, "prefix": "Represent this sentence for searching relevant passages: ", "rerank_query": "rewritten", "reranker": "ncbi/MedCPT-Cross-Encoder", "reranker_disk_bytes": 876888660, "reranker_load_s": 2.85, "retrieval_threshold": 0.0, "rewrite": true, "rewrite_model": "qwen3:0.6b", "rewrite_prompt_sha256": "945eb0733345db9575c97f3d9224c9583cb474ea503318a19ef86bf7b06f5243", "score_kind": "rerank"}`, golden sha256 `cc08395a87ca`, git `3d61760-dirty`, 595 chunks in the collection
- Score kind: rerank; 45 on-corpus and 20 off-corpus questions

A question is refused at t when its top-1 score is below t, or when it has no hit at all (a reranked run drops hits below the cosine pre-filter first). Only the top-1 score is used. Off-corpus refused is good. On-corpus lost is the cost: questions the papers do cover that would be refused too.

| t | Off-corpus refused | Off-corpus refused rate | On-corpus lost | On-corpus lost rate |
|---|---|---|---|---|
| 0.00 | 0 / 20 | 0.00 | 0 / 45 | 0.00 |
| 0.01 | 2 / 20 | 0.10 | 0 / 45 | 0.00 |
| 0.02 | 2 / 20 | 0.10 | 0 / 45 | 0.00 |
| 0.05 | 4 / 20 | 0.20 | 0 / 45 | 0.00 |
| 0.10 | 4 / 20 | 0.20 | 0 / 45 | 0.00 |
| 0.15 | 4 / 20 | 0.20 | 0 / 45 | 0.00 |
| 0.20 | 4 / 20 | 0.20 | 0 / 45 | 0.00 |
| 0.25 | 4 / 20 | 0.20 | 0 / 45 | 0.00 |
| 0.30 | 4 / 20 | 0.20 | 0 / 45 | 0.00 |
| 0.35 | 4 / 20 | 0.20 | 0 / 45 | 0.00 |
| 0.40 | 4 / 20 | 0.20 | 0 / 45 | 0.00 |
| 0.45 | 4 / 20 | 0.20 | 0 / 45 | 0.00 |
| 0.50 | 4 / 20 | 0.20 | 0 / 45 | 0.00 |
| 0.55 | 4 / 20 | 0.20 | 0 / 45 | 0.00 |
| 0.60 | 4 / 20 | 0.20 | 0 / 45 | 0.00 |
| 0.65 | 4 / 20 | 0.20 | 0 / 45 | 0.00 |
| 0.70 | 4 / 20 | 0.20 | 0 / 45 | 0.00 |
| 0.75 | 4 / 20 | 0.20 | 0 / 45 | 0.00 |
| 0.80 | 4 / 20 | 0.20 | 0 / 45 | 0.00 |
| 0.85 | 4 / 20 | 0.20 | 0 / 45 | 0.00 |
| 0.90 | 4 / 20 | 0.20 | 0 / 45 | 0.00 |
| 0.95 | 5 / 20 | 0.25 | 0 / 45 | 0.00 |
| 0.98 | 6 / 20 | 0.30 | 0 / 45 | 0.00 |
| 0.99 | 6 / 20 | 0.30 | 0 / 45 | 0.00 |
| 1.00 | 20 / 20 | 1.00 | 45 / 45 | 1.00 |

The cut-off is chosen by the owner after seeing this curve. A cut-off picked on the same questions it is scored on looks better than it will be on new ones.
