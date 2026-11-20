# Retrieval run: hybrid

Note: golden set LLM-labelled (Claude), not human-verified.

- Run: `hybrid`, config `{"collection": "eval_hybrid", "config_name": "hybrid", "depth": 20, "embedder": "BAAI/bge-small-en-v1.5", "fusion": "rrf", "golden_validated": true, "hybrid": true, "prefilter_threshold": 0.6, "prefix": "Represent this sentence for searching relevant passages: ", "reranker": null, "retrieval_threshold": 0.0, "rewrite": true, "rewrite_model": "qwen3:0.6b", "rewrite_prompt_sha256": "945eb0733345db9575c97f3d9224c9583cb474ea503318a19ef86bf7b06f5243", "score_kind": "rrf"}`, golden sha256 `7f236dd41db4`, git `0d961f8-dirty`, 595 chunks in the collection
- Score kind: rrf; threshold: 0.00; 45 on-corpus and 20 off-corpus questions

## By k (on-corpus questions)

Precision is the share of the top-k chunks that sit in a labelled section (how much of the context is on target). Recall vs ceiling is recall divided by the best the k slots allow, min(k, number of labels); 1.00 means as many labelled sections as k slots can hold. Both are reported, not used by the decision rule. Wrong section is the share of the top-k chunks that come from a paper with a label for the question but from a section without one (right paper, wrong section). Crowding is the mean number of distinct papers in the top-k chunks. Table hit rate is the share of `table` questions with a table chunk in a labelled section in the top-k. Evidence tokens are the mean token estimate of the top-k chunks (n/a when the run holds no token counts). The default k stays 5; this table only measures.

| k | Recall | nDCG | MRR | Precision | Recall vs ceiling | Wrong section | Crowding | Table hit rate | Evidence tokens |
|---|---|---|---|---|---|---|---|---|---|
| 3 | 0.578 | 0.702 | 0.900 | 0.637 | 0.719 | 0.222 | 1.44 | 0.750 | 1564 |
| 5 | 0.648 | 0.697 | 0.900 | 0.516 | 0.679 | 0.311 | 1.73 | 0.875 | 2486 |
| 10 | 0.773 | 0.748 | 0.907 | 0.407 | 0.773 | 0.378 | 2.47 | 1.000 | 4931 |

## Hybrid search

Hits come from one Qdrant query: a dense list (cosine at or above 0.60) and a BM25 keyword list, merged by reciprocal rank fusion (RRF). The keyword query is the same rewritten question without the dense instruction prefix.

Scores are RRF values: rank-based, neither cosine nor BM25, and small (they cannot be compared with the dense runs' cosine scores). The run is not cut at 0.6; the cosine cut already applies to the dense list inside the query. Treat the score as an order, not as a confidence.

## By question type (k=5)

The n-gram overlap is the longest run of words a question shares with its source chunks (U3 probe: a paraphrased question should have a low overlap).

| Type | n | Recall@5 | nDCG@5 | MRR@5 | Precision@5 | Recall@5 vs ceiling | Empty rate | Mean overlap | Max overlap |
|---|---|---|---|---|---|---|---|---|---|
| factual | 10 | 0.850 | 0.802 | 0.800 | 0.280 | 0.850 | 0.00 | 2.9 | 4 |
| paraphrased | 10 | 0.568 | 0.618 | 1.000 | 0.480 | 0.568 | 0.00 | 2.0 | 3 |
| cross_paper | 8 | 0.324 | 0.431 | 0.812 | 0.550 | 0.438 | 0.00 | 2.6 | 4 |
| table | 8 | 0.771 | 0.793 | 0.875 | 0.475 | 0.771 | 0.00 | 2.4 | 5 |
| multi_section | 9 | 0.689 | 0.819 | 1.000 | 0.822 | 0.746 | 0.00 | 2.6 | 3 |
| off_corpus | 20 | n/a | n/a | n/a | n/a | n/a | 0.00 | n/a | n/a |

## Thresholded results

| Measure | Value |
|---|---|
| On-corpus empty-result rate (lower is better) | 0.00 |
| Off-corpus no-result rate (higher is better) | 0.00 |
| Max off-corpus top-1 score | 1.000 |

Few questions: cross_paper (n=8), table (n=8), multi_section (n=9). Treat these rows as rough.
