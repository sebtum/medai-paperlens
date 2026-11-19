# T10 spike: BM25 inside Qdrant

## What

The question: can Qdrant itself compute BM25 keyword scores (no `fastembed`, no new dependency), and can one hybrid query (dense plus BM25, merged by RRF) run against it?

I tested this on a scratch collection `spike_bm25_scratch` with 28 real chunks copied (read-only) from `papers`. The scratch collection is deleted. `papers` and `eval_bge-base-en-v1.5` were not touched (checked afterwards: both still listed).

## Result

Yes, it works. No new dependency is needed. Stop-and-ask about `fastembed` is not triggered.

- Details: server Qdrant 1.18.2, client `qdrant-client` 1.18.0, `fastembed` not installed. The server ran the `qdrant/bm25` model itself: the client only sent `Document(text, model)`. The bad-model error confirms the split: an unknown model name returns "qdrant/nope is not found among supported models. Check if `cloud_inference` is set to True or `fastembed` is installed (for local inference)?", so the known `qdrant/bm25` name is served by the server.
- Details: create, upsert, BM25-only query, hybrid query all succeeded on the first run. The hybrid query returned 20 results in about 21 ms (28 points, local).

### Evidence: BM25-only queries (28 chunks, top hits as `paper, chunk, score`)

| Query | Top hit | Reading |
|---|---|---|
| "disjoint sets of N + N tokens coordinates" (words from `maira_2` chunk 31) | `maira_2` 10 (16.2), then `maira_2` 31 (14.7) | Right paper, target is rank 2. Chunk 10 shares the same coordinate-token wording, so this is plausible. |
| "ablation" | `maira_2` 31 (4.9), only hit | Exact term found in the one chunk that has it. |
| "Appendices D.5 and D.6" | `maira_2` 31 (8.7) | Exact reference found, this chunk says "Appendices D.5 and D.6". |
| "MAIRA-2" | five `maira_2` chunks | Model name finds its own paper. |
| "the" | no hits | Stopword only: empty result, no error. |

### Evidence: hybrid query

Dense prefetch (the vector of chunk 31, limit 20) plus BM25 prefetch ("disjoint sets of N + N tokens coordinates", limit 20), fused with RRF. Top 3: `maira_2` 10 (0.833), `maira_2` 31 (0.833), `maira_2` 5 (0.5). 20 results back. Fused scores are rank-based (RRF), not cosine or BM25 values, so they cannot be compared with either.

### Observed text behaviour (default `qdrant/bm25` options)

- Details: lowercase: yes. "THE ABLATION" gives the same hits as "ablation".
- Details: stemming: yes, English. "ablation" and "ablations" give identical hits and scores; "coordinate" and "coordinates" the same. In a probe document, "Running RUNS runner ran" became 3 distinct terms: running and runs share one stem (weight 1.901 for two occurrences), runner and ran stay separate.
- Details: stopwords: yes, English. "the and of a model" became 1 term (`model`). A query of only stopwords gives an empty list.
- Details: tokenizer: splits on punctuation. "MAIRA-2", "maira-2" and "MAIRA 2" give identical results (the same terms: `maira` and, probably, `2`). A probe "GPT-4V Table 3 CheXpert-v1.0 x12 ⟨x12⟩" gave 8 terms, so a number like "3" or "4V" is kept as a term, not dropped. Exact table numbers therefore match, but "Table 3" will also match any chunk with the token "3" (weak on its own).
- Details: accents: "Café naïve résumé" gave 3 terms. I did not check whether `é` is folded to `e` (`ascii_folding` option exists).
- Details: IDF: scores come from document frequency inside the collection (modifier `IDF`, computed by the server at query time). Scores shift when more chunks are added, so the 28-chunk values above are not what 595 chunks will give.
- Details: sparse vector content: one chunk of 78 non-zero terms came back with a max weight of 1.81. Stored values are the term-frequency part; IDF is applied by the server.
- Details: options must match between index and query. A query `Document` with `options={"language": "none"}` returned nothing against chunks indexed with the defaults (the stemmed term did not match the unstemmed one). `lowercase=False`, `stemmer=None` and `k/b/avg_len` options were accepted without error and gave the same hit here, so I did not learn their effect. Recommendation: use the defaults everywhere (index and query) and do not pass `options`.

## Working code shape

Tested exactly like this (Python, `qdrant-client` 1.18.0, `from qdrant_client import QdrantClient, models`):

```python
# 1. Collection: NAMED dense vector + sparse bm25 with IDF
client.create_collection(
    "spike_bm25_scratch",
    vectors_config={"dense": models.VectorParams(size=384, distance=models.Distance.COSINE)},
    sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)},
)

# 2. Upsert: dense vector plus a Document that the server turns into the sparse vector
client.upsert("spike_bm25_scratch", points=[
    models.PointStruct(
        id=point_id,
        payload=payload,
        vector={
            "dense": dense_vector,                                   # list[float], 384
            "bm25": models.Document(text=chunk_text, model="qdrant/bm25"),
        },
    ),
], wait=True)

# 3. BM25 only
client.query_points(
    "spike_bm25_scratch",
    query=models.Document(text=question, model="qdrant/bm25"),
    using="bm25",
    limit=20,
)

# 4. Hybrid: two prefetches, fused by RRF
client.query_points(
    "spike_bm25_scratch",
    prefetch=[
        models.Prefetch(query=dense_query_vector, using="dense", limit=20, score_threshold=0.6),
        models.Prefetch(query=models.Document(text=question, model="qdrant/bm25"), using="bm25", limit=20),
    ],
    query=models.FusionQuery(fusion=models.Fusion.RRF),
    limit=20,
    with_payload=True,
)
```

- Details: the dense `score_threshold=0.6` on a `Prefetch` is accepted (tested in a second run, 20 results). It filters only the dense list; the BM25 list has no cosine threshold, so a chunk can enter through BM25 alone.
- Details: the real `papers` collection has an UNNAMED dense vector (`ingestion.py`). `eval_hybrid` must be created with the named vector `dense`, and every dense search against it must pass `using="dense"`. `search_hits` as written cannot be reused unchanged for that collection.
- Details: in the spike the BM25 text was the payload `text`. For T10 use `embed_text`, as the plan says, so keywords and dense vector see the same text.
- Details: on Windows, printing chunk text with `⟨` fails on the default cp1252 console. Set `PYTHONIOENCODING=utf-8` when running spike or eval scripts that print payload text.

## Not checked

- Scale: only 28 chunks. Not checked: 595 chunks, upsert time with server-side encoding, query latency at full size, IDF behaviour at that size.
- Whether the BM25 hits beat the dense hits on the eval questions. That is the T10 run, not the spike.
- The effect of the `Bm25Config` options (k, b, avg_len, stemmer, lowercase, ascii_folding, language, stopwords). Only that they are accepted or, for `language="none"`, that mismatched options break matching.
- Accent folding, and how `2` / single letters are treated in the tokenizer.
- Behaviour of the Docker image `qdrant/qdrant:latest` on other machines: the server here is 1.18.2. Older servers may not know `qdrant/bm25`. `docker-compose.yml` uses `latest`, so pin a version before adopting.
- Dense score_threshold effect on fused ranking (only checked that the call runs).
