# ADR-0025: Retrieval defaults after the Phase 9 evaluation

## Status

Accepted (Phase 9 Execute, T10). Applies the decision rule of ADR-0023 to the runs of
ADR-0024. Amends ADR-0022 (its "no bge query prefix yet" line) and ADR-0012 (the
provider now has a query prefix).

## Context

Phase 9 measured five retrieval settings on the golden set: the score threshold, `top_k`,
query rewriting, the embedder and the bge query prefix. All numbers below come from the
committed runs in `data/eval/runs/` (golden sha256 `4a4733e5fa76`, 50 questions, 150
labels, run at git `11318b7`) and the reports in `data/eval/reports/`.

**Read these numbers with the following limits in mind.**
- **Golden set LLM-labelled (Claude), not human-verified.** One labeller wrote and labelled
  every question (ADR-0023). A shared blind spot would not show up as an error.
- 45 on-corpus questions and only **5 off-corpus** ones. The off-corpus rate moves in
  steps of 0.20.
- 8 to 9 questions per type (factual and paraphrased: 10), so per-type numbers are rough.
- The golden questions are clean sentences written by Claude. Real user queries are
  messier, so a result about rewriting does not carry over to them with certainty.
- The rewriter is the app's own `qwen3:0.6b` through `rewrite_node`. The rewrite
  comparison judges that weak rewriter, not query rewriting in general.

Method (ADR-0023): metric nDCG@5 (a 0 to 1 score for how well the labelled sections are
ranked in the top 5; higher is better) on the 45 on-corpus questions at t = 0.6. A
paired bootstrap (1000 resamples within each type, 95 % range) of the mean difference
B minus A. B replaces A only if the lower end of the range is above 0 and no type is
clearly worse (mean difference at most -0.05 and more losses than wins).

## Decision

| Setting | Result | Decision |
|---|---|---|
| Query prefix | `prefix` vs `baseline`: nDCG@5 0.566 vs 0.525, +0.041, range [+0.012, +0.074], 11 wins / 4 losses / 30 ties, no type clearly worse | **Adopted** (B replaces A) |
| Rewrite | `no_rewrite` vs `baseline`: 0.545 vs 0.525, +0.020, range [-0.006, +0.046], 8 / 6 / 31 | **Kept** (rewrite stays on; A stays) |
| Embedder | `bge_base` vs `baseline`: 0.550 vs 0.525, +0.026, range [-0.035, +0.084], 17 / 14 / 14; clearly worse on `table` (-0.069, 1 win / 3 losses) | **Kept** `bge-small-en-v1.5` (A stays) |
| Score threshold | Curve is flat from 0.40 to 0.68; off-corpus no-result rate 0.00 in every run | **Kept** `DEFAULT_SCORE_THRESHOLD = 0.6`, no UI change |
| `top_k` | Measured only (k = 3 / 5 / 10, table below) | **Kept** 5 |

Neither the rewrite nor the embedder comparison produced a winner, so the stop condition
of the plan (re-plan if rewrite-off or bge-base wins) did not trigger. No graph, Compose
or UI change follows from this ADR.

### Query prefix (adopted)

- `SentenceTransformerProvider(model, query_prefix="")` puts `query_prefix` in front of
  the text in `embed()`, which is the query side. Documents are embedded by ingestion
  straight through `model.encode()` and are never prefixed. The `EmbeddingProvider`
  Protocol is unchanged, so ADR-0022's idea of a separate query/document split is not
  needed.
- New setting `EMBEDDING_QUERY_PREFIX` (`get_query_prefix()` in
  `app/retrieval/embedding.py`):
  - set (even to an empty string): used as is, and the empty string switches it off;
  - unset: models whose name starts with `BAAI/bge-` get the BGE instruction
    `Represent this sentence for searching relevant passages: `, other models get none.
- The eval target prepends `RetrievalConfig.prefix` itself, and the eval's own embedder
  is built with an empty provider prefix. Each named config therefore sets its prefix
  exactly once; the app default cannot leak into `baseline`, `no_rewrite` or `bge_base`
  (regression test in `tests/test_eval_runner.py`).
- Check: for all 45 on-corpus golden questions, the app's default provider with the new
  default prefix returns the same top-5 chunks, in the same order, as the stored
  `prefix` run.

Gain by type (nDCG@5, `prefix` minus `baseline`):

| Type (n) | baseline | prefix | Difference | Wins / losses / ties |
|---|---|---|---|---|
| factual (10) | 0.676 | 0.676 | +0.000 | 0 / 0 / 10 |
| paraphrased (10) | 0.452 | 0.524 | +0.073 | 5 / 1 / 4 |
| cross_paper (8) | 0.292 | 0.292 | +0.000 | 0 / 0 / 8 |
| table (8) | 0.565 | 0.626 | +0.061 | 3 / 0 / 5 |
| multi_section (9) | 0.610 | 0.677 | +0.067 | 3 / 3 / 3 |

The gain sits mostly in paraphrased and table questions. Factual and cross_paper
questions do not change at all.

### Rewrite (kept)

The rewrite did not earn its place on this set (no_rewrite is nominally higher, +0.020),
but the evidence is too weak to switch it off: the range includes 0 and spans -0.006 to
+0.046. Reasons to keep it:
- The golden questions are clean sentences, so rewriting has little to repair. Messy
  real queries may behave differently. This set cannot say.
- The comparison judged the weak 0.6B rewriter (13 of 50 rewrites equal the question).
- Switching it off changes the graph and the latency profile, which the plan reserves
  for a clear win.

**Follow-up bug, not fixed here.** `rewrite_node` leaves the literal text
`Rewritten question:` in some rewrites (6 of 50), because the prompt ends with that
phrase and the node does not strip it. This affects the rewrite comparison. Fixing it
needs a cache rebuild and all runs repeated, so it is left for a later change.

### Embedder (kept)

`bge-base-en-v1.5` is nominally +0.026 better but the range is wide and it is clearly
worse on `table` questions (8 questions, 3 losses, mean -0.069). It is also a larger
model than `bge-small`. `bge-small-en-v1.5` stays (ADR-0022).

### Score threshold (kept at 0.6, and inert)

The 0.6 cut-off removes nothing. Recall@5 and nDCG@5 are identical from t = 0.40 to
0.68, so no lower t gives a different result. The highest top-1 score of an off-corpus
question is 0.829 (0.781 to 0.831 across runs), above the cut-off, so **the app answers
off-corpus questions from unrelated chunks**. No off-corpus question drops out before
t = 0.78, and at that point 31 % of on-corpus questions return nothing. Raising t can
only lose recall.

| t | Recall@5 | nDCG@5 | On-corpus empty | Off-corpus no-result |
|---|---|---|---|---|
| 0.40 to 0.68 | 0.509 | 0.525 | 0.00 | 0.00 |
| 0.70 | 0.497 | 0.514 | 0.00 | 0.00 |
| 0.72 | 0.464 | 0.488 | 0.04 | 0.00 |
| 0.74 | 0.437 | 0.461 | 0.09 | 0.00 |
| 0.76 | 0.378 | 0.397 | 0.27 | 0.00 |
| 0.78 | 0.355 | 0.372 | 0.31 | 0.20 |
| 0.80 | 0.258 | 0.281 | 0.44 | 0.20 |

A score cut-off is the wrong tool for refusing off-corpus questions. A better signal is
outside Phase 9 (see Follow-up).

### `top_k` (kept at 5, measured only)

Recall always rises with k, so the rule cannot judge it. Baseline run, on-corpus:

| k | Recall | nDCG | MRR | Distinct papers in top-k | Table hit rate | Evidence tokens |
|---|---|---|---|---|---|---|
| 3 | 0.463 | 0.536 | 0.693 | 1.40 | 0.625 | 1465 |
| 5 | 0.509 | 0.525 | 0.703 | 1.80 | 0.625 | 2430 |
| 10 | 0.687 | 0.598 | 0.715 | 2.47 | 0.875 | 4709 |

k = 10 finds more labelled sections but uses about twice the evidence tokens, which is
above the 4000-token evidence budget of `generate_node` (it would drop the lowest-ranked
items). The effect on a 0.6B generator is not measured here (Phase 11).

### Baseline numbers and the paraphrase gap

All runs, 45 on-corpus questions, k = 5, t = 0.6:

| Run | nDCG@5 | Recall@5 | MRR@5 | Off-corpus no-result rate |
|---|---|---|---|---|
| baseline | 0.525 | 0.509 | 0.703 | 0.00 |
| no_rewrite | 0.545 | 0.514 | 0.727 | 0.00 |
| prefix | 0.566 | 0.531 | 0.767 | 0.00 |
| bge_base | 0.550 | 0.543 | 0.706 | 0.00 |

Per type, baseline nDCG@5: factual 0.676, paraphrased 0.452, cross_paper 0.292, table
0.565, multi_section 0.610. **Paraphrased questions are clearly harder than factual
ones: gap 0.224 nDCG@5** (0.195 Recall@5), and the prefix closes only about a third of
it (paraphrased 0.524). `cross_paper` is the weakest type in every run (the top 5 holds
1.8 distinct papers on average).

## Alternatives Considered

### Switch the rewrite off

Pros:
- Simpler graph, one LLM call less, nominally higher nDCG@5 (+0.020).

Cons:
- The gain is not shown (the range includes 0). The set holds only clean questions, so it
  cannot show harm or benefit for messy ones.

### Adopt `bge-base-en-v1.5`

Pros:
- Nominally higher mean and Recall@5 (0.543).

Cons:
- Clearly worse on `table`, wide range, larger model.

### Raise or lower the threshold

Pros:
- A raised threshold could turn off-corpus questions into refusals.

Cons:
- Off-corpus scores (up to 0.83) overlap with on-corpus ones. The first t that removes
  an off-corpus question (0.78) also empties 31 % of the on-corpus ones.

### Change `top_k`

Pros:
- k = 10 raises Recall@k from 0.509 to 0.687.

Cons:
- About twice the evidence tokens; generation effects are not measured. The plan made
  `top_k` measure-only.

## Consequences

Positive:
- Queries get the BGE instruction by default, with a measured gain of +0.041 nDCG@5.
- No re-ingest: the prefix touches only the query text, and documents are unchanged.

Negative:
- The prefix applies from the next API start. In Docker Compose the `api` image must be
  rebuilt and restarted to pick it up.
- Any other embedder gets no default prefix; a user of another model with its own
  query instruction must set `EMBEDDING_QUERY_PREFIX`.
- The off-corpus refusal gap stays open (see Follow-up).

## Deliberately not decided

- Whether rewriting helps on messy real queries.
- Whether a larger rewriter model changes the rewrite result.
- Any refusal signal for off-corpus questions.
- Reranking and hybrid search (Phase 10); generation quality (Phase 11).
- `top_k` against answer quality.

## Follow-up

- Bug: `rewrite_node` keeps the literal `Rewritten question:` in some outputs. After a
  fix, rebuild `data/eval/rewrites.jsonl` and re-run the rewrite comparison.
- A better refusal signal than a score cut-off for off-corpus questions (outside
  Phase 9; candidates are a reranker score, an LLM check or a margin between top hits).
  More than 5 off-corpus golden questions would be needed to judge it. The ranked options are in
  `docs/roadmap.md`, section "Phase 9 findings that feed Phase 10".
- A human check of a sample of the golden set (10-entry spot-check list in
  `docs/phases/09-retrieval-evaluation/3-execute-T8.md`).
- Messier test queries for the rewrite question.
