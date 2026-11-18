# ADR-0026: Reranker (model, depth, query text, placement, refusal option)

## Status

Proposed (Phase 10 Execute, T7). The measurements below are final. The Decision section
is **not filled yet: the owner decides** (see "OWNER DECISION PENDING"). When the owner
decides, this ADR becomes Accepted (or Rejected if no reranker is adopted) and the ADR
index gets its row. Applies the decision rule of ADR-0023 and the runner of ADR-0024.
Follows ADR-0025, which left the off-corpus refusal gap open and named a reranker score
as one candidate signal.

## Context

Phase 9 showed that dense search usually finds the answer but ranks it too low: about
four in five labelled sections are in the top 20, only about half in the top 5. It also
showed that no off-corpus question is ever refused, because the cosine scores of
off-corpus questions (up to 0.84) overlap with those of on-corpus questions (ADR-0025).

A **reranker** (a cross-encoder) reads the question and one passage together and gives a
0 to 1 score. The retrieval step keeps its top 20 (or 40) dense hits, the reranker
re-scores them, and the best 5 go to generation. The rerank step already exists in the
app as its own workflow step (T8) and is **off by default** until this ADR decides.

What was measured (T5, T6, T7). All numbers come from the committed runs in
`data/eval/runs/` and the reports in `data/eval/reports/`: golden set v2, sha256
`cc08395a87ca`, 65 questions (45 on-corpus, 20 off-corpus), 159 labels, chunk count 595,
code at git `3d61760-dirty`. Metric: nDCG@5 (a 0 to 1 score for how well the labelled
sections are ranked in the top 5) on the 45 on-corpus questions.

- Three models: `cross-encoder/ms-marco-MiniLM-L-6-v2` (91 MB of weights), `BAAI/bge-reranker-base`
  (1.1 GB), `ncbi/MedCPT-Cross-Encoder` (438 MB). Scores go through a sigmoid so all
  three are 0 to 1. Reranker text is `rerank_passage(title, section, text)`, `max_length`
  512, same function in the app and the eval.
- Three variants per model: rewritten question at depth 20, original question at depth
  20, rewritten question at depth 40. Depth is how many dense hits the reranker sees.
- Reference for every comparison: `prefix`, the dense default of ADR-0025 (nDCG@5 0.564).
- Rule (ADR-0023): stratified paired bootstrap (1000 resamples within each question
  type, 95 % range) of the mean nDCG@5 difference B minus A. B replaces A only if the
  lower end of the range is above 0 and no type is clearly worse (type mean difference
  at most -0.05 and more losses than wins). Reranked runs are cut at 0 (no cut), the
  dense run at 0.6 cosine, as `compare` does by default.

**Read the numbers with these limits in mind.**

- **Golden set LLM-labelled (Claude), not human-verified.** The same author wrote the
  questions and the labels (ADR-0023). A shared blind spot would not show up as an
  error.
- Only **45 on-corpus questions** (5 to 10 per type) and **20 off-corpus questions**
  (5 from Phase 9, 15 near-miss ones written in T4 before any reranker result was seen).
  The off-corpus rate moves in steps of 0.05. Per-type rows are rough.
- **Label bias in favour of reranking, small.** The Phase 9 labels came from dense runs
  (top 10 of the dense setups), which favours dense. The 9 labels added in T6 came from
  the top 5 of the nine reranked runs, so they can only help reranked runs (and dense
  runs only where a dense run also found those sections). Measured here: with the 150
  labels from before the pass, the gain over `prefix` is 0.122 / 0.139 / 0.070 (MiniLM,
  bge, MedCPT, depth 20) against 0.120 / 0.140 / 0.075 with all 159 labels, so the pass
  moved the gains by -0.002 to +0.005; at depth 40 by +0.002 to +0.007. The dense runs
  moved by at most +0.002. The effect is small but points the same way for the
  MedCPT and depth-40 rows. MedCPT at depth 20 has the thinnest margin (lower end of
  the range +0.009), so a bias of this size could matter for that row. No other verdict
  is close enough to flip.
- Timing is one CPU-only machine, one timed run per setup, after one warm-up call,
  reranker only (no search, no rewrite). Machine noise is not measured.
- Every question reached the reranker with hits: no question was empty after the 0.6
  cosine pre-filter in any of the nine runs (0 of 65), so the pre-filter caveat of the
  refusal curve (questions with no hit count as refused at every cut-off) did not apply.
- The golden questions are clean sentences. Real questions are messier. The rewriter is
  the app's `qwen3:0.6b`.

## Results

### Each reranker against the dense default (`prefix`, nDCG@5 0.564)

| Run | Depth | Query | nDCG@5 | Gain over `prefix`, 95 % range | Wins / losses / ties | Verdict by the rule |
|---|---|---|---|---|---|---|
| `rerank_minilm` | 20 | rewritten | 0.684 | +0.120 [+0.046, +0.199] | 23 / 7 / 15 | B replaces A |
| `rerank_minilm_orig` | 20 | original | 0.699 | +0.135 [+0.059, +0.216] | 21 / 7 / 17 | B replaces A |
| `rerank_minilm_d40` | 40 | rewritten | 0.725 | +0.161 [+0.078, +0.252] | 25 / 7 / 13 | B replaces A |
| `rerank_bge` | 20 | rewritten | 0.704 | +0.140 [+0.056, +0.225] | 22 / 9 / 14 | B replaces A |
| `rerank_bge_orig` | 20 | original | 0.685 | +0.121 [+0.048, +0.203] | 18 / 10 / 17 | B replaces A |
| `rerank_bge_d40` | 40 | rewritten | 0.718 | +0.154 [+0.068, +0.252] | 24 / 10 / 11 | B replaces A |
| `rerank_medcpt` | 20 | rewritten | 0.639 | +0.075 [+0.009, +0.155] | 17 / 12 / 16 | B replaces A |
| `rerank_medcpt_orig` | 20 | original | 0.645 | +0.081 [+0.020, +0.151] | 18 / 10 / 17 | B replaces A |
| `rerank_medcpt_d40` | 40 | rewritten | 0.688 | +0.124 [+0.040, +0.220] | 20 / 12 / 13 | B replaces A |

All nine pass the rule against `prefix`. No question type is clearly worse in any of them.
The gain sits in every type (factual +0.03 to +0.22, paraphrased +0.04 to +0.16,
cross_paper +0.07 to +0.16, table +0.10 to +0.30, multi_section +0.04 to +0.13).

### Variants (does the choice inside one model matter?)

A is the depth-20 rewritten run of the same model, B is the variant.

| Model | B | Difference B minus A, 95 % range | W / L / T | Verdict |
|---|---|---|---|---|
| MiniLM | original query | +0.015 [-0.005, +0.036] | 9 / 4 / 32 | A stays (range includes 0) |
| MiniLM | depth 40 | +0.040 [-0.003, +0.099] | 6 / 3 / 36 | A stays (range includes 0, by 0.003) |
| bge | original query | -0.019 [-0.042, +0.004] | 7 / 7 / 31 | A stays; paraphrased clearly worse (-0.069, 3 wins / 4 losses) |
| bge | depth 40 | +0.014 [-0.021, +0.062] | 5 / 8 / 32 | A stays (range includes 0) |
| MedCPT | original query | +0.006 [-0.025, +0.039] | 6 / 8 / 31 | A stays (range includes 0) |
| MedCPT | depth 40 | +0.049 [-0.0005, +0.116] | 4 / 3 / 38 | A stays (range includes 0, by 0.0005) |

Neither the query text nor the depth gives a difference that the rule can call real for
any model. Depth 40 has a positive mean for all three models (+0.014 to +0.049); its
range includes 0 for all three, and only just for MiniLM (-0.003) and MedCPT (-0.0005). Recall among the hits the reranker can see is
0.828 at depth 20 and 0.914 at depth 40, the same for all three models.

### The models against each other

A is the first run, B the second; a negative difference means B scores lower.

| A | B | Difference B minus A, 95 % range | W / L / T | Verdict |
|---|---|---|---|---|
| `rerank_minilm` | `rerank_bge` | +0.019 [-0.025, +0.076] | 13 / 13 / 19 | A stays (no real difference) |
| `rerank_minilm` | `rerank_medcpt` | -0.045 [-0.090, -0.004] | 9 / 17 / 19 | A stays; B lower, clearly worse on factual (-0.074) and table (-0.080) |
| `rerank_bge` | `rerank_medcpt` | -0.064 [-0.123, -0.014] | 12 / 18 / 15 | A stays; B lower, clearly worse on factual (-0.174) and table (-0.075) |
| `rerank_minilm_d40` | `rerank_bge_d40` | -0.007 [-0.046, +0.033] | 12 / 16 / 17 | A stays (no real difference) |
| `rerank_minilm_d40` | `rerank_medcpt_d40` | -0.037 [-0.094, +0.018] | 12 / 19 / 14 | A stays; B clearly worse on factual (-0.074), cross_paper (-0.068) |
| `rerank_bge_d40` | `rerank_medcpt_d40` | -0.030 [-0.077, +0.013] | 13 / 16 / 16 | A stays; B clearly worse on factual (-0.093), table (-0.069) |
| `rerank_bge` | `rerank_minilm_d40` | +0.021 [-0.022, +0.065] | n/a | A stays (no real difference) |

At depth 20, MiniLM and bge cannot be told apart, and MedCPT is lower than both with a
range fully below 0. At depth 40 the three are closer; the ranges include 0 but MedCPT is
flagged as clearly worse in some types.

### Time, size and gain per second

Time is the reranker call per question after one warm-up (n = 65 questions), on one
CPU-only machine. Gain is nDCG@5 over `prefix` divided by the median time. The model
size is the real weight file (the recorded cache sizes of 890 MB for MiniLM and 877 MB
for MedCPT include duplicate folders and a second revision; bge is 1.1 GB either way).

| Model | Weights | Load | Depth | Query | Median ms | 90th pct ms | nDCG@5 gain per second |
|---|---|---|---|---|---|---|---|
| MiniLM | 91 MB | 4.1 s | 20 | rewritten | 1656 | 1713 | 0.073 |
| MiniLM | 91 MB | 4.1 s | 20 | original | 1615 | 1704 | 0.084 |
| MiniLM | 91 MB | 4.1 s | 40 | rewritten | 3117 | 3438 | 0.052 |
| bge | 1112 MB | 5.3 s | 20 | rewritten | 9795 | 11787 | 0.014 |
| bge | 1112 MB | 5.1 s | 20 | original | 10572 | 14240 | 0.011 |
| bge | 1112 MB | 7.3 s | 40 | rewritten | 23194 | 27223 | 0.007 |
| MedCPT | 438 MB | 2.9 s | 20 | rewritten | 8884 | 13055 | 0.008 |
| MedCPT | 438 MB | 2.3 s | 20 | original | 9755 | 10160 | 0.008 |
| MedCPT | 438 MB | 2.6 s | 40 | rewritten | 17862 | 18859 | 0.007 |

For one request: the earlier benchmark (`docs/benchmarks.md`, `qwen3:0.6b`, no thinking,
Sept 2026) gives 2.2 s rewrite and 4.8 s generate, about 7 s without search or
embedding. No end-to-end time with the real evidence prompt was found in the docs, so 7
s is a lower bound for today's request. Added median rerank time on top of 7 s:
MiniLM +23 to +45 %, bge +140 to +331 %, MedCPT +127 to +255 %.

### Refusal on the top-1 score (gate curve)

A question is refused at cut-off t when its top-1 reranker score is below t. Off-corpus
refused is the gain. On-corpus lost (questions the papers do cover that would be refused
too) is the cost. Full curves: `data/eval/reports/gate-rerank_{minilm,bge,medcpt,minilm_d40}.md`.

| Run | t | Off-corpus refused (of 20) | On-corpus lost (of 45) |
|---|---|---|---|
| `rerank_minilm` | 0.10 | 5 | 2 |
| `rerank_minilm` | 0.50 | 9 | 5 |
| `rerank_minilm` | 0.80 | 14 | 9 |
| `rerank_minilm` | 0.90 | 16 | 16 |
| `rerank_bge` | 0.10 | 8 | 5 |
| `rerank_bge` | 0.30 | 13 | 7 |
| `rerank_bge` | 0.50 | 16 | 11 |
| `rerank_bge` | 0.70 | 17 | 15 |
| `rerank_medcpt` | 0.05 to 0.90 | 4 | 0 |
| `rerank_medcpt` | 0.9995 | 9 | 1 |
| `rerank_medcpt` | 0.99995 | 12 | 3 |
| `rerank_medcpt` | 0.99999 | 12 | 8 |
| `rerank_minilm_d40` | 0.10 | 5 | 1 |
| `rerank_minilm_d40` | 0.50 | 7 | 4 |
| `rerank_minilm_d40` | 0.80 | 12 | 9 |

- No model separates the two groups. The probability that an off-corpus question scores
  below an on-corpus question (area under the curve) is 0.80 (MiniLM), 0.84 (bge),
  0.87 (MedCPT), 0.80 (MiniLM depth 40). Perfect separation would be 1.00.
- The five Phase 9 off-corpus questions are easy: at t = 0.9 MiniLM refuses all 5, at
  t = 0.5 bge refuses all 5. The 15 near-miss questions are the hard part: at the same
  cut-offs 11 of 15 are refused, and on-corpus losses are 16 of 45 (MiniLM) and 11 of 45
  (bge).
- MedCPT: its score is squeezed against 1 (all on-corpus top-1 scores are 0.9995 or
  higher; 14 of 20 off-corpus ones are above 0.99). Only 4 off-corpus questions fall
  below 0.04. A cut-off exists but only in the 4th to 5th decimal (t between 0.9995 and
  0.99995, which is a logit of about 7.6 to 9.9). That is brittle: tiny score changes
  (model revision, float precision, other chunk text) would move it.
- Uncertainty of the best points with 95 % Wilson ranges (off-corpus out of 20, on-corpus
  out of 45): MiniLM t = 0.08: 5/20 [0.11, 0.47] refused, 1/45 lost; bge t = 0.34: 14/20
  [0.48, 0.85] refused, 8/45 lost [0.09, 0.31]; MedCPT t = 0.9999: 11/20 [0.34,
  0.74] refused, 2/45 lost [0.01, 0.15]. A 0.05 step in the range is a few questions.
- **These cut-offs are optimistic.** The cut-off would be chosen on the same 20
  off-corpus questions it is judged on. A repeated half split (cut-off chosen on half of
  the questions with at most 10 % on-corpus loss, scored on the other half) gave a
  held-out off-corpus refusal of 0.35 (MiniLM), 0.40 (bge), 0.52 (MedCPT), 0.34 (MiniLM
  depth 40) against 0.44 to 0.61 on the half it was chosen on, with 0.04 to 0.11
  on-corpus lost.

### Fixed items

- Reranked runs are cut at 0 (no cut) in `compare`; the 0.6 cosine pre-filter still
  applies before the reranker (inert in all nine runs: 0 of 65 questions empty).
- In all runs the off-corpus no-result rate in the normal run report is 0.00 (nothing is
  refused without a cut-off).
- Wrong-section share at k = 5 (right paper, wrong section): `prefix` 0.351, MiniLM
  0.284 / 0.298 / 0.280, bge 0.347 / 0.351 / 0.329, MedCPT 0.320 / 0.324 / 0.316
  (depth 20 rewritten / original / depth 40 rewritten). The T11 trigger (25 %) is
  evaluated on the adopted setup, not here.

## Decision

**OWNER DECISION PENDING.** Nothing below is decided. The numbers above are the input.
"No reranker" is a valid outcome.

| Field | Options | Owner decision |
|---|---|---|
| Time limit (reranker time per request, on the owner's CPU) | seconds, set first | _to fill_ |
| Adopt a reranker at all? | yes / no (no: `RERANKER_MODEL` stays off, only the numbers are recorded) | _to fill_ |
| Model | MiniLM / bge / MedCPT (the rule's best result within the time limit) | _to fill_ |
| Depth (hits the reranker sees) | 20 or 40 (retrieve fetches this many) | _to fill_ |
| Query text the reranker reads | rewritten or original question | _to fill_ |
| Refusal option | hard cut-off (below it the app answers "No relevant papers found.") / low-confidence label (the app still answers and shows "the papers may not cover this") / nothing (numbers in this ADR only) | _to fill_ |
| If a cut-off or label: its value and model | one number, plus the stated uncertainty and the same-questions caveat above | _to fill_ |
| Placement | own workflow step between retrieve and generate (implemented in T8; kept unless the owner objects) | _to fill_ |

To fill after the decision: the setting values (`RERANKER_MODEL`, `RETRIEVE_DEPTH`,
`RERANK_MIN_SCORE` or `RERANK_LOW_SCORE` if any), the default change in `.env.example`
and Compose, and whether ADR-0027 (refusal signal) is needed (only for a hard cut-off).

## Alternatives Considered

These are the options measured, without a recommendation.

### Models

- **MiniLM-L-6 (cross-encoder/ms-marco-MiniLM-L-6-v2).** Smallest (91 MB), fastest
  (1.7 s median at depth 20), mean nDCG@5 0.684 to 0.725. Not distinguishable from bge.
  General web-search training data, not medical.
- **bge-reranker-base.** Largest (1.1 GB), about 6 times slower than MiniLM
  (9.8 s at depth 20), mean 0.704 to 0.718. Not distinguishable from MiniLM.
- **MedCPT cross-encoder.** Medical training data (PubMed), 438 MB, about 5 times slower
  than MiniLM, lowest means (0.639 to 0.688) and lower than MiniLM and bge at depth 20
  by a range fully below 0, clearly worse on factual and table. Its score is saturated
  near 1, which makes a refusal cut-off brittle. Its card says "public domain" and not
  for diagnostic use, which matches the project's non-clinical scope.

### Depth

- **20.** Reranker time about halves against 40; recall among visible hits 0.828.
- **40.** Recall among visible hits 0.914; mean nDCG@5 higher for all three models
  (+0.014 to +0.049) but not a real difference by the rule; reranker time about
  doubles (1.9x to 2.4x).

### Query text

- **Rewritten.** The app's own rewrite (cleaned of the leaked prefix in T1).
- **Original.** The user's question. Saves nothing in the rerank call itself (same
  time) but removes the dependency on the rewrite for this step. Differences are within
  noise for MiniLM and MedCPT, and bge is lower on paraphrased questions (-0.069).

### Placement as its own step

A separate `rerank` workflow step between retrieve and generate is already implemented
(T8): the stream shows 6 steps, `citations` is sent when `rerank` ends, and the step only
cuts to 5 when no model is set. Rejected in Discuss: reranking inside the retrieve step
(fewer interface changes, but the rerank time would be hidden from the stream and the UI).

### Refusal

- **Hard cut-off on the top-1 score.** Refuses part of the off-corpus questions but
  also questions the papers cover (see the gate table).
- **Low-confidence label.** Same signal, a looser cut-off, the app still answers.
- **Nothing.** Dense score threshold stays inert (ADR-0025).
- An LLM check or a score margin between top hits was not measured.

## Consequences

Not written. They depend on the choices above. Facts that hold whichever option is
chosen:

- The rerank step is already in the graph and the stream, off by default; leaving it off
  changes nothing for users.
- Any adopted model downloads on first use into the Hugging Face cache volume, and is
  loaded lazily on the first request unless warm-up is added.
- The eval keeps working for the dense setups; reranked runs carry `score_kind: rerank`.

## Deliberately not decided

- Whether the reranker helps answer quality (Phase 11).
- Hybrid (keyword plus vector) search and the reranker together (T10, ADR-0028).
- Whether chunk size should change after the reranker (T11).
- A human check of the golden set labels.

## Follow-up

- Fill the Decision section after the owner review and set the status.
- If a refusal option other than "nothing" is chosen: ADR-0027 and the app change (T9).
- Re-measure the chosen cut-off on new off-corpus questions, not the 20 used here.
- Optional: repeat the three timing runs to measure noise (one timed run each so far).
