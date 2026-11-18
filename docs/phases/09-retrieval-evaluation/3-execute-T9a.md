# T9a: Ingest, rewrites, four v0 runs

**Status:** done. Everything ran live. No code changed. All numbers below are v0 (unpooled, 50 questions, one labeller) and provisional. The real decisions are T10 on v1.

| Step | Result |
|---|---|
| bge-base ingest into `eval_bge-base-en-v1.5` | Pass (595 chunks; `papers` still 595 chunks, bge-small) |
| Rewrites (`data/eval/rewrites.jsonl`) | Pass, 50 rewrites, 0 fallbacks. Quality is mixed (see part 6) |
| Four v0 runs | Pass. Runs 2 to 4 used `--no-llm` and worked from the cache |
| Truncation-exactness check | Pass, 0 mismatches |
| Comparisons and curve | Done, labelled provisional |
| Gates | pytest 428 passed, ruff clean. No code changed |
| Files with paper text | None (checked) |

Words used below:
- **Rewrite:** the question turned into a search query by the small LLM.
- **nDCG@5:** a score from 0 to 1 for how well the labelled sections are ranked in the top 5. Higher is better.
- **Recall@5:** the share of labelled sections found in the top 5.
- **Off-corpus no-result rate:** the share of off-corpus questions that return nothing at the cut-off. Higher is better.

## 1. bge-base ingest

**What it is:** a second copy of the 595 chunks, embedded with `BAAI/bge-base-en-v1.5` (768 numbers per chunk instead of 384), in its own collection.

**Example:** `EMBEDDING_MODEL=BAAI/bge-base-en-v1.5 python -m scripts.ingest --collection eval_bge-base-en-v1.5`

**Result:**

| Collection | Chunks | Vector size | `embedding_model` |
|---|---|---|---|
| `papers` | 595 (unchanged) | 384 | `BAAI/bge-small-en-v1.5` |
| `eval_bge-base-en-v1.5` | 595 | 768 | `BAAI/bge-base-en-v1.5` |

- Point ids are identical in both collections (same chunking, uuid5 of paper and chunk index).
- The `bge_base` pre-flight passed (50 questions, 595 chunks).
- No container was touched. `papers` was never ingested into.

**Not checked:** whether chunk text is identical between the two collections (same code and same parsed cache, so it should be).

## 2. Rewrites

**What it is:** one rewrite per question, made by the real `make_rewrite_node` and Ollama, saved in `data/eval/rewrites.jsonl`.

**Example:** q007 "Which optimizer, momentum and dropout rate were used when finetuning Med-PaLM M?" became "Which optimizer and dropout rate were used when finetuning Med-PaLM M?" (it dropped "momentum").

**Result:**

| Fact | Value |
|---|---|
| Rewrites made | 50 (45 on-corpus, 5 off-corpus), all by the LLM |
| Model | `qwen3:0.6b` (the `OLLAMA_MODEL` default; I did not read `.env`) |
| Fallbacks or errors | 0. Ollama accepted `temperature=0` and `seed=0` without error |
| Identical to the question | 13 of 50 (q004, q005, q010, q014, q018, q027, q028, q034, q036, q041, q045, q046, q047) |
| Starting with the literal text "Rewritten question:" | 6 of 50 (q009, q015, q029, q031, q035, q042) |
| Others | Light paraphrases of the question, same length. Not answers, none empty |

**Not checked:** that Ollama really applied the seed (no error is not proof). A second LLM pass was not run, so determinism of the rewrites is unproven. Later runs use the cache, so they are deterministic anyway.

## 3. The four v0 runs

**What it is:** each question searched once per setup, top 20 kept, then cut at k=5 and similarity 0.6.

**Example:** `run --config prefix --name prefix_v0 --no-llm` (needs no LLM, as the cache holds all rewrites).

**Result (45 on-corpus questions, k=5, t=0.6):**

| Run | nDCG@5 | Recall@5 | MRR@5 | On-corpus empty rate | Off-corpus no-result rate | Max off-corpus top-1 |
|---|---|---|---|---|---|---|
| baseline_v0 | 0.521 | 0.510 | 0.688 | 0.00 | 0.00 | 0.829 |
| no_rewrite_v0 | 0.540 | 0.516 | 0.712 | 0.00 | 0.00 | 0.831 |
| prefix_v0 | 0.561 | 0.534 | 0.750 | 0.00 | 0.00 | 0.781 |
| bge_base_v0 | 0.544 | 0.549 | 0.694 | 0.00 | 0.00 | 0.788 |

Per type, nDCG@5 / Recall@5:

| Type (n) | baseline | no_rewrite | prefix | bge_base |
|---|---|---|---|---|
| factual (10) | 0.676 / 0.650 | 0.676 / 0.650 | 0.676 / 0.650 | 0.682 / 0.750 |
| paraphrased (10) | 0.459 / 0.465 | 0.460 / 0.495 | 0.532 / 0.555 | 0.493 / 0.580 |
| cross_paper (8) | 0.295 / 0.243 | 0.310 / 0.275 | 0.295 / 0.243 | 0.343 / 0.284 |
| table (8) | 0.565 / 0.583 | 0.569 / 0.521 | 0.626 / 0.583 | 0.496 / 0.521 |
| multi_section (9) | 0.578 / 0.576 | 0.659 / 0.598 | 0.643 / 0.594 | 0.670 / 0.552 |

Crowding (mean distinct papers in the top 5 chunks) and table hit rate at k=5:

| Run | Crowding | Table hit rate |
|---|---|---|
| baseline_v0 | 1.80 | 0.625 |
| no_rewrite_v0 | 1.67 | 0.750 |
| prefix_v0 | 1.69 | 0.625 |
| bge_base_v0 | 1.71 | 0.625 |

Files: `data/eval/runs/<name>.json` and `data/eval/reports/run-<name>.md` for each of the four. Run files hold only ids, section labels, scores, token counts, overlap numbers and our own rewritten query text. Longest section label is 115 characters. No chunk text.

**Not checked:** `--langsmith` was not used (T9b).

## 4. Truncation-exactness check

**What it is:** the runner keeps the top 20 at threshold 0 and applies (k, threshold) afterwards. The check asks Qdrant directly for top 5 at score 0.6 and compares.

**Example:** for q001, direct Qdrant search (limit 5, threshold 0.6) and "top 20, keep score >= 0.6, first 5" return the same chunk ids in the same order.

**Result:** 0 mismatches.

| Config | Questions checked | Mismatches (direct vs derived) | Stored run file vs fresh search |
|---|---|---|---|
| baseline_v0 | 50 (45 on-corpus) | 0 | 0 differ |
| prefix_v0 | 50 | 0 | 0 differ |
| no_rewrite_v0 | 50 | 0 | 0 differ |
| bge_base_v0 | 50 | 0 | 0 differ |

The brief asked for baseline on-corpus only. I checked all four configs on all 50 questions because it was cheap. The script is in `C:\Users\sebas\AppData\Local\Temp\claude\8ef53501dc8e147d\15ab05fa-077f-41d8-af17-1d964322eac2\scratchpad\trunc_check.py` (not committed). It also shows the stored hits equal a fresh search, so the run files are reproducible.

**Not checked:** other k or thresholds (the exactness argument covers them, but only 5 / 0.6 was tested directly).

## 5. Provisional comparisons and threshold curve

**What it is:** `compare baseline_v0 X_v0` uses the paired bootstrap rule from the plan. `curve` sweeps the threshold from 0.40 to 0.80.

**Provisional. v0 is unpooled. Do not adopt anything from this.** The verdict lines exactly as printed:

| Comparison | Printed verdict | Mean delta nDCG@5 | 95 % range | Wins / losses / ties |
|---|---|---|---|---|
| no_rewrite vs baseline | A stays: lower end of the range is -0.0057, not above 0 | +0.020 | [-0.0057, +0.0459] | 8 / 6 / 31 |
| prefix vs baseline | B replaces A: lower end of the range is above 0 and no type is clearly worse | +0.040 | [+0.0122, +0.0734] | 11 / 4 / 30 |
| bge_base vs baseline | A stays: lower end of the range is -0.0389, not above 0; clearly worse on: table | +0.024 | [-0.0389, +0.0820] | 17 / 14 / 14 |

Note on "A stays" for `no_rewrite`: the rule only says the gain is not shown. The mean is slightly above baseline, so rewrite-off did not look worse.

Threshold curve for baseline_v0 (full table in `data/eval/reports/curve-baseline_v0.md`):

| t | Recall@5 | nDCG@5 | On-corpus empty | Off-corpus no-result |
|---|---|---|---|---|
| 0.40 to 0.68 | 0.510 | 0.521 | 0.00 | 0.00 |
| 0.70 | 0.499 | 0.510 | 0.00 | 0.00 |
| 0.74 | 0.437 | 0.457 | 0.09 | 0.00 |
| 0.78 | 0.357 | 0.372 | 0.31 | 0.20 |
| 0.80 | 0.264 | 0.283 | 0.44 | 0.20 |

Reports are in `data/eval/reports/`: `compare-<B>-vs-baseline_v0.md` (three) and `curve-baseline_v0.md`.

**Not checked:** a lower-threshold rule outcome (the curve is flat below 0.68, so no lower t changes anything).

## 6. Surprises and problems

| # | Observation | Why it matters |
|---|---|---|
| 1 | The 0.6 threshold does nothing. Off-corpus no-result rate is 0.00 in all four runs. Top-1 scores for off-corpus questions go up to 0.83. The curve is flat from 0.40 to 0.68. Only at t=0.78 does one off-corpus question drop out (0.20), and then 31 % of on-corpus questions return nothing. | The threshold cannot separate off-corpus from on-corpus questions with these embedders. The app answers off-corpus questions from irrelevant chunks. T10 should state this plainly. |
| 2 | 6 of 50 rewrites start with the text "Rewritten question:". 13 of 50 rewrites equal the question. | The app's prompt ends with "Rewritten question:" and the node does not strip it. The small model sometimes repeats it. This is what production does, so the eval is faithful. The rewrite comparison is about this weak rewriter, not rewriting in general. |
| 3 | `cross_paper` is the weakest type in every run (nDCG@5 0.30 to 0.34, Recall 0.24 to 0.28). Top-5 shows only 1.7 to 1.8 distinct papers on average. | Cross-paper questions need chunks from several papers. Many have 6 to 7 labels, so Recall@5 has a low ceiling. Part of this is the metric, part is the search. |
| 4 | `bge_base` is worse on `table` (nDCG@5 0.496 vs 0.565, MRR 0.50) but better on factual and multi_section. | This is the cause of "clearly worse on: table". 8 questions only. |
| 5 | `factual` and `cross_paper` have identical nDCG@5 under baseline and prefix (10 and 8 ties). | Looks odd but the truncation check shows the stored hits equal a fresh search with the prefix, so the prefix was applied. Top-5 sections just did not change. |
| 6 | `paraphrased` (0.46) is clearly lower than `factual` (0.68) for the baseline. | Fits the plan's U3 idea: paraphrase is harder. The question text has little word overlap with the source (max 3). |
| 7 | Table hit rate at k=5 is 0.625 to 0.750, i.e. 3 to 4 of 8 table questions miss the table chunk. | A measurement only. Scrambled tables from the T8 weak-spot list may play a part. |

Nothing in the numbers points to a broken label or a broken run. Pooling in T9b is the check for missing labels.

## 7. Gates

| Gate | Result |
|---|---|
| `pytest tests/` | 428 passed |
| `ruff check .` | All checks passed |
| `mypy app/`, `bandit -r app/` | not run (no code changed) |
| Git status | New untracked: `data/eval/rewrites.jsonl`, `data/eval/runs/`, `data/eval/reports/`. Nothing committed |

No deviations from the plan. No code or test changes.

## 8. Open points for the owner

| # | What is decided | Options in practice | Recommendation |
|---|---|---|---|
| 1 | The "Rewritten question:" prefix in 6 rewrites (and 13 unchanged rewrites) | A: keep as is. Matches production, goes into the ADR as a finding. B: strip the prefix in `rewrite_node`. Small app change, then the cache must be rebuilt and all runs redone. | A for T9 and T10. Record it in ADR-0025 as a follow-up bug in the app. |
| 2 | The inert 0.6 threshold (finding 1) | A: only report it in T10. B: also plan a later fix (a better refusal signal than a score cut-off). | A now. Mention B in the ADR as a next step; it is outside Phase 9's allowed code changes. |

## 9. What T9b needs

- Pooling over the union of the top 10 of all four runs (`pool baseline_v0 no_rewrite_v0 prefix_v0 bge_base_v0 --out <file>`), skipping off-corpus. Check the 10 spot-check questions and the off-corpus questions first (see T8).
- Edit `data/eval/golden.jsonl` only through pooling. Write `data/eval/CHANGELOG.md` and record the v0 to v1 baseline Recall@5 change (v0 baseline Recall@5 is 0.510).
- Any golden edit changes the hash, so all four runs must be redone on v1. Rewrites stay valid for unchanged question text (the cache key uses the question hash), but the runs need re-running. Keep the `_v0` files as history.
- Then `scripts.sync_golden` and `run --config baseline --langsmith` (key is in `.env`).
- Commit `data/eval/rewrites.jsonl`, runs and reports (these contain no paper text).
- bge-base collection exists already. No re-ingest needed unless chunking changes.
