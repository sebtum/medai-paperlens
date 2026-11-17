# T9c: v1 runs, comparisons, LangSmith

**Status:** done, no problems. The four runs, the comparisons, the sync and the first real LangSmith experiment all worked on the first try. No code changed. The pooling estimate for the baseline matched the real v1 numbers exactly. These are the numbers T10 will use. Golden set is still LLM-labelled (Claude), not human-verified.

| Check | Result |
|---|---|
| Four v1 runs (`baseline`, `no_rewrite`, `prefix`, `bge_base`, all `--no-llm`) | Pass. Golden sha256 `4a4733e5fa76`, git `11318b7` (not dirty) |
| v1 baseline vs pooling estimate | Matched: Recall@5 0.509, nDCG@5 0.525 |
| Comparisons and curve on v1 | Done. Verdicts in part 3 |
| LangSmith sync | Dataset `paperlens-retrieval-golden`, 50 examples created, tag `11318b7`. Second sync is a no-op |
| LangSmith experiment | Pass. First real `aevaluate` call. Scores equal the local ones |
| Privacy check | Pass. No chunk text in runs, reports or rewrites |
| Gates | `pytest` 428 passed, `ruff` clean. `mypy`/`bandit` not run (no code changed) |
| `.env` | Never read or printed |

Words used below:
- **nDCG@5:** a score from 0 to 1 for how well the labelled sections are ranked in the top 5. Higher is better.
- **Recall@5:** the share of labelled sections found in the top 5.
- **MRR@5:** how high the first correct section sits. 1.0 means always first.
- **Off-corpus no-result rate:** the share of off-corpus questions for which nothing passes the cut-off. Higher is better.
- **`aevaluate`:** LangSmith's function that runs a search function over a dataset and scores it.

## 1. The four v1 runs

**What it is:** each of the 50 questions searched once per setup. Top 20 kept, then cut at k=5 and similarity 0.6. Rewrites came from the cache (`--no-llm`).

**Example:** `run --config prefix --no-llm` wrote `data/eval/runs/prefix.json` and `data/eval/reports/run-prefix.md`. The `_v0` files were not touched.

**Result (45 on-corpus questions, k=5, t=0.6):**

| Run | nDCG@5 | Recall@5 | MRR@5 | Off-corpus no-result rate |
|---|---|---|---|---|
| baseline | 0.525 | 0.509 | 0.703 | 0.00 |
| no_rewrite | 0.545 | 0.514 | 0.727 | 0.00 |
| prefix | 0.566 | 0.531 | 0.767 | 0.00 |
| bge_base | 0.550 | 0.543 | 0.706 | 0.00 |

Per type, nDCG@5 / Recall@5:

| Type (n) | baseline | no_rewrite | prefix | bge_base |
|---|---|---|---|---|
| factual (10) | 0.676 / 0.650 | 0.676 / 0.650 | 0.676 / 0.650 | 0.682 / 0.750 |
| paraphrased (10) | 0.452 / 0.455 | 0.452 / 0.485 | 0.524 / 0.540 | 0.487 / 0.565 |
| cross_paper (8) | 0.292 / 0.221 | 0.307 / 0.252 | 0.292 / 0.221 | 0.379 / 0.273 |
| table (8) | 0.565 / 0.583 | 0.569 / 0.521 | 0.626 / 0.583 | 0.496 / 0.521 |
| multi_section (9) | 0.610 / 0.600 | 0.690 / 0.622 | 0.677 / 0.619 | 0.675 / 0.548 |

- The off-corpus no-result rate is 0.00 in all four runs. The 0.6 threshold still does nothing (see part 3).
- On-corpus empty-result rate is 0.00 in all four. Max off-corpus top-1 score: 0.829 / 0.831 / 0.781 / 0.788.

**Not checked:** a second run to prove determinism. It did happen by accident: the baseline ran twice more (for LangSmith) and the file hash was identical each time.

## 2. v0 to v1 baseline, and the pooling estimate

**What it is:** the baseline on the 142-label v0 set against the 150-label v1 set, and against the estimate made in T9b by rescoring the old hits.

**Example:** cross_paper Recall@5 falls from 0.243 to 0.221 because new labels were added that the top 5 does not contain.

**Result:**

| Measure | v0 | Estimate (T9b) | Real v1 | Estimate matched? |
|---|---|---|---|---|
| Recall@5 | 0.510 | 0.509 | 0.509 | Yes |
| nDCG@5 | 0.521 | 0.525 | 0.525 | Yes |
| MRR@5 | 0.688 | not estimated | 0.703 | n/a |

| Type (n) | Recall@5 v0 to v1 | nDCG@5 v0 to v1 | MRR@5 v1 |
|---|---|---|---|
| factual (10) | 0.650 to 0.650 | 0.676 to 0.676 | 0.700 |
| paraphrased (10) | 0.465 to 0.455 | 0.459 to 0.452 | 0.745 |
| cross_paper (8) | 0.243 to 0.221 | 0.295 to 0.292 | 0.625 |
| table (8) | 0.583 to 0.583 | 0.565 to 0.565 | 0.625 |
| multi_section (9) | 0.576 to 0.600 | 0.578 to 0.610 | 0.796 |

- Every per-type value equals the T9b estimate. This is expected: the estimate used the same stored hits and the same metric code. It confirms the new runs did not change the hits.
- The change from v0 to v1 is small. Pooling found few missing labels (8 on 142). The label set was nearly complete for the four setups tested.

**Not checked:** missing labels that none of the four setups retrieves (pooling cannot see those).

## 3. Comparisons and threshold curve on v1

**What it is:** `compare baseline X` uses the paired bootstrap rule from the plan (nDCG@5, 45 questions, 1000 resamples within each type, 95 % range). `curve` sweeps the threshold from 0.40 to 0.80. I report the printed verdicts only. No adoption decision is made here.

**Example:** `compare baseline prefix` printed the line below and wrote `data/eval/reports/compare-prefix-vs-baseline.md`.

**Result, verdict lines exactly as printed:**

| Comparison | Printed verdict | Mean delta nDCG@5 | 95 % range | Wins / losses / ties |
|---|---|---|---|---|
| no_rewrite vs baseline | A stays: lower end of the range is -0.0058, not above 0 | +0.020 | [-0.0058, +0.0459] | 8 / 6 / 31 |
| prefix vs baseline | B replaces A: lower end of the range is above 0 and no type is clearly worse | +0.041 | [+0.0123, +0.0737] | 11 / 4 / 30 |
| bge_base vs baseline | A stays: lower end of the range is -0.0347, not above 0; clearly worse on: table | +0.026 | [-0.0347, +0.0835] | 17 / 14 / 14 |

The v0 verdicts were the same in all three cases.

Per-type mean delta (nDCG@5, B minus A), wins / losses / ties:

| Type | no_rewrite | prefix | bge_base |
|---|---|---|---|
| factual | +0.000 (0/0/10) | +0.000 (0/0/10) | +0.006 (1/1/8) |
| paraphrased | +0.001 (3/4/3) | +0.073 (5/1/4) | +0.035 (5/4/1) |
| cross_paper | +0.015 (1/1/6) | +0.000 (0/0/8) | +0.087 (5/2/1) |
| table | +0.005 (1/1/6) | +0.061 (3/0/5) | -0.069 (1/3/4), clearly worse |
| multi_section | +0.081 (3/0/6) | +0.067 (3/3/3) | +0.065 (5/4/0) |

**Threshold curve (baseline, k=5):** full table in `data/eval/reports/curve-baseline.md`.

| t | Recall@5 | nDCG@5 | On-corpus empty | Off-corpus no-result |
|---|---|---|---|---|
| 0.40 to 0.68 | 0.509 | 0.525 | 0.00 | 0.00 |
| 0.70 | 0.497 | 0.514 | 0.00 | 0.00 |
| 0.72 | 0.464 | 0.488 | 0.04 | 0.00 |
| 0.74 | 0.437 | 0.461 | 0.09 | 0.00 |
| 0.76 | 0.378 | 0.397 | 0.27 | 0.00 |
| 0.78 | 0.355 | 0.372 | 0.31 | 0.20 |
| 0.80 | 0.258 | 0.281 | 0.44 | 0.20 |

- The curve is flat up to 0.68, so the 0.6 threshold removes nothing. No off-corpus question drops out before 0.78, and by then 31 % of on-corpus questions return nothing. Max off-corpus top-1 is 0.829.
- Only 5 off-corpus questions: the no-result rate moves in steps of 0.20.

**k = 3 / 5 / 10, baseline run report** (measure only; the default k stays 5):

| k | Recall | nDCG | MRR | Crowding (distinct papers) | Table hit rate | Evidence tokens |
|---|---|---|---|---|---|---|
| 3 | 0.463 | 0.536 | 0.693 | 1.40 | 0.625 | 1465 |
| 5 | 0.509 | 0.525 | 0.703 | 1.80 | 0.625 | 2430 |
| 10 | 0.687 | 0.598 | 0.715 | 2.47 | 0.875 | 4709 |

**U3, paraphrased vs factual (baseline, k=5):**

| Measure | factual (10) | paraphrased (10) | Gap |
|---|---|---|---|
| nDCG@5 | 0.676 | 0.452 | 0.224 |
| Recall@5 | 0.650 | 0.455 | 0.195 |
| Longest shared word run with the source (mean / max) | 2.9 / 4 | 2.0 / 3 | n/a |

Paraphrased is clearly harder. The gap is the same for `no_rewrite` (0.452) and shrinks only with `prefix` (0.524) and `bge_base` (0.487).

**Not checked:** a lower-threshold rule outcome beyond what the curve prints. The 5 off-corpus questions are too few for a firm statement.

## 4. LangSmith sync

**What it is:** `scripts.sync_golden` copies `golden.jsonl` into a LangSmith dataset. Git stays the source of truth.

**Example:** `.\.venv\Scripts\python.exe -m scripts.sync_golden`

**Result:**

| Item | Value |
|---|---|
| Dry-run option | None exists in the script. The first call is the real sync (safe: it creates the dataset or changes nothing) |
| Dataset name | `paperlens-retrieval-golden` |
| Examples created | 50 (0 updated, 0 deleted) |
| Version tag | `11318b7` (short git hash of HEAD) |
| golden sha256 | `4a4733e5fa76` |
| Dataset | newly created |
| Second sync output | `Dataset 'paperlens-retrieval-golden' is up to date (sha256 4a4733e5fa76); nothing written.` (no-op) |

No authentication or region problem. The experiment link points to the EU host (`eu.smith.langchain.com`), so the key and region in `.env` work.

**Not checked:** what happens on a golden change after the dataset exists (update and delete path) against the real service. Only the unit tests with a fake client cover it.

## 5. LangSmith experiment and the equality check

**What it is:** `run --config baseline --no-llm --langsmith` also runs the same search as a LangSmith experiment (k=5, t=0.6). LangSmith calls our search function once per dataset example and our evaluators score each result. Afterwards the runner compares every per-question score with the local run (tolerance 1e-9) and exits with an error on any difference.

**Example:** the experiment is named `retrieval-baseline-f778f891`.

**Result:** the check passed. The printed line: `LangSmith experiment finished at k=5, t=0.6; its scores equal the local run.` No API-shape problems at all: `aevaluate` accepted the arguments, the synced examples worked as data, and the evaluator result shape was accepted.

| Score (feedback key) | Scored on | Value (LangSmith = local) |
|---|---|---|
| `recall@5` | 45 on-corpus | 0.509 |
| `mrr@5` | 45 on-corpus | 0.703 |
| `ndcg@5` | 45 on-corpus | 0.525 |
| `empty_on_corpus` | 45 on-corpus | 0.00 |
| `no_result_off_corpus` | 5 off-corpus | 0.00 |

I did not read the numbers back from the LangSmith website. They equal the local ones because the runner's own check enforces equality for every question and key.

**What you see on the LangSmith website** (EU host, your account):
- Left menu "Datasets & Experiments", dataset `paperlens-retrieval-golden`.
- Tab "Examples": 50 rows. Inputs are `id` and `question`. Reference outputs are `type` and the labelled sections with grades. No paper text.
- Tab "Experiments": one row per experiment, named `retrieval-baseline-<suffix>`, with the average of each score above and the run config in its metadata (embedder, collection, prefix, rewrite model, golden hash, git hash).
- Open an experiment: one row per question with the query used, the 20 ranked hits (paper id, section label, chunk number, chunk type, score) and the five scores.
- Dataset version history: the tag `11318b7`.

**Two experiments exist.** I ran the LangSmith step twice. The first run's output was cut off in my shell, so I repeated it to capture the experiment name. The first experiment has a different suffix (name not captured). Both hold identical scores. Delete one on the website if you like. The local `baseline.json` is byte-identical after both runs.

**Not checked:** how the page looks on the website (I did not open it).

## 6. Privacy check

**What it is:** a scan for chunk text in everything the evals write, plus a list of what left the machine.

**Example:** I took every 8-word sequence from all 1190 chunk texts (both collections) and looked for it in the runs, reports and rewrites.

**Result:**

| File group | 8-word matches with chunk text |
|---|---|
| `data/eval/runs/*.json` (8 files) | Matches exist, but every one lies inside a section label (headings such as "... to generate medical visual question answering data ..."). With section labels excluded: 0 |
| `data/eval/reports/*.md` (16 files) | 0 |
| `data/eval/rewrites.jsonl` | 0 |
| `data/eval/CHANGELOG.md` | 0 |
| `data/eval/golden.jsonl` | Section labels only, same as runs |

- A run file question record holds only: `id`, `type`, `query` (our own question or rewrite), `ngram_overlap`, `hits` (paper id, section label, chunk number, chunk type, score) and `chunk_tokens` (numbers). Longest section label: 115 characters.
- **Sent to LangSmith, dataset:** `{id, question}` as inputs; `{type, relevant: [paper_id, section, grade]}` as reference outputs; the golden hash as example metadata.
- **Sent to LangSmith, experiment:** the target returns `{query, hits: [paper_id, section, chunk_index, chunk_type, score]}`. The five scores. Metadata: run config, hashes, k, threshold. No chunk text, no abstracts.
- Extra traces: the search function makes no LLM call under `--no-llm`, so the `@traceable` LLM methods were not invoked.

**Not checked:** whether `LANGSMITH_TRACING` is set in `.env` (I did not read it). Even if it is, nothing in this path is traced beyond the above. Paraphrased matches with fewer than 8 words in a row were not tested.

## 7. Problems and surprises

| # | Observation | Why it matters |
|---|---|---|
| 1 | The real v1 baseline equals the pooling estimate exactly | Confirms the pooling changed labels but not hits. It says nothing about missing labels outside the pool |
| 2 | Printed verdicts for all three comparisons are the same as in v0 | The 8 new labels did not change any decision. Means moved by up to 0.01 |
| 3 | `compare` shows bge_base better on 4 of 5 types, worse on table (-0.069, 3 losses of 8) | The "clearly worse on: table" flag blocks adoption on a rule of the plan. 8 questions only |
| 4 | `factual` and `cross_paper` do not change at all under `prefix` (all ties) | Seen in v0 too. The T9a truncation check showed the prefix was applied |
| 5 | The 0.6 threshold is inert (part 3) | The app answers off-corpus questions from unrelated chunks. T10 should state it |
| 6 | No dry-run option in `sync_golden` | The first call wrote to LangSmith. No harm, the dataset did not exist |
| 7 | I ran the experiment twice (see part 5) | One duplicate experiment on LangSmith. Harmless |

## 8. Gates

| Gate | Result |
|---|---|
| `pytest tests/ -q` | 428 passed |
| `ruff check .` | All checks passed |
| `mypy app/`, `bandit -r app/` | Not run: no code or test changed |
| Git status | Only new untracked files in `data/eval/runs/` (4) and `data/eval/reports/` (8: 4 run reports, 3 comparisons, 1 curve). `golden.jsonl` and `rewrites.jsonl` unchanged. All `_v0` files untouched |

Deviations: none from the brief. No code change was needed. I used one scratch script (`leak.py` in the scratchpad) that reads Qdrant only, not LangSmith, for the privacy scan.

## 9. Open points for the owner

| # | What is decided | Options in practice | Recommendation |
|---|---|---|---|
| 1 | The duplicate LangSmith experiment | A: leave both. B: delete the older one on the website | B, if you want a clean list. No effect on results |
| 2 | Commit the new run files and reports (no paper text, checked) | A: commit all 12 new files. B: commit only the final reports | A. ADR-0024 says runs are committed |
| 3 | Whether T10 may adopt `prefix` | The printed verdict is "B replaces A". The plan allows a prefix setting as a code change. Gain is +0.041 nDCG@5, mainly paraphrased and table | Decide in T10 after reading the ADR notes. Not decided here |
| 4 | `bge_base` and `no_rewrite` | Both "A stays". If either won, the plan says stop and re-plan with you | Nothing to do now |

## 10. What T10 needs

- Commit the 12 new files first (runs, reports). The `baseline` run record carries git `11318b7`.
- Use these numbers: baseline nDCG@5 0.525 / Recall@5 0.509; verdicts and per-type tables in part 3; curve and k table in part 3; U3 gap 0.224 nDCG.
- ADR-0025 must record every outcome, including "kept", the inert threshold, the `table` flag on bge_base, and the sentence "golden set LLM-labelled (Claude), not human-verified".
- Known weaknesses to state: 5 off-corpus questions; 8 to 9 questions per type; one labeller; the weak 0.6B rewriter (13 of 50 rewrites equal the question, 6 start with "Rewritten question:").
- Allowed code changes: threshold constant plus UI slider, or the prefix setting (`query_prefix`, `EMBEDDING_QUERY_PREFIX`).

## Files

| File | State |
|---|---|
| `data/eval/runs/{baseline,no_rewrite,prefix,bge_base}.json` | new |
| `data/eval/reports/run-{baseline,no_rewrite,prefix,bge_base}.md` | new |
| `data/eval/reports/compare-{no_rewrite,prefix,bge_base}-vs-baseline.md` | new |
| `data/eval/reports/curve-baseline.md` | new |
| `docs/phases/09-retrieval-evaluation/3-execute-T9c.md` | new (this file) |
