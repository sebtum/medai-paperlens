# T9b part 1: Pooling and golden set v1

**Status:** done. Golden set v1 is written and valid, but not committed. The pooling added 8 labels to 6 questions. The v0 set was nearly complete. All numbers below are estimates on one labeller's reading ("golden set LLM-labelled (Claude), not human-verified").

| Check | Result |
|---|---|
| Pool built from the union of 4 runs, off-corpus skipped | Done, default threshold 0.40 |
| Candidates read | 383 (all). First pass 288 (251 at score 0.74 or more, plus 37 for the 10 spot-check questions), follow-up pass the other 95 |
| Labels added | 8 on 6 questions (1 at grade 2, 7 at grade 1). 375 candidates rejected. The follow-up pass added nothing |
| Off-corpus questions | None is answered by the corpus. No finding |
| `tests/test_golden_set.py` | 4 passed |
| Pre-flight `baseline` and `bge_base` | Pass (595 chunks in each collection) |
| pytest, ruff | 428 passed, all clean (mypy and bandit not run: no code changed) |
| Estimated baseline Recall@5 (v0 to v1) | 0.510 to 0.509 (unchanged in practice) |
| Not done on purpose | No re-runs on v1, no `sync_golden`, no `--langsmith`, no commit |

Files: `data/eval/golden.jsonl` (changed, 150 labels, was 142), `data/eval/CHANGELOG.md` (new). Scratch files (pool lists, chunk text, scripts) are in the scratchpad folder, not in the repo.

Words used below:
- **Pooling:** reading the sections the search returned that no label covers, to find answers the author missed.
- **Grade:** 2 = the section fully answers. 1 = part of the answer or clear support.
- **Threshold:** the minimum similarity score a chunk needs to enter the pool list.

## 1. The pool

**What it is:** the list of unlabelled sections that appeared in the top 10 of any of the four v0 runs.

**Example:** for q021 (human rating of generated text) the list held a pathalign table that states the rating scale and blinding.

**Result:**

| Item | Value |
|---|---|
| Threshold | 0.40 (the default) |
| Candidates (sections) | 383 over 45 on-corpus questions |
| Read | 383 (all) |
| By question type | paraphrased 93, cross_paper 84, factual 80, multi_section 63, table 63 |

- Order of work: first pass on the 251 candidates with score 0.74 or more, plus 37 lower-score ones for the 10 spot-check questions (q012, q013, q017, q021, q022, q023, q026, q027, q029, q039). The follow-up pass read the remaining 95 (score below 0.74, the other 35 questions). It added nothing.
- Scores in these runs sit between 0.60 and 0.87, so a threshold of 0.40 cuts nothing. The list is the full union.

**Not checked:** chunk text was shown to me cut at 600 characters (450 for the 37 spot-check extras). Where a candidate looked possibly relevant I searched or read the whole chunk. A late sentence in a long chunk could still be missed.

## 2. Adjudication counts

**What it is:** each candidate read against its question and judged: add at grade 2, add at grade 1, or reject.

**Example (rejected):** q013 (zero-shot tuberculosis). The Med-PaLM M Abstract says "zero-shot generalization to novel medical concepts" but never names the disease or the specialist comparison. Same topic only, so not added.

**Result:**

| Type | Read | Added g2 | Added g1 | Rejected |
|---|---|---|---|---|
| paraphrased | 93 | 0 | 1 | 92 |
| cross_paper | 84 | 1 | 3 | 80 |
| factual | 80 | 0 | 0 | 80 |
| multi_section | 63 | 0 | 3 | 60 |
| table | 63 | 0 | 0 | 63 |
| Total | 383 | 1 | 7 | 375 |

| Paper | Read | Added |
|---|---|---|
| maira_2 | 64 | 3 |
| pathalign | 59 | 2 |
| cxr_vlm_generation | 53 | 0 |
| med_palm_m | 45 | 1 |
| deep_tumor_vqa | 42 | 1 |
| radfm | 31 | 0 |
| llava_med | 29 | 1 |
| conch | 24 | 0 |
| intersectional_fairness_vlm | 24 | 0 |
| gaze2report | 12 | 0 |

- Factual and table questions got nothing. For factual questions I also searched the full text for the key numbers (for example "beam size of 3", "Adafactor with momentum 0.9") and found each only in the labelled section.
- The 8 additions touch 6 questions: q019, q021 (3), q022, q038, q041, q045. Full list at the end.
- The follow-up pass (95 low-score candidates) found no addition. Typical rejects: radfm `Discussion > Clinical impact` for q020 (a claim that the design handles 2D and 3D, not how), `Methods > Data splits` and other data sections for q002 and q018, and annotation-protocol sections for q038.
- The one grade-2 addition is a judgement call: pathalign `Supplementary tables` for q021. It states scale, blinding and order. I kept it at 2 because the author gave grade 2 to the other pathalign sections of that question on the same basis.

**Not checked:** a second reader. One labeller, one reading.

## 3. Off-corpus check

**What it is:** read the top hits of q046 to q050 in all four runs, and search the whole text for the key terms.

**Example:** q049 (quantisation, pruning, latency) top hit is a LLaVA-Med ablation passage on training variants. No chunk mentions quantisation, pruning or latency.

**Result:** none of the five is answered by the corpus.

| Question | Key terms searched | Matches |
|---|---|---|
| q046 federated learning | federated | 0 |
| q047 conformal prediction | conformal, coverage guarantee | 0 |
| q048 adversarial attacks | adversarial, prompt injection | Only GANs and domain-adversarial fairness training, not attacks |
| q049 quantisation, pruning | quantisation, pruning, latency, memory footprint | 0 |
| q050 cost-effectiveness | cost-effective, health-economic | 0 |

No question needs to be retyped. q047's top hit (a "diagnostic certainty gap" section) is about confidence, not coverage guarantees.

**Not checked:** top 10 per run was read as short excerpts (380 characters), plus the term search. A synonym I did not search for could hide a hit.

## 4. Odd or wrongly named sections

**What it is:** the T8 weak-spot list: sections with false names or odd parents. Did the search return them, and did any answer a question?

**Result:**

| Section | In the pool? | Verdict |
|---|---|---|
| maira_2 annotation-protocol sections (`Workflow overview`, `Sample-level quality control`, `Options`) | `Workflow overview` once (q028, rank 4, score 0.835), `Sample-level...` and `Options` once each at low score | Text is about drawing bounding boxes. Not relevant. Rejected |
| maira_2 `Additional qualitative examples > MAIRA-2 model inputs` | Once (q038) | A worked example, not a description. Rejected |
| maira_2 `Current Study`, `FINDINGS phrases` | `Current Study` once at low score | Not relevant |
| llava_med `Experiments > Visual input example, ...` (the chatbot results) | q003, q019 | Holds the chatbot results, not the answers to those questions. Rejected |
| radfm `Dataset > Prompting ChatGPT ...` | No (already labelled in q023) | No change |
| radfm `Results > Qualitative Results`, med_palm_m `Detailed Med-PaLM M Performance`, pathalign `Supplementary tables` | `Detailed Med-PaLM M Performance` often, `Supplementary tables` 7 times | Mostly not relevant. Only pathalign `Supplementary tables` got a label (q021) |

- Overall, 14 of 900 top-5 slots (4 runs, 45 questions) were odd sections. That is 1.6 percent.
- One odd-named section turned out useful: maira_2 `Qualitative evaluation ... > Method` (a real expert-review description under a long name). It was added to q021.

**Not checked:** that no other odd section exists. Only sections the four runs returned were examined.

## 5. Estimated effect on the baseline

**What it is:** `baseline_v0` stored hits scored against the v1 labels with the project's metric functions (k=5, t=0.6, 45 on-corpus questions). Only an estimate. The real v1 runs come in part 2.

**Example:** multi_section nDCG rises because q038, q041, q045 now have extra labels that the baseline already returned in the top 5.

**Result:** Recall@5 / nDCG@5, v0 to estimated v1.

| Type (n) | Recall@5 | nDCG@5 |
|---|---|---|
| factual (10) | 0.650 to 0.650 | 0.676 to 0.676 |
| paraphrased (10) | 0.465 to 0.455 | 0.459 to 0.452 |
| cross_paper (8) | 0.243 to 0.221 | 0.295 to 0.292 |
| table (8) | 0.583 to 0.583 | 0.565 to 0.565 |
| multi_section (9) | 0.576 to 0.600 | 0.578 to 0.610 |
| All (45) | 0.510 to 0.509 | 0.521 to 0.525 |

What it says:
- Few labels were missing. 8 new labels on 142 (about 6 percent), found on 6 of 45 questions.
- Recall can fall when a label is added: the new section is not in the top 5, so the share found shrinks. This happened for paraphrased and cross_paper.
- The other three runs move the same way (all Recall@5 within -0.006 to -0.001, nDCG@5 +0.004 to +0.006). The ranking of setups is not expected to change from this alone.
- The low cross_paper recall (0.22) is not a labelling gap. I did not pad cross_paper labels. The added labels are few and each one answers its question.

**Recomputed after the follow-up pass:** unchanged, because that pass added no labels. The numbers above stand.

**Not checked:** a second reader would likely find a few more or different labels. Pooling only sees sections the four runs returned in their top 10. The real v1 runs are needed for any decision.

## 6. Problems and surprises

| # | Observation | Why it matters |
|---|---|---|
| 1 | The default pool list is long (383 candidates) | First pass used score 0.74 or more (251). A follow-up pass read the rest. The final record uses the full list at the default 0.40 |
| 2 | Added labels can lower Recall@5 | Not a bug. The denominator grows. Read the nDCG too |
| 3 | A background section, `Metadata and Structured Description Generation` (deep_tumor_vqa, q041), was rejected as too thin (two sentences, duplicates the labelled child section) | Borderline. A different reader might add it at grade 1 |
| 4 | The earlier `python -` call in my shell opened an interactive Python and ran 2 minutes before I stopped it | No effect on files. Mentioned for honesty |

No test failed. No code changed.

## 7. Gates

| Gate | Result |
|---|---|
| `pytest tests/` | 428 passed |
| `pytest tests/test_golden_set.py` | 4 passed |
| `run --config baseline --preflight-only` | Passed (595 chunks, 50 questions) |
| `run --config bge_base --preflight-only` | Passed (595 chunks in `eval_bge-base-en-v1.5`) |
| `ruff check .` | All checks passed |
| `mypy app/`, `bandit -r app/` | Not run (no code changed) |
| File format | One JSON object per line, same field order, LF, trailing newline. Diff: 6 lines changed in `golden.jsonl` |

## 8. Open points for the owner

| # | What is decided | Options in practice | Recommendation |
|---|---|---|---|
| 1 | The banned-label list in `app/evaluation/golden.py` (T8 option B) | A: leave it. B: add the odd `maira_2` annotation names. The pool shows they hardly ever surface (1.6 percent of top-5 slots, none answers a question) | A. Option B would only change test code and has no effect on results |
| 2 | A second reader for the 8 additions and the borderline rejects | A: accept as is. B: you spot-check the 8 additions (list below) | A. All 383 candidates are read, each addition has a reason in the changelog |
| 3 | Judgement calls: q021 pathalign grade 2, q041 Metadata section rejected | A: keep as is. B: you spot-check these two | A. Both are logged with reasons in `data/eval/CHANGELOG.md` |

## 9. What part 2 needs

1. Commit `data/eval/golden.jsonl` and `data/eval/CHANGELOG.md` (and T9a files) first. `sync_golden` refuses an uncommitted golden file, and run files record the git hash.
2. Run `scripts.sync_golden`, then redo the four runs on v1 (names such as `baseline`, `no_rewrite`, `prefix`, `bge_base`). Rewrites in `data/eval/rewrites.jsonl` stay valid (question text unchanged). Keep the `_v0` files as history.
3. Run `run --config baseline --langsmith` and check the local and LangSmith numbers match.
4. Compare the real v1 baseline Recall@5 with this estimate (0.509) and with v0 (0.510). Record the difference.
5. Commit rewrites, runs and reports (no paper text in any of them).

## Additions (golden set v1)

Format: question, paper, section, grade, reason. Same table as in `data/eval/CHANGELOG.md`.

| Question | Paper | Section | Grade | Reason |
|---|---|---|---|---|
| q019 | llava_med | Conclusions | 1 | One sentence states the method: instruction data made by a language-only GPT-4 with a self-instruct pipeline. |
| q021 | pathalign | Supplementary tables | 2 | Rating-rubric table: 1 to 5 scale, blinded to text source, random order. |
| q021 | med_palm_m | Results > Med-PaLM M performs encouragingly on radiology report generation across model scales | 1 | Side-by-side ranking of four reports by four raters. Overlaps labelled sections. |
| q021 | maira_2 | Qualitative evaluation of twenty random MAIRA-2 generated reports > Method | 1 | Radiologist review with edit/delete/add tool and a none/minor/significant impact rating. |
| q022 | maira_2 | Methods > Data | 1 | Indication, Technique, Comparison and lateral view are extracted per study as inputs. |
| q038 | maira_2 | Extended results > Further ablations on additional inputs | 1 | Names the extra inputs (prior study, Comparison, lateral image, Technique, Indication). |
| q041 | deep_tumor_vqa | Introduction | 1 | Defines the four question types, 29 subtypes and the two answer formats. |
| q045 | pathalign | Introduction | 1 | One variant uses only the contrastive loss, the other adds a frozen LLM in two stages. |
