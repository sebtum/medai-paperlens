# Golden set changelog

Golden set LLM-labelled (Claude), not human-verified.

## v0 to v1 (2026-11-17): pooling

v0 is the 50-question set as authored (142 labels, 83 at grade 2, 59 at grade 1).
v1 adds labels found by pooling. No question text, id, type or `source_chunks` entry changed. No label was removed or re-graded.

**Pooling method.** An author can miss a section that answers a question. To find such sections without favouring one search setup, the top-10 results of all four v0 runs (`baseline_v0`, `no_rewrite_v0`, `prefix_v0`, `bge_base_v0`) were merged. Every section in that union that no label covered was read against its question. Command: `python -m scripts.eval_retrieval pool baseline_v0 no_rewrite_v0 prefix_v0 bge_base_v0` (default threshold 0.40). Off-corpus questions are skipped by the pool command and were checked separately.

**Standard used.** Grade 2: the section directly and fully answers the question. Grade 1: it holds part of the answer or clear support. Otherwise nothing is added. A section on the same topic only is not added.

**Candidate list.** The first pass read 288 of the 383 candidates. A second pass read the remaining 95 (score below 0.74) with the same standard and added nothing.

| Item | Value |
|---|---|
| Threshold used | 0.40 (the default; every candidate in the union of the four runs' top 10) |
| Candidates read | 383 (all of them) |

**Counts.**

| Result | Candidates |
|---|---|
| Added at grade 2 | 1 |
| Added at grade 1 | 7 |
| Rejected | 375 |
| Total labels after | 150 (84 at grade 2, 66 at grade 1) |

Off-corpus questions q046 to q050: the top hits of all four runs were read, and the corpus was searched for the key terms. No section answers any of them. Nothing changed for them.

## Additions

| Question | Type | Paper | Section | Grade | Reason |
|---|---|---|---|---|---|
| q019 | paraphrased | llava_med | Conclusions | 1 | One sentence states the method: instruction data made by a language-only GPT-4 with a self-instruct pipeline. Same support as the already-labelled Abstract and Introduction. |
| q021 | cross_paper | pathalign | Supplementary tables | 2 | The rating-rubric table states the 1 to 5 scale, that raters were blinded to the text source, and that texts came in random order. |
| q021 | cross_paper | med_palm_m | Results > Med-PaLM M performs encouragingly on radiology report generation across model scales | 1 | Describes the side-by-side ranking by four clinician raters of four reports. Overlaps the labelled evaluation sections, so only supporting. |
| q021 | cross_paper | maira_2 | Qualitative evaluation of twenty random MAIRA-2 generated reports > Method | 1 | A radiologist reviewed generated findings with an edit/delete/add tool and rated clinical impact as none, minor or significant. A review setup, not a blinded scale. |
| q022 | cross_paper | maira_2 | Methods > Data | 1 | States that the Indication, Technique and Comparison sections and the lateral view are extracted per study as model inputs. |
| q038 | multi_section | maira_2 | Extended results > Further ablations on additional inputs | 1 | Names the extra inputs (prior image and report, Comparison, lateral image, Technique, Indication). Does not say how missing inputs are handled. |
| q041 | multi_section | deep_tumor_vqa | Introduction | 1 | Defines the four question types and 29 subtypes, and the two answer formats. Does not cover generation or metadata. |
| q045 | multi_section | pathalign | Introduction | 1 | Says one variant is trained with the contrastive loss only and the second adds a frozen LLM in the two-stage setup. No query-vector count. |

Golden set LLM-labelled (Claude), not human-verified.

## Rewrite cache cleanup (2026-11-18): Phase 10, rewriter leak

This changes the stored rewrites in `rewrites.jsonl`, not the golden set. No question, id, type, label or `source_chunks` entry changed, so the golden hash is the same.

**What was wrong.** For 6 of the 50 questions, the small rewrite model repeated the label "Rewritten question:" at the start of its answer, and that label was stored and searched with. The rewriter now strips such a label (any case, repeated), plus surrounding quotes and whitespace (`clean_rewrite` in `app/workflow/rewrite.py`).

**What was done.** The stored rewrites were cleaned in place. Ollama was not called, so the other 44 rewrites are byte-for-byte the same text and the next run measures only this fix. The prompt text did not change. The cache key now hashes the prompt plus a cleanup version (`REWRITE_CLEANUP_VERSION`, now `1`), so all 50 lines got a new key and a later change to the cleanup invalidates the file.

| Question | Text before | Text after |
|---|---|---|
| q009 | Rewritten question: What hardware and duration did RoentGen fine-tune for 60k training steps? | What hardware and duration did RoentGen fine-tune for 60k training steps? |
| q015 | Rewritten question: Is accuracy higher when answer options are provided than when the answer must be typed out freely? | Is accuracy higher when answer options are provided than when the answer must be typed out freely? |
| q029 | Rewritten question: What is the F1 score achieved by the gaze-guided model in the clinical-efficacy comparison, and which compared method scores higher? | What is the F1 score achieved by the gaze-guided model in the clinical-efficacy comparison, and which compared method scores higher? |
| q031 | Rewritten question: What were the total average multi-choice and free-text scores of the best-performing model and of random guessing in the 3D CT question-answering benchmark? | What were the total average multi-choice and free-text scores of the best-performing model and of random guessing in the 3D CT question-answering benchmark? |
| q035 | Rewritten question: How did the scores of chest X-ray report generation change after the classification task was removed from the 84B model? | How did the scores of chest X-ray report generation change after the classification task was removed from the 84B model? |
| q042 | Rewritten question: Which datasets were selected for the intersectional fairness study, why were other datasets excluded, and how were the cohorts split into subgroups and train, validation, and test sets? | Which datasets were selected for the intersectional fairness study, why were other datasets excluded, and how were the cohorts split into subgroups and train, validation, and test sets? |

Exactly 6 rewrite texts changed (q009, q015, q029, q031, q035, q042). Runs saved before this change (Phase 9) used the old rewrites for these 6 questions.
