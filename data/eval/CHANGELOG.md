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

## v2 (frozen, 2026-11-18): Phase 10, 15 near-miss questions and reranker pooling

**Status: frozen (task T6, 2026-11-18).** Draft hash `af6ce07a...64af` (65 questions, 150 labels) became final hash `cc08395a87ca30fc302b98065395f77404f192d3999f22428cab009b38022b0c` after the labelling pass below. All 13 runs of Phase 10 carry the final hash. Golden set LLM-labelled (Claude), not human-verified.

v2 adds 15 off-corpus questions, q051 to q065. No question text, id, type, label or `source_chunks` entry of q001 to q050 changed (the first 50 lines of `golden.jsonl` and of `rewrites.jsonl` are byte-identical to v1).

**Golden hash change.** The golden file hash changes from `4a4733e5fa7623f324adf0c7704ae9ec49326366a66f9a74071b746c90a52649` (v1, 50 questions) to `af6ce07add4ad753e9d202adfe107718958bede10c7c51b296e81bb1837264af` (v2 draft, 65 questions). Runs saved before this change carry the v1 hash.

**Quotas.** `off_corpus` 5 to 20, total 50 to 65. `QUOTAS` in `app/evaluation/golden.py` and the two golden tests were updated.

**What the new questions are.** Medical-AI questions that look like they belong to the 10 papers but are not answered by them. They are written before any reranker result exists, and they were not tuned to the dense scores. Kinds of near miss:

| Kind | Questions |
|---|---|
| Same modality, different task | q053 (survival from slides), q055 (brain tumour segmentation), q056 (segmentation with LLaVA-Med), q057 (MRI tumour grading), q063 (molecular biomarkers from slides) |
| Model the corpus does not cover | q051 (GPT-4V on retinal photos), q054 (MedGemma), q063 (UNI, Virchow) |
| Dataset or setting the corpus does not use | q052 (echocardiography video), q062 (cardiac MRI), q065 (colonoscopy video), q061 (low-resource languages) |
| Topic next to a covered paper | q058 (Stable Diffusion on skin photos), q059 (bias in generated reports), q060 (eye tracking for slides), q064 (retrieval-augmented generation against hallucination) |

**Corpus check.** For each question, the dense top-10 sections (from a scratch run with the baseline setup and real Ollama rewrites) were listed with `pool --include-off-corpus`, and the chunk text was read from the `papers` collection (every chunk that contained a key term, plus the section list of all 150 hits). No section answers any of the 15. Dropped: 0. Rewritten: 0.

| Item | Value |
|---|---|
| Questions written | 15 |
| Dropped or rewritten after the check | 0 |
| Flagged by the safety filter (`classify_node`) | 0 |
| New rewrites from Ollama (`qwen3:0.6b`) | 15 (q051 to q065), none starts with "Rewritten question:" |

Golden set LLM-labelled (Claude), not human-verified. The same author wrote these questions and judged the retrieved sections, so the "no answer in the corpus" judgement is not independent.

### Reranker pooling pass (2026-11-18, task T6)

The nine reranked runs put some sections into the top 5 that no dense run showed. An author can miss a section that answers a question, so these sections were read before v2 was frozen. No question text, id, type or `source_chunks` entry changed. No label was removed or re-graded.

**Method.** The pool is the union of the top 5 chunks (threshold 0, no cut) of nine runs: `rerank_minilm`, `rerank_bge`, `rerank_medcpt`, each rewritten at depth 20, original question at depth 20 (`_orig`) and rewritten at depth 40 (`_d40`). Command: `python -m scripts.eval_retrieval pool <the nine runs> --k 5 --threshold 0`. The chunk text of every candidate was read from the `papers` collection and judged by the Phase 9 standard (grade 2 fully answers, grade 1 part of the answer or clear support, a section on the same topic only is not added). Candidates were judged by their text, not by the reranker score or rank. The labeller is the same author who wrote the questions, so this is not a blind pass in the human sense.

| Item | Value |
|---|---|
| Questions pooled | 45 on-corpus (off-corpus skipped by the pool command, checked separately below) |
| Candidate sections read | 241 |
| Added at grade 2 | 0 |
| Added at grade 1 | 9 |
| Rejected | 232 (all recorded with a reason in `data/eval/adjudicated.jsonl`) |
| Total labels after | 159 (84 at grade 2, 75 at grade 1), was 150 |

Candidates by reranker (a candidate counts for a reranker if any of its three runs had it in the top 5): `minilm` 127 candidates and 5 additions, `bge` 140 and 5, `medcpt` 140 and 6. 25, 41 and 56 candidates were seen by only that one reranker. By question type, the additions went to cross_paper (3) and multi_section (6). Factual, paraphrased and table questions got no addition (62, 55 and 39 candidates).

**Off-corpus questions (q046 to q065).** The top-5 sections of the nine runs were checked for all 20 (196 candidate sections) with a key-term search of each chunk and a read of every hit. No section answers any of them. The closest is q064 (retrieval-augmented generation against hallucination): MAIRA-2 sections report that prior studies and the lateral view reduce hallucinated comparisons and lateral mentions. That is extra input, not retrieval-augmented generation, so the question stays off-corpus (same call as in the v2 draft). Nothing changed for any off-corpus question.

**Additions.**

| Question | Type | Paper | Section | Grade | Reason |
|---|---|---|---|---|---|
| q022 | cross_paper | maira_2 | Introduction | 1 | States that MAIRA-2 takes the lateral view, prior frontal image, prior report, Indication, Technique and Comparison sections as inputs. Same support as the labelled Methods > Data. |
| q022 | cross_paper | med_palm_m | Appendix > Chest X-ray report generation | 1 | Prompt examples give the view ('Given the LATERAL view X-ray image') and the 'Reason for the study' next to the image. |
| q028 | cross_paper | maira_2 | Discussion | 1 | One sentence names the image encoder (Rad-DINO-MAIRA-2) and the language model (Vicuna 7B v1.5). Same facts as the labelled architecture section. |
| q037 | multi_section | conch | Results > Zero-shot cross-modal retrieval | 1 | One sentence states the segmentation method: divide each WSI into tiles and zero-shot classify each tile. The section label is about retrieval, the chunk holds this sentence. |
| q037 | multi_section | conch | Online Methods > Supervised classification experiments | 1 | Describes tile embeddings from the image encoder and an attention-pooled slide-level representation for slide classification (supervised variant only). |
| q039 | multi_section | med_palm_m | Appendix > MultiMedBench | 1 | Lists which datasets and splits enter the training mixture (for example PubMedQA is evaluation only) and holds the mixture-ratio figure caption. Does not say how the ratios were chosen. |
| q041 | multi_section | deep_tumor_vqa | Discussion and Conclusion | 1 | One sentence states that structured metadata and QA pairs are generated from organ and lesion segmentation masks. Does not define the question types. |
| q042 | multi_section | intersectional_fairness_vlm | Discussion | 1 | States how subgroups were defined (age bins, binary gender, binarised race) and why: data availability and minimum subgroup sample size. |
| q042 | multi_section | intersectional_fairness_vlm | Methods > Experimental Design and Baseline Comparisons | 1 | States that models were trained on HAM10000 and tested on the held-out HAM10000 test set (internal) and on BCN20000 (external). No split sizes. |

**Notable rejections.** A section that was read and could be argued either way: llava_med `Experiments > Visual input example, Biomedical image from [6]` for q044 holds the data-size and two-stage ablation result, but its heading is a banned odd Phase 8 label, so it cannot be labelled. Others with a reason in `adjudicated.jsonl`: radfm `Discussion > Clinical impact` for q020 (same call as Phase 9), med_palm_m `Appendix > Detailed Med-PaLM M Performance` for q035 (says joint training helps, gives no result for dropping classification), pathalign `Supplementary figures` for q018 (captions of the case and slide hierarchy), radfm `Evaluation > Evaluation on Public Benchmarks` for q026 (dataset descriptions, no results). The rejection counts per question are the rows of `adjudicated.jsonl`.

**Freeze.** `validate_golden` passes. Hash after: `cc08395a87ca30fc302b98065395f77404f192d3999f22428cab009b38022b0c` (v2 draft was `af6ce07a...64af`). The 13 runs were re-run on this file with `--no-llm` so every run record carries it. The LangSmith dataset sync is not part of T6 (the orchestrator handles it, it needs a committed file).

Golden set LLM-labelled (Claude), not human-verified. The same author wrote the questions and judged the retrieved sections, and a different or second reader would likely add or reject some of these.

### v2 amendment, T10 hybrid labelling pass (2026-11-20)

The two hybrid runs (`hybrid`, `hybrid_rerank`) put some sections into their top 5 that no label or earlier rejection covered. An unlabelled section counts as irrelevant, so these were read before the hybrid runs are compared. No question text, id, type or `source_chunks` entry changed. No label was removed or re-graded. The golden hash changed, so every run record must be re-run on the new file.

**Method.** The pool is the union of the top 5 chunks (no cut) of `hybrid` and `hybrid_rerank`, minus sections already labelled or rejected: 48 candidates over 45 questions (command: `pool hybrid hybrid_rerank --k 5`). The chunk text of every candidate was read from the `papers` collection (`sections --dump-text`, folder outside the repo) and judged against the question text only, by the Phase 9 and T6 standard: grade 2 fully answers, grade 1 holds part of the answer or clear support, same topic only is not added. Scores and ranks were not used. The rejections were written to `adjudicated.jsonl` with the same helper (`record_rejections`) as in T6, one reason per candidate, in one process (the `pool --reject` command checks the run hash against the golden file and the hybrid runs are on the old hash).

| Item | Value |
|---|---|
| Candidates read | 48 |
| Added at grade 2 / grade 1 | 0 / 2 |
| Rejected | 46 (each with a reason in `data/eval/adjudicated.jsonl`) |
| Total labels after | 161 (84 at grade 2, 77 at grade 1), was 159 |

**Additions.**

| Question | Type | Paper | Section | Grade | Reason |
|---|---|---|---|---|---|
| q022 | cross_paper | med_palm_m | Appendix > Details on Human Evaluation | 1 | States that the input to the model is a single image and the indication from a single study. Same kind of support as the labelled Appendix > Chest X-ray report generation. |
| q043 | multi_section | gaze2report | Introduction | 1 | States the contributions: visual-gaze tokens combined with generation instructions as the prompt, and evaluation on multiple datasets with NLG and clinical efficacy metrics. Does not name the datasets or metrics. |

**Notable rejections.** maira_2 `Additional qualitative examples > MAIRA-2 model inputs` for q012 is a banned false heading (example-figure label), so it could not be labelled in any case. conch `Results > Zero-shot classification of diverse tissues and diseases` for q002 names TCGA BRCA but gives no slide counts. deep_tumor_vqa `Accuracy Breakdown across Demographic and Imaging Factors` for q016 reports accuracy by subgroup but not a training method. gaze2report `Abstract` for q029 gives no F1 value. cxr_vlm_generation `Stable Diffusion Fine-Tuning > Training details` for q040 gives hardware and weight sources, not which parts were frozen. Most others were another paper or the same topic only.

**Hash.** Before: `cc08395a87ca30fc302b98065395f77404f192d3999f22428cab009b38022b0c`. After: `7f236dd41db41e38e4b5ae79a706e1265b4ca9bb98d151b2658a5beaad2fcd24`. `validate_golden` passes (65 questions, 161 labels).

Golden set LLM-labelled (Claude), not human-verified. The same author wrote the questions and judged the retrieved sections, and a second reader would likely add or reject some of these.

**Runs after this pass (2026-11-20).** Re-run on hash `7f236dd41db4`: `baseline`, `no_rewrite`, `prefix`, `bge_base`, `hybrid`, `hybrid_rerank`, `rerank_minilm`, `rerank_minilm_orig`, `rerank_minilm_d40`. Not re-run, still on `cc08395a87ca`: the six `rerank_bge*` and `rerank_medcpt*` runs. Those models are not used (ADR-0026), so the old records stay as historical evidence and cannot be compared with the new ones. The `*_v0` runs belong to golden set v0/v1 and are unchanged.

### v2 amendment, T11 chunk-size labelling pass (2026-11-20)

The two chunk-size runs (`chunk256_merge`, `chunk256_nomerge`, see ADR-0028 addendum) cut the papers differently, so they put some sections into their top 5 that no label or earlier rejection covered. An unlabelled section counts as irrelevant, so these were read before the runs were compared. No question text, id, type or `source_chunks` entry changed. No label was removed or re-graded. The golden hash changed, so every run record that is still used was re-run on the new file.

**Method.** The pool is the union of the top 5 chunks (no cut) of the two chunk-size runs, minus sections already labelled or rejected: 20 candidates over 45 questions (command: `pool chunk256_merge chunk256_nomerge --k 5`). The chunk text of every candidate was read from the `eval_chunk256_merge` collection (`sections --dump-text`, folder outside the repo) and judged against the question text only, by the same standard as before: grade 2 fully answers, grade 1 holds part of the answer or clear support, same topic only is not added. Scores and ranks were not used for the decision. The rejections were written to `adjudicated.jsonl` with `record_rejections`, one reason per candidate.

| Item | Value |
|---|---|
| Candidates read | 20 |
| Added at grade 2 / grade 1 | 0 / 3 |
| Rejected | 17 (each with a reason in `data/eval/adjudicated.jsonl`) |
| Total labels after | 164 (84 at grade 2, 80 at grade 1), was 161 |

**Additions.**

| Question | Type | Paper | Section | Grade | Reason |
|---|---|---|---|---|---|
| q023 | cross_paper | llava_med | Introduction | 1 | States that GPT-4 creates instructions from the text alone of image-text pairs. Summary-level support, like the labelled maira_2 and radfm sections at grade 1. |
| q023 | cross_paper | llava_med | Biomedical Visual Instruction-Following Data > Conversation | 1 | Figure 1 caption: the data is generated by GPT-4 from the caption and the figure's mentions only, without the image. Sub-section of a labelled grade 2 section. |
| q038 | multi_section | maira_2 | Discussion | 1 | Lists the inputs (frontal and lateral views, prior frontal image and full report, Indication, Technique, Comparison). Does not say how missing inputs are handled. |

**Notable rejections.** maira_2 `Results > MAIRA-2 establishes the new task of grounded reporting` for q006 gives RadFact scores on GR-Bench and PadChest-GR, not the Kendall correlation with ReXVal. intersectional_fairness_vlm `Introduction` for q032 frames the problem and gives no counts of missed malignancies. pathalign `Supplementary figures` for q021 holds per-rater result captions, not the rating setup. Most others were another paper or the same topic only.

**Hash.** Before: `7f236dd41db41e38e4b5ae79a706e1265b4ca9bb98d151b2658a5beaad2fcd24`. After: `0c62886fd4897ee84bacf8a776c44fa319ae12ef5a8f3c8338254e598467b097`. `validate_golden` passes (65 questions, 164 labels).

Golden set LLM-labelled (Claude), not human-verified. The same author wrote the questions and judged the retrieved sections, and a second reader would likely add or reject some of these.

**Runs after this pass (2026-11-20).** Re-run on hash `0c62886fd489`: `baseline`, `no_rewrite`, `prefix`, `bge_base`, `hybrid`, `hybrid_rerank`, `rerank_minilm`, `rerank_minilm_orig`, `rerank_minilm_d40`, plus the new `chunk256_merge` and `chunk256_nomerge`. Not re-run, still on `cc08395a87ca`: the six `rerank_bge*` and `rerank_medcpt*` runs (owner decision, those models are not used). The numbers quoted in ADR-0025 and ADR-0026 were measured on earlier hashes and are not changed; the re-run records differ from them (for example `baseline` nDCG@5 0.537 now, 0.525 in ADR-0025; `rerank_minilm_d40` 0.724 now, 0.725 in ADR-0026), so compare numbers only within one hash. The `*_v0` runs belong to golden set v0/v1 and are unchanged.
