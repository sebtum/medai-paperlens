# T8: The 50 questions (golden set v0)

**Status:** done. 50 questions with 142 labels, all checks pass. It is "LLM-labelled (Claude), not human-verified": one author made it.

| Check | Result |
|---|---|
| Counts per type (10 / 10 / 8 / 8 / 9 / 5) and every paper in at least 3 questions | Pass (smallest paper has 4) |
| Validator `validate_golden` | 0 violations |
| Live pre-flight against `papers` (595 chunks) | Pass, re-run by the orchestrating session |
| No paraphrased question shares 4 or more words in a row with its source | Pass (worst: 3) |
| No question flagged unsafe by `classify_node` | Pass |
| Tests, ruff, mypy, bandit | 428 passed, all clean (re-run by the orchestrating session) |
| Verify pass | 10 questions got extra labels, 0 rewritten, 0 dropped |

Files added: `data/eval/golden.jsonl`, `tests/test_golden_set.py` (4 tests). No run files, no `rewrites.jsonl`, no LangSmith sync.

Words used below:
- **Chunk:** one stored piece of a paper (id like `maira_2:12`).
- **Section label:** the `Section > Subsection` text stored with a chunk. Our labels point at these.
- **Grade:** 2 = the section answers the question directly. 1 = it holds part of the answer or clear support.
- **Pooling:** the later step (T9) where we look at what the search returned that we did not label.

## Section: How the set was built

**What it is:** two separate passes. First a draft, then a skeptical re-read.

**Draft:**
1. All chunk text of the 10 papers was dumped to a folder outside the repo, so it can never be committed.
2. The agent read the main text, methods and discussion of every paper. It skimmed long number tables and appendices by section name and by search.
3. Each question was written from a concrete passage, and that chunk was listed in `source_chunks`.
4. Difficulty is spread. Many answers sit in hard-to-find sections, for example `RadFact metric > Implementation details` (q006, q034) or `Online Methods > Downstream evaluation datasets` (q002).
5. Off-corpus questions: the agent searched all 10 papers for the key terms (federated, conformal, adversarial, quantisation, pruning, latency, cost-effectiveness). None is answered in the corpus.

**Verify pass:** for each question, ask: does each labelled section really answer it? Is a grade 2 complete? Is another section answering but unlabelled? Is the question ambiguous?

**Example:** q044 (LLaVA-Med training stages) first had two labels. Verify added the Abstract at grade 1, because it also states the two stages and the training time.

**Result:** 10 labels added, 0 removed, 0 questions rewritten or dropped.

**Not checked:** the verify pass used the same author and the same reading of the papers. A shared blind spot would not be caught. Pooling in T9 is the next check.

## Section: Counts

| Type | Drafted | Got extra labels in verify | Rewritten | Dropped | Final |
|---|---|---|---|---|---|
| factual | 10 | 0 | 0 | 0 | 10 |
| paraphrased | 10 | 3 | 0 | 0 | 10 |
| cross_paper | 8 | 3 | 0 | 0 | 8 |
| table | 8 | 0 | 0 | 0 | 8 |
| multi_section | 9 | 4 | 0 | 0 | 9 |
| off_corpus | 5 | 0 | 0 | 0 | 5 |
| Total | 50 | 10 | 0 | 0 | 50 |

Labels: 132 in the draft, 142 after verify (83 at grade 2, 59 at grade 1).

Questions that label each paper: conch 4, cxr_vlm_generation 5, deep_tumor_vqa 6, gaze2report 5, intersectional_fairness_vlm 5, llava_med 6, maira_2 9, med_palm_m 9, pathalign 6, radfm 8.

## Section: Copy detector results

**What it is:** the longest run of words in a row that a question shares with its source chunk. A paraphrased question with 4 or more must be rewritten.

| Type | Worst overlap (words) | Questions |
|---|---|---|
| paraphrased | 3 | q011, q014, q016 |
| factual | 4 | q009 |
| cross_paper | 4 | q022, q027 |
| table | 5 | q035 |
| multi_section | 3 | q037, q040, q043, q044, q045 |
| off_corpus | 0 | all |

**Example:** q035 (table type) shares 5 words. That is fine. Only paraphrased questions have a limit.

**Result:** none of the paraphrased questions reaches 4. None was rewritten.

**Not checked:** single shared words and synonyms. A question can be close to the source in meaning without sharing 4 words in a row.

## Section: Pre-flight against the live database

**What it is:** a scan of the whole Qdrant collection. It checks that every labelled section and source chunk exists, that table questions have a table chunk, and that the embedding model matches.

**Real output (re-run by the orchestrating session):**

```
INFO:app.evaluation.runner:Pre-flight passed: 595 chunks in 'papers', 50 golden questions
Pre-flight passed for baseline: 50 questions, 595 chunks in 'papers'.
```

**Result:** the collection has 595 chunks (415 text, 165 table, 15 caption), all from `BAAI/bge-small-en-v1.5`. Read-only use, no container touched.

**Not checked:** no search was run, so there are no scores yet.

## Section: Known weak spots

| Weak spot | Detail |
|---|---|
| The `maira_2` banned-label rule (T2's guess) | Too narrow, not too strict. It matches no real section name, so it bans nothing. |
| Odd `maira_2` sections the rule misses | `Qualitative evaluation of twenty random ... > Workflow overview`, `Sample-level quality control`, `Options`, `Phrase labels`, and similar (an annotation protocol under a wrong parent). Also `Additional qualitative examples > FINDINGS phrases`, `Current Study`, `MAIRA-2 model inputs`. None is labelled. |
| Wrongly named sections that hold real content (these are labelled) | radfm `Dataset > Prompting ChatGPT ...` (a figure caption used as a heading), used in q023. radfm `Results > Qualitative Results` (holds Table 8), q026. med_palm_m `Appendix > Detailed Med-PaLM M Performance` (Table A.1), q039. pathalign `Supplementary tables` (Table C.3), q045. |
| A banned section that does answer something | llava_med `Experiments > Visual input example, ...` holds the chatbot results. No question needs it. |
| conch has no table chunks | So no table question uses conch. |
| gaze2report has only 7 sections | Almost any search hits some label. q001, q017, q029, q043 are easy. |
| Scrambled table chunks | Some tables have broken cell order (cxr 27-34, deep_tumor 27). Only readable ones were used. |
| One labeller | A systematic error would not be caught. |

**Example:** q023 labels radfm `Dataset > Prompting ChatGPT ...` at grade 2. The heading is wrong, but the chunk really holds the ChatGPT prompt.

## Section: Decision for you

**What is decided:** what to do about the misleading section names above.

| Option | In practice | Recommendation |
|---|---|---|
| A. Leave the validator as is | v0 stays simple. T9 pooling shows whether the search returns these odd sections. | Recommended for v0 |
| B. Add the real odd `maira_2` and `radfm` names to the banned list now | Cleaner labels. Needs a small change in `app/evaluation/golden.py` and its tests. | Only if T9 pooling shows these sections |

## Section: Spot-check list (10 questions the author is least sure about)

You said you will not review by hand. If you ever do, start here. The question text is in the table below.

| Question | Why unsure |
|---|---|
| q027 | Broad: seven labels over four papers. deep_tumor `Introduction` at grade 2 is only a short remark about BLEU and ROUGE. |
| q021 | Seven labels. Does each paper really describe "how the rating was organised"? |
| q023 | Labels a false heading in radfm. |
| q026 | The radfm label sits in a section named `Qualitative Results` that holds a table. |
| q022 | Loose test. The radfm and med_palm appendix labels are grade 1 on weak evidence. |
| q013 | The disease is not named. "How close to a specialist" lives in a table chunk. |
| q039 | Mixes four sections, one of which holds Table A.1 under a different heading. |
| q012 | Four labels, two at grade 1 on indirect support. |
| q017 | The gaze2report Abstract at grade 2 makes this easy to hit. |
| q029 | The answer needs a comparison with another method in the same table. |

## Section: What T9 needs from this

- `data/eval/golden.jsonl` is v0. Any edit changes its hash and invalidates all runs and rewrites. Edit it only through pooling in T9.
- `run --config baseline` needs Ollama up, to create `data/eval/rewrites.jsonl` (not created yet).
- For `bge_base`, ingest first (see the T7 report).
- When pooling, look first at the 10 spot-check questions, the 5 off-corpus questions (a hit above the cut-off there is a real failure), and the odd sections listed above.
- LangSmith sync (`scripts.sync_golden`) was not run. The key is now in `.env`, so T9 can do it.
- The build script is in the scratchpad, not the repo. `golden.jsonl` is the source of truth.

## Section: Gates

| Gate | Result |
|---|---|
| `pytest tests/` | 428 passed (424 before, plus 4 new) |
| `ruff check .` | All checks passed |
| `mypy app/` | no issues in 42 files |
| `bandit -r app/` | no issues |
| `pip-audit` | not run (no dependency touched) |

`tests/test_golden_set.py` loads the real file and manifests. It checks 0 violations, the quotas, that `classify_node` flags no question, and that off-corpus questions have empty labels. It needs no Qdrant, so CI can run it.

## Section: The full question list

Our own question text and section names only. Format: `paper_id: section (grade)`.


| Id | Type | Question | Labelled sections |
|---|---|---|---|
| q001 | factual | What beam size does Gaze2Report use for beam search when it generates reports at test time? | gaze2report: Experiment Design and Results (2) |
| q002 | factual | How many slides are in the TCGA BRCA dataset that CONCH uses, and how many of them are invasive ductal versus invasive lobular carcinoma? | conch: Online Methods > Downstream evaluation datasets (2) |
| q003 | factual | How many questions make up the evaluation set for the GPT-4-judged chatbot test of LLaVA-Med, and how many are conversation versus detailed-description questions? | llava_med: Experiments > Biomedical Visual Chatbot (2) |
| q004 | factual | By how much does cropping around the organ with nnUNet raise kidney tumor sensitivity for the LLaMA2 version of M3D in DeepTumorVQA? | deep_tumor_vqa: Evaluation on DeepTumorVQA Benchmark > Improving Lesion Recognition via Segmentation-based Preprocessing (2) |
| q005 | factual | Which value of the fairness regularisation weight lambda was set for CMAC-MMD, and how was it chosen? | intersectional_fairness_vlm: Methods > Experimental Design and Baseline Comparisons (2) |
| q006 | factual | What Kendall rank correlation did RadFact's logical F1 reach with radiologist error counts on the ReXVal dataset? | maira_2: RadFact metric > Implementation details (2) |
| q007 | factual | Which optimizer, momentum and dropout rate were used when finetuning Med-PaLM M? | med_palm_m: Med-PaLM M: A Proof of Concept for Generalist Biomedical AI > Putting it all together: Med-PaLM M (2) |
| q008 | factual | How many whole slide images were chosen for the pathologist evaluation of PathAlign, and how were they split between common and other specimen types? | pathalign: Supplemental methods > Evaluation details (2); pathalign: Methods > Evaluation (1) |
| q009 | factual | On what hardware and for how long did RoentGen fine-tune for 60k training steps? | cxr_vlm_generation: Stable Diffusion Fine-Tuning > Training details (2) |
| q010 | factual | Which language model is used to initialize RadFM, and what is the total parameter count of the final model? | radfm: Building Generalist Foundation Model for Radiology > Training Procedure (2); radfm: Discussion > Limitations (1) |
| q011 | paraphrased | How can a pathology image-and-text model turn tile-by-tile scores into one prediction for an entire gigapixel scan when it has no labelled training slides? | conch: Online Methods > Extending zero-shot transfer to WSIs (2); conch: Results > Zero-shot classification of diverse tissues and diseases (2); conch: Results > Few-shot classification with task-specific supervised learning (1) |
| q012 | paraphrased | Does supplying an automated chest radiograph report writer with the patient's earlier exam and its report reduce invented references to past comparisons? | maira_2: Results > Prior studies reduce temporal hallucinations (2); maira_2: Extended results > Further ablations on additional inputs (1); maira_2: Discussion (1); maira_2: Extended background and related work > Why do we expect additional inputs to help? (1) |
| q013 | paraphrased | Can one multi-task medical AI model recognise an infectious lung disease on chest radiographs that it was never trained to label, and how close does it get to a purpose-built specialist? | med_palm_m: Results > Med-PaLM M demonstrates zero-shot generalization to novel medical tasks and concepts (2); med_palm_m: Evaluation > Evaluation of language enabled zero-shot generalization (1) |
| q014 | paraphrased | Does swapping the stock language component of a text-to-image diffusion generator for one pretrained on radiology reports improve chest radiograph synthesis? | cxr_vlm_generation: Fidelity and diversity of generated images (2); cxr_vlm_generation: Introduction (1); cxr_vlm_generation: Conclusion and future work (1); cxr_vlm_generation: Factual correctness of generated images > Zero-shot Image-Text Retrieval (1) |
| q015 | paraphrased | For 3D scan question answering by vision-language assistants, is accuracy higher when answer options are offered than when the answer must be typed out freely? | deep_tumor_vqa: Evaluation on DeepTumorVQA Benchmark > Analysis of Benchmarking Results (2); deep_tumor_vqa: Evaluation on DeepTumorVQA Benchmark > Details of VLM Evaluation (1) |
| q016 | paraphrased | How can a medical image-text classifier be trained so that its confidence is equally dependable for patients of different ages, sexes and ethnicities, while not needing those personal attributes when deployed? | intersectional_fairness_vlm: Methods > The Cross-Modal Alignment Consistency (CMAC-MMD) Framework (2); intersectional_fairness_vlm: Abstract (1); intersectional_fairness_vlm: Introduction (1); intersectional_fairness_vlm: Discussion (1); intersectional_fairness_vlm: Methods > Study Design and Clinical Cohort Selection (1) |
| q017 | paraphrased | How can a chest radiograph report writer make use of where radiologists looked when reading, even though no eye-tracking is available for new images at deployment? | gaze2report: Abstract (2); gaze2report: Methodology (2); gaze2report: Introduction (1); gaze2report: Experiment Design and Results (1); gaze2report: Conclusion (1) |
| q018 | paraphrased | Why is it difficult to pair each digitised tissue slide with the right portion of its written diagnosis? | pathalign: Methods > Curating image-text pairs (2); pathalign: Introduction (1); pathalign: Discussion > Limitations (1); pathalign: Supplemental methods > Creating image-text pairs (1); pathalign: Abstract (1) |
| q019 | paraphrased | How can instruction data for a picture-based chat assistant in the life sciences be created cheaply, without people writing the instructions? | llava_med: Biomedical Visual Instruction-Following Data (2); llava_med: Introduction (1); llava_med: Abstract (1); llava_med: Prompts > Prompting GPT-4 to generate medical visual instruction-following data (1) |
| q020 | paraphrased | How can a single imaging network be built to take in both ordinary flat radiographs and volumetric CT or MRI scans? | radfm: Building Generalist Foundation Model for Radiology > Architecture Detail (2); radfm: Discussion > RadFM is the first foundation model that unified 2D and 3D radiologic images (2); radfm: Building Generalist Foundation Model for Radiology > Training Procedure (1) |
| q021 | cross_paper | Which studies had radiologists or pathologists rate generated text, and how was the rating organised (scale, blinding, side-by-side or independent)? | med_palm_m: Evaluation > Clinician evaluation of radiology report generation (2); radfm: Evaluation > Evaluation on RadBench (2); pathalign: Methods > Evaluation (2); pathalign: Supplemental methods > Evaluation details (2); cxr_vlm_generation: Factual correctness of generated images > Qualitative evaluation (2); maira_2: Results > Expert review reveals areas of strength and weakness (1); med_palm_m: Appendix > Details on Human Evaluation (1) |
| q022 | cross_paper | Which multimodal chest X-ray report systems feed clinical context such as the study indication or imaging view into the model alongside the image? | maira_2: Methods > Incorporating additional context (2); med_palm_m: Med-PaLM M: A Proof of Concept for Generalist Biomedical AI > Putting it all together: Med-PaLM M (2); radfm: Discussion > Clinical impact (1); med_palm_m: Appendix > MultiMedBench (1) |
| q023 | cross_paper | Which papers use GPT-4 or ChatGPT to automatically turn existing text into training data, and what exactly do they prompt the model to produce? | llava_med: Biomedical Visual Instruction-Following Data (2); llava_med: Prompts > Prompting GPT-4 to generate medical visual instruction-following data (2); radfm: Dataset > Prompting ChatGPT to generate medical visual question answering data for Rad3D (2); radfm: Dataset > Medical Multimodal Dataset (MedMD) (1); maira_2: Extended methods > Rules (2); maira_2: Extended methods > Preparation of grounded reporting data (1) |
| q024 | cross_paper | Which studies break down model accuracy by patient subgroups such as age, sex or scanner, and what patterns do they find? | deep_tumor_vqa: Accuracy Breakdown across Demographic and Imaging Factors (2); intersectional_fairness_vlm: Results > Cohort Characteristics and the Source of Diagnostic Disparities (2); intersectional_fairness_vlm: Results > CMAC-MMD Improves Diagnostic Performance While Reducing Missed Diagnosis Disparities in Dermatology (2); intersectional_fairness_vlm: Results > CMAC-MMD Demonstrates Cross-Domain Generalisability in Glaucoma Detection (2) |
| q025 | cross_paper | How do the two histopathology image-text models assign a whole slide to a disease class using text prompts? | conch: Online Methods > Extending zero-shot transfer to WSIs (2); conch: Results > Zero-shot classification of diverse tissues and diseases (1); conch: Online Methods > Zero-shot transfer on ROIs/tiles (1); conch: Results > Few-shot classification with task-specific supervised learning (1); pathalign: Methods > Evaluation (2); pathalign: Supplemental methods > Evaluation details (2) |
| q026 | cross_paper | Which models report results on the VQA-RAD, SLAKE or PathVQA benchmarks, and how do they compare with earlier supervised methods? | llava_med: Experiments > Performance on Established Benchmarks (2); med_palm_m: Results > Med-PaLM M performs near or exceeding SOTA on all MultiMedBench tasks (2); med_palm_m: Appendix > Detailed Med-PaLM M Performance (2); radfm: Results > Transferring Ability to Public Benchmarks (2); radfm: Results > Qualitative Results (1) |
| q027 | cross_paper | Which papers argue that word-overlap metrics like BLEU and ROUGE are inadequate for judging medical text, and what do they use instead? | maira_2: Methods > RadFact: An evaluation suite for (grounded) reports (2); med_palm_m: Evaluation (2); radfm: Discussion > Limitations (2); radfm: Evaluation > Evaluation on RadBench (1); radfm: Discussion > A general evaluation benchmark for radiology foundation models (1); deep_tumor_vqa: Introduction (2); med_palm_m: Appendix > Detailed Med-PaLM M Performance (1) |
| q028 | cross_paper | Which pretrained language model and image encoder does each radiology report generator start from? | gaze2report: Methodology (2); gaze2report: Experiment Design and Results (2); maira_2: Methods > MAIRA-2 architecture (2); radfm: Building Generalist Foundation Model for Radiology > Training Procedure (2); radfm: Building Generalist Foundation Model for Radiology > Architecture Detail (1); med_palm_m: Med-PaLM M: A Proof of Concept for Generalist Biomedical AI > Model preliminaries (2) |
| q029 | table | On MIMIC-CXR, what F1 does the gaze-guided model reach in the clinical-efficacy comparison, and which compared method scores higher? | gaze2report: Experiment Design and Results (2) |
| q030 | table | Which vision-encoder initialisation gave the highest average across the three medical VQA benchmarks after fine-tuning, and what was that average? | llava_med: Experiments > Performance on Established Benchmarks (2) |
| q031 | table | In the 3D CT question-answering benchmark, what were the total average multi-choice and free-text scores of the best-performing model and of random guessing? | deep_tumor_vqa: Evaluation on DeepTumorVQA Benchmark > Details of VLM Evaluation (2); deep_tumor_vqa: Evaluation on DeepTumorVQA Benchmark > Analysis of Benchmarking Results (1) |
| q032 | table | How many missed malignancies did the held-out dermatology test set contain under standard training versus the fairness method, and how many were prevented? | intersectional_fairness_vlm: Results > CMAC-MMD Improves Diagnostic Performance While Reducing Missed Diagnosis Disparities in Dermatology (2); intersectional_fairness_vlm: Discussion (1) |
| q033 | table | How did classifier AUROC compare when trained on 30k real, 30k synthetic, or a mix of both chest radiographs? | cxr_vlm_generation: Data augmentation (2); cxr_vlm_generation: Introduction (1) |
| q034 | table | How accurate and how fast was the open Llama3-70B compared with GPT-4 as the sentence-level entailment checker? | maira_2: RadFact metric > Implementation details (2) |
| q035 | table | When the classification task was dropped from training, how did the 84B model's chest X-ray report generation scores change? | med_palm_m: Results > Med-PaLM M performs encouragingly on radiology report generation across model scales (2); med_palm_m: Results > Med-PaLM M demonstrates zero-shot generalization to novel medical tasks and concepts (2) |
| q036 | table | What AUC and balanced accuracy were obtained when classifying slides as biopsy versus resection? | pathalign: Methods > Evaluation (2); pathalign: Results (1); pathalign: Supplemental methods > Evaluation details (1) |
| q037 | multi_section | How does CONCH tile gigapixel slides and use the tiles for both slide-level classification and tumor segmentation? | conch: Online Methods > WSI processing (2); conch: Online Methods > Extending zero-shot transfer to WSIs (2); conch: Results > Zero-shot segmentation (2); conch: Results > Zero-shot classification of diverse tissues and diseases (1); conch: Results > Few-shot classification with task-specific supervised learning (1) |
| q038 | multi_section | Which images and report sections does MAIRA-2 take as input, and how are missing inputs handled in its prompt? | maira_2: Methods > Incorporating additional context (2); maira_2: Extended methods > Additional MAIRA-2 model and training details (2); maira_2: Introduction (1); maira_2: Methods > Data (1) |
| q039 | multi_section | How was the multitask training mixture put together from the benchmark's datasets, and how were the task mixture ratios chosen? | med_palm_m: Med-PaLM M: A Proof of Concept for Generalist Biomedical AI > Putting it all together: Med-PaLM M (2); med_palm_m: Appendix > Med-PaLM M Training Details (2); med_palm_m: MultiMedBench: A Benchmark for Generalist Biomedical AI (1); med_palm_m: Appendix > Detailed Med-PaLM M Performance (1) |
| q040 | multi_section | Which parts of the diffusion pipeline were frozen, fine-tuned or retrained from scratch in the compared adaptation strategies, and which choice gave the best image fidelity? | cxr_vlm_generation: Stable Diffusion Fine-Tuning > Fine-tuning and training from scratch (2); cxr_vlm_generation: Fidelity and diversity of generated images (2); cxr_vlm_generation: Stable Diffusion Fine-Tuning > Few-shot fine-tuning (1); cxr_vlm_generation: Conclusion and future work (1); cxr_vlm_generation: Introduction (1) |
| q041 | multi_section | How are the question types of DeepTumorVQA defined and generated, and from what extracted metadata? | deep_tumor_vqa: DeepTumorVQA Dataset > Dataset Construction (2); deep_tumor_vqa: Metadata and Structured Description Generation > Metadata Extraction from Segmentation Masks (2); deep_tumor_vqa: Task Definitions and Generation Logic (2); deep_tumor_vqa: DeepTumorVQA Dataset > Overview (1); deep_tumor_vqa: Metadata and Structured Description Generation > Structured Report-style Description (1) |
| q042 | multi_section | Which datasets were chosen for the intersectional fairness study, why were the others rejected, and how were the cohorts split into subgroups and train, validation and test sets? | intersectional_fairness_vlm: Methods > Study Design and Clinical Cohort Selection (2); intersectional_fairness_vlm: Results > Cohort Characteristics and the Source of Diagnostic Disparities (2) |
| q043 | multi_section | How does the model turn a chest X-ray plus predicted gaze into a prompt for the language model, and which metrics and datasets are used to evaluate the generated reports? | gaze2report: Methodology (2); gaze2report: Experiment Design and Results (2); gaze2report: Abstract (1) |
| q044 | multi_section | What are the two training stages of LLaVA-Med, how long does each take, and what do the ablations say about training longer or on more data? | llava_med: Adapting Multimodal Conversational Models to the Biomedical Domain (2); llava_med: Experiments > Performance on Established Benchmarks (2); llava_med: Abstract (1) |
| q045 | multi_section | How do the retrieval and text-generation variants of PathAlign differ in architecture and training setup, including the number of query vectors and loss terms? | pathalign: Methods > Modeling (2); pathalign: Supplemental methods > Modeling: image-text alignment (2); pathalign: Supplementary tables (1) |
| q046 | off_corpus | What evidence is there that federated learning across hospitals can train medical vision-language models without pooling patient images centrally? | none (off corpus) |
| q047 | off_corpus | How do conformal prediction or similar methods provide coverage guarantees on the uncertainty of medical vision-language model answers? | none (off corpus) |
| q048 | off_corpus | How vulnerable are medical vision-language models to adversarial image perturbations or prompt-injection attacks, and which defences have been proposed? | none (off corpus) |
| q049 | off_corpus | How much do post-training quantisation or pruning cut the memory use and inference latency of medical multimodal models on hospital hardware? | none (off corpus) |
| q050 | off_corpus | What health-economic analyses show the cost-effectiveness of deploying AI triage for chest radiographs in hospitals? | none (off corpus) |
