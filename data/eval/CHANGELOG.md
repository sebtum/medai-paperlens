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
