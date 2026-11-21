# T10 labels: labelling pass for the hybrid runs

**Status:** done. 48 candidate sections read, 2 got a label, 46 were rejected. The golden file changed, so the hybrid runs (and every other run) must be re-run on it. The golden set is LLM-labelled (Claude), not human-verified.

## What

The two hybrid runs showed 48 sections in their top 5 that nobody had judged yet. A section without a label counts as wrong in the metrics, which would punish the hybrid setups unfairly. I read the text of each one and judged it against the question text only (no scores, no ranks), with the same strict rules as the earlier labelling passes: grade 2 fully answers, grade 1 holds part of the answer, same topic only is not enough.

- Details: input was the pool list of 48 candidates over 45 questions. Chunk text came from `sections --collection papers --dump-text` into a folder outside the repo (not committed). Labels went into `data/eval/golden.jsonl` (two lines changed, LF endings kept). The 46 rejections went into `data/eval/adjudicated.jsonl` (232 rows before, 278 after), each with its own reason, written with `record_rejections` in one process. I did not call `pool --reject` itself: it checks that the run was made on the current golden hash, and after the two labels were added the hybrid runs no longer matched, so every call failed (about 19 s each).

## Example

q043 asks how Gaze2Report turns the X-ray plus gaze into a prompt and which metrics and datasets evaluate it. The Gaze2Report `Introduction` says the visual-gaze tokens are combined with generation instructions, and that evaluation uses several datasets with language-quality and clinical-efficacy metrics. That is part of the answer, so grade 1. Most rejections look like q029: the Gaze2Report `Abstract` talks about the method but gives no F1 value, so it does not answer the table question.

## Result

| Measure | Value |
|---|---|
| Candidates read | 48 |
| Added (grade 2 / grade 1) | 0 / 2 |
| Rejected | 46 |
| Labels after | 161 (84 at grade 2, 77 at grade 1), was 159 |

The two labels: q022 med_palm_m `Appendix > Details on Human Evaluation` (the model input is one image plus the study indication) and q043 gaze2report `Introduction`. One rejection was forced by a rule: maira_2 `Additional qualitative examples > MAIRA-2 model inputs` for q012 is a banned false heading. No candidate that deserved a label was blocked by a banned heading. I was unsure about fewer than 8 candidates (the close calls were q002 conch, q016 deep_tumor_vqa, q027 conch, q029 gaze2report and q040 cxr_vlm_generation, all rejected for missing the asked fact).

- Details: golden hash before `cc08395a87ca30fc302b98065395f77404f192d3999f22428cab009b38022b0c`, after `7f236dd41db41e38e4b5ae79a706e1265b4ca9bb98d151b2658a5beaad2fcd24`. `validate_golden` returns no violations (65 questions, 161 labels); `pytest tests/test_eval_golden.py`: 52 passed. A check script confirmed all 48 candidates are now either a label or a rejection row. `data/eval/CHANGELOG.md` has the new entry "v2 amendment, T10 hybrid labelling pass".
- Details: the labels are all grade 1, so they raise the ceiling slightly for hybrid runs that retrieved these two sections. Dense and reranked runs gain only where they also retrieved them. Not measured separately.

## Not checked

- The golden set is LLM-labelled (Claude), not human-verified. The same model wrote the questions and judged the text, so this is not an independent pass.
- Pooling only sees sections that the two hybrid runs put in their top 5. A labelled-worthy section that neither retrieved is still missing.
- I did not run the pool command afterwards to show 0 candidates left: it refuses because the saved runs carry the old hash. The check script gave the same answer for the 48 candidates. The orchestrator needs to re-run all runs on the new hash (and later sync LangSmith from a committed file).
- Nothing is committed; no run was re-run.

## Changed files

- `data/eval/golden.jsonl` (2 labels), `data/eval/adjudicated.jsonl` (46 rows), `data/eval/CHANGELOG.md` (new entry), `docs/phases/10-rerank-hybrid/3-execute-T10-labels.md` (this file)
