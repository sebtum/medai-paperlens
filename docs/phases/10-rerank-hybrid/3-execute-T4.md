# T4: Golden set v2, 15 near-miss questions

**Status:** done. All four acceptance criteria pass. No question had to be dropped or rewritten. The golden set stays a draft until T6 freezes it, and it is still LLM-labelled (Claude), not human-verified.

## What

I added 15 off-corpus questions (q051 to q065) to the golden set. An off-corpus question looks like it belongs to the 10 papers but is not answered by them. The search should find nothing useful for it, so these questions test later whether a reranker score can refuse them.

I wrote them before any reranker result existed. I did not look at or run a reranker. I did not change a question because of its dense score.

- Details: touched `data/eval/golden.jsonl` (+15 lines), `data/eval/rewrites.jsonl` (+15 lines), `data/eval/CHANGELOG.md` (v2 draft entry), `app/evaluation/golden.py` (`QUOTAS` off_corpus 5 to 20, total 50 to 65), `tests/test_golden_set.py` and `tests/test_eval_golden.py` (65 entries, quota message 20/19).
- Details: kinds of near miss: same modality with another task (q053, q055, q056, q057, q063), a model the corpus does not cover (q051, q054, q063), a modality, dataset or setting the corpus does not use (q052, q062, q065, q061), a topic next to a covered paper (q058, q059, q060, q064). Phrasing varies: q057 and q065 are terse, q058 is long.

## Example

| Question | Why it is a near miss |
|---|---|
| q051: "How well does GPT-4V detect diabetic retinopathy in retinal fundus photographs?" | The corpus does mention GPT-4V (RadFM compares with it, on radiology). One paper also uses fundus photos (the fairness paper, for glaucoma). No paper tests GPT-4V on retinal images or looks at diabetic retinopathy. |
| q055: "How do medical vision-language models perform on the BraTS brain tumour segmentation benchmark?" | The corpus has tumour work (DeepTumorVQA, abdominal CT) and mentions segmentation. It does not use BraTS or brain MRI, and it answers questions about images, it does not segment. The top 10 are all DeepTumorVQA sections. |
| q058: "Is there any evidence that fine-tuning Stable Diffusion on skin lesion photographs gives realistic dermatology images, and how would a team check that the generated images are not copies of real patients' photos?" | One paper fine-tunes Stable Diffusion, but on chest X-rays. Another paper uses skin lesion photos, but for disease classification. Neither does both, and neither checks for copied images. |

## Result

| Acceptance criterion | Result |
|---|---|
| Entries q051 to q065, type `off_corpus`. Quota 5 to 20, total 50 to 65. Tests updated | Pass |
| Each checked against the corpus (dense top-10 sections read, no answer) | Pass: 15 of 15 survive. Dropped: 0. Rewritten: 0. |
| None flagged by the safety filter. The 15 rewrites come from Ollama | Pass: 0 flagged of 65. 15 new rewrites from `qwen3:0.6b`, no fallback, none starts with "Rewritten question:" |
| Logged in `CHANGELOG.md` as v2 draft | Pass, hash change noted, not frozen |

Gates:

| Gate | Result |
|---|---|
| Full `pytest tests/` | 590 passed |
| `ruff check` and `ruff format --check` on the 3 touched Python files | clean |
| Safety filter on all 65 questions (`classify_node`), also covered by `test_no_question_is_flagged_unsafe` | 0 flagged |

- Details: golden hash v1 `4a4733e5...2649` (50 questions) to v2 draft `af6ce07a...64af` (65 questions). Line endings are LF (no CR in either file).
- Details: corpus check. I ran `run --config baseline --name scratch_v2_rewrites --skip-validate` (no `--no-llm`, so Ollama wrote the rewrites; `--skip-validate` because the quota was still 5 at that point). I listed the dense top 10 with `pool scratch_v2_rewrites --k 10 --threshold 0 --include-off-corpus --adjudicated <empty temp file>` (so the real `adjudicated.jsonl` was not touched and does not exist yet). I read the chunk text from the `papers` collection (read-only). All 150 hits (15 questions by 10) were listed by section. Every chunk that contained a key term of its question was read in context (for example GPT-4V, fundus, segmentation, skin, race, hallucination, polyp, UNI, eye). Chunks with no key term were judged by section title and score only. The scratch run and report files were deleted afterwards.
- Details: rewrites. `rewrites.jsonl` now has 65 lines. The first 50 lines are byte-identical to before (checked with `cmp`). The first 50 lines of `golden.jsonl` are also byte-identical. The cache has no stored rewrite that starts with "Rewritten question:".
- Details: the plan example "How does GPT-4 perform on dermatology images?" was not used. The Med-PaLM M paper covers dermatology, so the corpus partly answers it. For the same reason I avoided mammography, genomic variant calling, text question answering with Med-PaLM 2, and comparisons with BiomedCLIP (CONCH compares against it).

## Not checked

- The same author (Claude) wrote the questions and read the hits to decide that the corpus has no answer. That is not an independent check. A human may find a section that partly answers q056 or q064 (LLaVA-Med and MAIRA-2 mention nearby ideas such as hallucination reduction from extra inputs), or q059 (the fairness paper discusses bias for classification, not for report generation). I judged these as no answer, because none addresses the asked task.
- Only the dense top 10 was read. A section outside the top 10 could answer a question. I also did not read chunks with no key term in full.
- I did not run any reranker, as required, so how these questions behave under reranking is not known yet. The question set is the same for later labelling in T6, where the same author labels hits. All numbers stay LLM-labelled, not human-verified.
- The questions use the model's knowledge from before November 2026 (for example MedGemma). A paper added to the corpus later could change an answer.
- Small rewrite model: `qwen3:0.6b` kept most questions nearly unchanged (q065 and q060 became shorter). That is how the app behaves, so it is what the eval measures.

## Changed files

- `data/eval/golden.jsonl`
- `data/eval/rewrites.jsonl`
- `data/eval/CHANGELOG.md`
- `app/evaluation/golden.py`
- `tests/test_golden_set.py`
- `tests/test_eval_golden.py`
- `docs/phases/10-rerank-hybrid/3-execute-T4.md` (this report)
