# T6: Reranker runs, labelling pass, freeze v2

**Status:** done. All three acceptance criteria pass. 13 runs are saved on the frozen golden set (65 questions, 159 labels). The pooling pass added 9 labels and rejected 232 candidates. Everything comes from a golden set that is LLM-labelled (Claude), not human-verified.

## What

I ran the three rerankers in three variants each (9 runs), plus the four dense setups again (4 runs), all with `--no-llm` and one after the other. Then I read every section that the nine reranked runs put into their top 5 and that no label covered. Nine sections got a label. The rest were rejected and logged. Then I froze golden set v2 and ran all 13 again so every run carries the final file hash.

A reranker is a second model that re-scores the top hits and keeps the best 5. "Depth" is how many hits the reranker sees (20 or 40). "Original" means the reranker reads the user's question, not the rewritten one.

- Details: commands, all via `.\.venv\Scripts\python.exe -m scripts.eval_retrieval`: `run --config rerank_<m> --no-llm`, `run --config rerank_<m> --name rerank_<m>_orig --rerank-query original --no-llm`, `run --config rerank_<m> --name rerank_<m>_d40 --depth 40 --no-llm` for minilm, bge, medcpt; `run --config <dense> --no-llm` for baseline, no_rewrite, prefix, bge_base. `--name` is required with `--depth` and `--rerank-query` (as T5 noted). Preflight passed (595 chunks, no stale labels, embedding model matches). Qdrant was only read.
- Details: the whole set ran twice: first on the v2 draft (to build the pool), then again on the frozen file. Machine was otherwise idle for the timed runs. The first pass and the final pass gave very similar times (the final numbers are in the table).

## Example

A label that was added: q022 asks which chest X-ray report systems feed clinical context (indication, view) next to the image. The MAIRA-2 `Introduction` says the model takes "the lateral view, prior frontal image, prior report, Indication, Technique, and Comparison sections". The Phase 9 pass had already labelled `Methods > Data` for the same sentence-level fact, so `Introduction` got grade 1.

## Result

| Acceptance criterion | Result |
|---|---|
| 9 reranked runs (3 models by rewritten depth 20, original depth 20, rewritten depth 40) and the 4 dense runs on the v2 draft | pass |
| Pool over the union of the nine runs' top 5 (`--k 5 --threshold 0`), every candidate read, additions and rejections logged | pass: 241 candidates read, 9 added, 232 rejected (each row in `adjudicated.jsonl` has a reason); a second pool run after the changes shows 0 candidates left |
| Off-corpus questions stay without labels | pass: all 20 checked (196 candidate sections), none answers one, no label added |
| v2 frozen, `validate_golden` passes, CHANGELOG entry set from draft to frozen with the final hash | pass: hash `cc08395a87ca30fc302b98065395f77404f192d3999f22428cab009b38022b0c` |
| All 13 runs re-run on v2 with `--no-llm`, reports regenerated | pass: all 13 run files carry hash `cc08395a87ca`; checked in code |
| LangSmith sync | not done on purpose (the orchestrator handles it; it needs a committed file) |
| Quick checks | `pytest tests/`: 590 passed. `ruff check .`: clean. |

### The 13 runs (frozen v2, 45 on-corpus questions)

nDCG@5 is a 0 to 1 score for how well the labelled sections are ranked in the top 5 (higher is better). Recall@5 is the share of labelled sections found in the top 5. Recall@depth is the share found among all hits the reranker could see (20 or 40); it is the most a reranker at that depth can reach. Wrong-section share is the share of top-5 chunks from a paper that has a label for the question but from a section that has none. Time is one reranker call per question after one warm-up, on this CPU-only machine, not counting search or rewrite. No numbers here are interpreted; T7 compares them.

| Run | nDCG@5 | Recall@5 | Recall@depth | Wrong-section share | Rerank ms, median | Rerank ms, p90 |
|---|---|---|---|---|---|---|
| prefix (dense reference) | 0.564 | 0.521 | 0.828 (at 20) | 0.351 | n/a | n/a |
| baseline (dense) | 0.535 | 0.504 | 0.783 (at 20) | 0.356 | n/a | n/a |
| no_rewrite (dense) | 0.546 | 0.506 | 0.791 (at 20) | 0.347 | n/a | n/a |
| bge_base (dense) | 0.543 | 0.531 | 0.806 (at 20) | 0.320 | n/a | n/a |
| rerank_minilm | 0.684 | 0.635 | 0.828 (at 20) | 0.284 | 1655 | 1712 |
| rerank_minilm_orig | 0.699 | 0.642 | 0.828 (at 20) | 0.298 | 1614 | 1703 |
| rerank_minilm_d40 | 0.725 | 0.683 | 0.914 (at 40) | 0.280 | 3116 | 3438 |
| rerank_bge | 0.704 | 0.645 | 0.828 (at 20) | 0.347 | 9795 | 11787 |
| rerank_bge_orig | 0.685 | 0.629 | 0.828 (at 20) | 0.351 | 10572 | 14239 |
| rerank_bge_d40 | 0.718 | 0.662 | 0.914 (at 40) | 0.329 | 23193 | 27223 |
| rerank_medcpt | 0.639 | 0.599 | 0.828 (at 20) | 0.320 | 8884 | 13055 |
| rerank_medcpt_orig | 0.645 | 0.622 | 0.828 (at 20) | 0.324 | 9755 | 10160 |
| rerank_medcpt_d40 | 0.688 | 0.656 | 0.914 (at 40) | 0.316 | 17861 | 18858 |

- Details: the Recall@depth value is the same for all three models at one depth (0.828 at 20, 0.914 at 40), because a reranker only reorders the hits it is given. At depth 20 it equals the `prefix` dense run (after the 0.6 cosine pre-filter, which is the same hits). Recall@depth uses the run's own cut-off (0.6 for dense, none for reranked runs, whose stored hits are already pre-filtered at 0.6).
- Details: dense runs differ a little from T2 only because of the 9 new labels. The 45 on-corpus questions and the hits are the same. nDCG@5: prefix 0.562 to 0.564, baseline 0.532 to 0.535, no_rewrite 0.545 to 0.546, bge_base 0.544 to 0.543.
- Details: reranker setup per model. Load time and disk size (whole cached repo): minilm 4.1 s and 890 MB, bge 5.3 s and 1134 MB, medcpt 2.9 s and 877 MB. The real MiniLM weights are 91 MB. The 890 MB comes from duplicate MiniLM folders in the Hugging Face cache (the ONNX and OpenVINO copies from T3). Nothing was deleted. Use 91 MB for MiniLM in any comparison.
- Details: off-corpus scores (reported only, not judged here). In all 13 runs the off-corpus no-result rate is 0.00. Highest off-corpus top-1 score: dense `prefix` 0.840, `rerank_minilm` 0.995, `rerank_bge` 0.984, `rerank_medcpt` 0.9999998 (the sigmoid is saturated near 1 for MedCPT). This is for T7's refusal-cut-off curve, not decided here.
- Details: git hash in the run files is `3d61760-dirty` (the working tree had my unstaged label changes). Chunk count 595.

### Labelling pass

| Measure | Value |
|---|---|
| Questions in the pool | 45 on-corpus |
| Candidate sections read (distinct) | 241 |
| Added at grade 2 / grade 1 | 0 / 9 |
| Rejected | 232 |
| Labels after | 159 (84 at grade 2, 75 at grade 1), was 150 |

By reranker (a candidate counts for a model if any of its three runs had it in the top 5):

| Reranker | Candidates | Labels added | Candidates seen only by this model |
|---|---|---|---|
| minilm | 127 | 5 | 25 |
| bge | 140 | 5 | 41 |
| medcpt | 140 | 6 | 56 |

By question type: cross_paper 3 additions (of 47 candidates), multi_section 6 (of 38). Factual (62 candidates), paraphrased (55) and table (39) got none.

The nine additions (all grade 1; full reasons in `data/eval/CHANGELOG.md`): q022 maira_2 `Introduction`; q022 med_palm_m `Appendix > Chest X-ray report generation`; q028 maira_2 `Discussion`; q037 conch `Results > Zero-shot cross-modal retrieval`; q037 conch `Online Methods > Supervised classification experiments`; q039 med_palm_m `Appendix > MultiMedBench`; q041 deep_tumor_vqa `Discussion and Conclusion`; q042 intersectional_fairness_vlm `Discussion`; q042 intersectional_fairness_vlm `Methods > Experimental Design and Baseline Comparisons`.

- Details: method. I pulled each candidate's chunk text from the `papers` collection (via `sections --dump-text` to a folder outside the repo, so no paper text is committed) and judged by the Phase 9 standard: grade 2 fully answers, grade 1 holds part of the answer or clear support, same topic only is not added. Rank and reranker score were not used. Where a chunk was longer than my view, I searched the full chunk for the key terms. I did not use `pool --reject` on the command line 232 times; I called the same helper (`record_rejections` in `app/evaluation/authoring.py`) once per reason, so each row has a reason. No code was changed.
- Details: reasons in `adjudicated.jsonl` are mostly the two default ones ("same paper as a label; this section does not state the asked fact" and "other paper; topic differs from the question"), plus specific reasons for about 25 judgement calls (for example radfm `Discussion > Clinical impact` for q020, same call as Phase 9).
- Details: one rejection was forced by a rule, not by the text. llava_med `Experiments > Visual input example, Biomedical image from [6]` for q044 states the "more data helps, stage 1 alone is not enough" ablation, but its heading is one of the banned odd Phase 8 labels (`Visual input example...`), so it cannot be labelled. Owner may want to look at this: the text is relevant, the heading looks like a chunking artefact.
- Details: one label is by the chunk, not the heading. conch `Results > Zero-shot cross-modal retrieval` for q037 holds the sentence "To perform segmentation, we divide each WSI into tiles and use zero-shot classification to predict the label of each tile" (a chunking artefact: the sentence sits under the retrieval heading).
- Details: off-corpus check. The pool command skips off-corpus questions, so I ran it with `--include-off-corpus` for the 20 off-corpus questions and searched every candidate chunk for key terms of its question, reading every hit. Nothing answers any of them. The nearest case is q064 (retrieval-augmented generation against hallucination): MAIRA-2 sections show that prior studies and the lateral view cut invented comparisons and lateral mentions. That is extra input, not retrieval-augmented generation, so I kept it off-corpus (same call as T4). I did not stop for it. If you disagree, q064 is the one to look at.
- Details: files. `data/eval/adjudicated.jsonl` is new (232 rows, LF endings). `golden.jsonl` keeps LF endings; all lines of q001 to q065 keep id, question, type and `source_chunks`; only `relevant` lists of q022, q028, q037, q039, q041, q042 grew.

## Not checked

- The golden set is LLM-labelled (Claude), not human-verified. The same author wrote the questions (T4, and v0 in Phase 9) and read the retrieved text now. This is not an independent or blind pass in the human sense: I did not write the questions in this session, but the labeller is still the same model. A second reader would likely add or reject some labels, in particular the nine grade-1 judgement calls above.
- Pooling only sees sections that the nine reranked runs put in their top 5. A labelled-worthy section that none of the nine retrieves is still missing. The dense top 10 pool of Phase 9 and this one overlap, but are not the same.
- Dense and reranked runs were both scored on the same final labels. The nine added labels came from reranked runs, so they can only help reranked runs (and dense runs only where a dense run also retrieved those sections). I did not measure this effect separately; the dense nDCG@5 moved by -0.001 to +0.003 against T2.
- Timing is one CPU-only machine, one run each, no repeats. Machine noise is not measured (the draft-pass and final-pass times were close).
- The off-corpus check is a key-term search plus reading hits. A paraphrased answer with none of my key terms could be missed.
- No LangSmith sync, no comparisons (T7), no interpretation of which reranker wins.
- Nothing is committed.

## Changed files

- `data/eval/golden.jsonl` (9 labels added), `data/eval/adjudicated.jsonl` (new), `data/eval/CHANGELOG.md` (v2 draft to frozen, pooling pass entry)
- `data/eval/runs/`: new `rerank_{minilm,bge,medcpt}.json`, `_orig.json`, `_d40.json` (9 files); overwritten `baseline.json`, `no_rewrite.json`, `prefix.json`, `bge_base.json`
- `data/eval/reports/`: new `run-rerank_*.md` (9 files); overwritten `run-baseline.md`, `run-no_rewrite.md`, `run-prefix.md`, `run-bge_base.md`
- `docs/phases/10-rerank-hybrid/3-execute-T6.md` (this file)
