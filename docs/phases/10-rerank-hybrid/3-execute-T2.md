# Phase 10, T2: Clean baseline re-run (execute report)

Yes, the Phase 9 outcomes still hold. The prefix still wins, the rewrite stays, `bge_base` is still not better, and the 0.6 score cut-off is still inert. No outcome flipped, so no owner decision is needed. All numbers come from a golden set that is LLM-labelled (Claude), not human-verified.

## What

The four Phase 9 setups were run again with the cleaned rewrites (T1 removed the leaked "Rewritten question:" label from 6 stored rewrites). Runs used `--no-llm`, so rewrites came from the committed cache and Ollama was not used. Then the three comparisons against `baseline` were re-run, and the threshold curve for `baseline`.

- Details: commands, all via `.\.venv\Scripts\python.exe -m scripts.eval_retrieval`: `run --config {baseline,no_rewrite,prefix,bge_base} --no-llm`, `compare baseline {prefix,no_rewrite,bge_base}`, `curve baseline`. The preflight (stale labels, embedding model) passed for all four runs. Golden hash is unchanged (`4a4733e5fa76`), 595 chunks, git `44b51cb-dirty`. No Qdrant collection was touched. The first attempt failed with a SyntaxError in `app/evaluation/report.py` (the other agent's half-finished edit). It ran cleanly after a short wait, and no code file was touched by me.

## Result

nDCG@5 is a 0 to 1 score for how well the labelled sections are ranked in the top 5 (higher is better). The verdict comes from the decision rule of ADR-0024: B replaces A only if the lower end of the 95 % range is above 0 and no question type is clearly worse.

| Setup (vs baseline) | Phase 9 nDCG@5 | New nDCG@5 | Verdict in Phase 9 | Verdict now |
|---|---|---|---|---|
| baseline | 0.525 | 0.532 | reference | reference |
| prefix | 0.566 | 0.562 | adopted (B replaces A) | adopted (B replaces A) |
| no_rewrite | 0.545 | 0.545 | rewrite kept (A stays) | rewrite kept (A stays) |
| bge_base | 0.550 | 0.544 | not better (A stays) | not better (A stays) |
| threshold 0.6 | flat 0.40 to 0.68, off-corpus no-result 0.00 | flat 0.40 to 0.68, off-corpus no-result 0.00 | inert, kept | inert, kept |

| Acceptance criterion | Result |
|---|---|
| Four runs with `--no-llm`, reports regenerated (with the Precision and Recall-versus-ceiling columns) | pass |
| Comparisons prefix, no_rewrite, bge_base against baseline re-run and noted against ADR-0025 | pass, all three outcomes unchanged |
| Stop and ask the owner if an ADR-0025 outcome flips | not triggered, nothing flipped |

Effect of the rewrite fix: small. Only 6 of 50 queries changed (q009, q015, q029, q031, q035, q042), and only those questions can have changed hits. The other 44 are identical.

- The prefix margin got thinner. The gain over baseline fell from +0.041 to +0.030, and the lower end of the range from +0.012 to +0.005, which is still above 0. Reason: the fix helped baseline (q042 went from 0.307 to 0.613) but did not help prefix (q015 and q031 each lost 0.10, q042 gained 0.04).
- The rewrite question stays open the same way: `no_rewrite` is +0.013 (was +0.020), range includes 0.
- `bge_base` is still clearly worse on `table` questions.
- The `no_rewrite` run has byte-identical hits to the Phase 9 file. This shows the setup reproduces exactly, so the changes elsewhere come from the fix only.

Why the prefix result deserves a second look (my reading, not a flip): its lower bound is now 0.005, so it passes the rule by a small margin. The Phase 9 decision stands under the rule as written.

- Details, comparisons against `baseline` (mean difference B minus A on 45 on-corpus questions, 95 % range, wins / losses / ties):
  - prefix: +0.030, [+0.0052, +0.0564], 11 / 6 / 28 (Phase 9: +0.041, [+0.0123, +0.0737], 11 / 4 / 30). No type clearly worse. Paraphrased +0.063, table +0.049, multi_section +0.038, factual and cross_paper +0.000.
  - no_rewrite: +0.013, [-0.0097, +0.0382], 7 / 6 / 32 (Phase 9: +0.020, [-0.006, +0.046], 8 / 6 / 31). No type clearly worse.
  - bge_base: +0.012, [-0.0461, +0.0698], 17 / 15 / 13 (Phase 9: +0.026, [-0.035, +0.084], 17 / 14 / 14). Clearly worse on table: -0.075, 1 win / 3 losses / 4 ties (Phase 9: -0.069, 1 / 3 / 4).
- Details, baseline per type nDCG@5: factual 0.676 (same), paraphrased 0.452 (same), cross_paper 0.292 (same), table 0.565 (same), multi_section 0.644 (was 0.610). Baseline Recall@5 is 0.509 (same), MRR@5 0.717 (was 0.703).
- Details, other run numbers, nDCG@5 / Recall@5 / MRR@5: no_rewrite 0.545 / 0.514 / 0.727; prefix 0.562 / 0.531 / 0.763 (Phase 9: 0.566 / 0.531 / 0.767); bge_base 0.544 / 0.543 / 0.698 (Phase 9: 0.550 / 0.543 / 0.706).
- Details, per-question nDCG@5 change from the fix, only the 6 changed questions had any change: baseline q042 +0.307 (others 0); prefix q015 -0.100, q031 -0.100, q042 +0.043; bge_base q009 -0.369, q015 +0.126, q031 -0.050.
- Details, threshold curve (baseline): Recall@5 0.509 and nDCG@5 0.532 are flat from t = 0.40 to 0.68, so 0.6 removes nothing. Max off-corpus top-1 score is still 0.829. Off-corpus no-result rate is 0.00 up to t = 0.76 and 0.20 from 0.78. On-corpus empty rate at 0.78 is 0.33 (Phase 9: 0.31); small shifts at 0.70 (0.02, was 0.00) and 0.80 (0.47, was 0.44).
- Details, wrong-section share at k = 5 (reported by the new report column, "right paper, wrong section"): baseline 0.369, no_rewrite 0.356, prefix 0.364, bge_base 0.329. The plan quotes 36 % for today's setup, which matches.
- Details, run record format: the run files only gained two config fields, `score_kind` (`cosine`) and `reranker` (`null`), and `rewrite_prompt_sha256` has the new cache key (`945eb073...`). Per-question fields and the hit layout are unchanged, and the golden hash and chunk count are the same. Old numbers keep the same meaning. This is T5's new meta format, already visible in the committed code.

## Not checked

- No flip means no owner question here, but whether a margin of +0.005 at the lower end is worth keeping the prefix is the owner's call if they want it. The rule says keep it.
- The leaked-label fix was tested only on the golden set's 6 affected questions. Real user queries are not tested.
- The golden set is LLM-labelled (Claude), not human-verified, and has only 5 off-corpus questions, so the off-corpus rate moves in steps of 0.20.
- Not run: the LangSmith experiment (`--langsmith`), and no rewrite was generated (Ollama not used).
- Nothing was committed. The run files and reports are modified in the working tree.

## Files

Changed (overwritten):
- `data/eval/runs/baseline.json`, `no_rewrite.json`, `prefix.json`, `bge_base.json`
- `data/eval/reports/run-baseline.md`, `run-no_rewrite.md`, `run-prefix.md`, `run-bge_base.md`
- `data/eval/reports/compare-prefix-vs-baseline.md`, `compare-no_rewrite-vs-baseline.md`, `compare-bge_base-vs-baseline.md`
- `data/eval/reports/curve-baseline.md`

Created: this report. The `_v0` Phase 9 files were not touched.
