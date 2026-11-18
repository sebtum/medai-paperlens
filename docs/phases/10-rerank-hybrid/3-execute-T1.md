# Phase 10, T1: Fix the rewriter leak (execute report)

All numbers below come from a golden set that is LLM-labelled (Claude), not human-verified. T1 does not run any eval.

## What

The question rewriter (a small model that rewrites the user's question before search) sometimes starts its answer with the words "Rewritten question:". That label was stored and searched with. The rewriter now strips it, and the 6 stored rewrites that had it were cleaned in place without calling Ollama.

- Details: `clean_rewrite(text)` in `app/workflow/rewrite.py` strips whitespace, a leading `Rewritten question:` (any case, repeated, optional space before the colon) and one pair of surrounding quotes (`"`, `'`, curly double quotes), until nothing changes. `rewrite_node` calls it before the fallback, so a bare label falls back to the original question. `_PROMPT` is unchanged.

## Example

q009, stored rewrite:

- Before: `Rewritten question: What hardware and duration did RoentGen fine-tune for 60k training steps?`
- After: `What hardware and duration did RoentGen fine-tune for 60k training steps?`

## Result

| Acceptance criterion | Result | Check |
|---|---|---|
| `clean_rewrite` strips the label (any case, repeated), quotes, whitespace; `rewrite_node` uses it before the fallback | pass | `test_clean_rewrite` (15 cases), `test_rewrite_node_uses_the_cleaned_text`, `test_rewrite_node_falls_back_when_only_the_label_is_returned` in `tests/test_workflow.py` |
| `_PROMPT` unchanged; cache key hashes prompt plus `REWRITE_CLEANUP_VERSION` | pass | `test_prompt_hash_covers_the_real_prompt_and_the_cleanup_version`, `test_a_new_cleanup_version_misses_the_cache`, existing `test_key_includes_question_model_and_prompt` |
| `_RecordingRewriteLlm` counts output empty after cleaning as a failure | pass | `test_output_empty_after_cleaning_is_a_failure`, `test_a_leaked_label_is_cleaned_before_it_is_stored` |
| `rewrites.jsonl` rewritten once, exactly 6 texts change, logged in `CHANGELOG.md`, test that no stored rewrite starts with the prefix | pass | Migration script reported exactly `q009, q015, q029, q031, q035, q042` changed; a second check against `HEAD` confirmed ids, question hashes and models are untouched and the other 44 texts are identical. `test_committed_rewrites_have_no_leaked_label`, `test_committed_rewrites_use_the_current_prompt_hash` |

Gates:

- `pytest tests/`: 506 passed (the full suite, which includes the other agent's current T3 files).
- `pytest tests/test_workflow.py tests/test_eval_target.py tests/test_eval_runner.py`: 155 passed.
- `ruff check` on all T1 files: clean. `ruff format`: clean.
- `mypy app/`: 1 error, in `app/retrieval/rerank.py:52` (T3's file, `CrossEncoder.predict` argument type). Not in any T1 file.
- `ruff check .`: 2 errors, both in `app/retrieval/rerank.py` / T3 work (one line too long at `rerank.py:73`). Not in any T1 file.

Details:
- Cache key: the field is still named `prompt_sha256` in the JSONL, and `rewrite_prompt_sha256()` keeps its name. It now returns `sha256(prompt + "\n" + REWRITE_CLEANUP_VERSION)`, version `"1"`. All 50 lines therefore have a new key (one shared value, `945eb073...`). Phase 9 run records store the old hash in their config, which is expected.
- File format: `rewrites.jsonl` was CRLF in the working tree but LF in git (`.gitattributes` pins `eol=lf`). It was written with LF, so the diff is only the changed fields.
- Changed texts are listed in `data/eval/CHANGELOG.md`, section "Rewrite cache cleanup (2026-11-18)".

## Not checked

- No eval was re-run. T2 does that, with the clean rewrites.
- Not checked against a live Ollama: whether the model still leaks the label in other forms (for example a different label wording). Only the exact "Rewritten question:" label is handled.
- Quote stripping removes one outer pair even when the quote marks do not belong together inside the text (for example `"A?" and "B?"`). Not expected in this corpus, not tested.
- The two lint and type errors in T3's `rerank.py` are left for that task.
