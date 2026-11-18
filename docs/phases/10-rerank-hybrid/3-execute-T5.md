# T5: Eval support for reranking

**Status:** done. All seven acceptance criteria pass. One reading of the plan needs your eye (see "Interpretation" below).

## What

The eval can now re-score the search hits with a reranker, time it, and report how often the top 5 holds the right paper but the wrong section. Nothing here needs Qdrant or a real model in tests.

- Details: touched `app/evaluation/target.py`, `runner.py`, `compare.py`, `report.py`, `metrics.py`, `authoring.py`, `langsmith_eval.py` and `scripts/eval_retrieval.py`. Dense configs (`baseline`, `no_rewrite`, `prefix`, `bge_base`) give the same hits and metrics as before. Only extra fields were added.

## Example

Run one reranker setup, then compare it with today's default and look at the refusal curve. Run names are free text.

```
python -m scripts.eval_retrieval run --config rerank_medcpt --no-llm
python -m scripts.eval_retrieval run --config rerank_medcpt --name rerank_medcpt_d40 --depth 40 --no-llm
python -m scripts.eval_retrieval run --config rerank_bge --name rerank_bge_orig --rerank-query original --no-llm
python -m scripts.eval_retrieval compare prefix rerank_medcpt
python -m scripts.eval_retrieval curve rerank_medcpt --gate
python -m scripts.eval_retrieval pool rerank_medcpt rerank_bge --k 5 --threshold 0 --reject q012 maira_2 "Methods" --reason "read, no answer"
```

What each one records:

- The run file gets `score_kind: rerank`, the reranker model, load time, disk size, and `rerank_ms` per question.
- The run report gets a "Reranker" table (median and 90th percentile of the time) and a "Wrong section" column.
- `compare` cuts a reranked run at 0 and a cosine run at 0.6 unless you pass `--threshold`.
- `curve --gate` writes `gate-<run>.md`: for each cut-off t from 0 to 1, how many off-corpus questions are refused and how many on-corpus questions are lost.
- `pool --reject` writes to `data/eval/adjudicated.jsonl` and later passes skip those rows.

I ran the wrong-section metric on the committed `prefix` run (no Qdrant needed): 0.364 at k=5. That matches the 36 % from the Phase 9 diagnosis.

- Details: model aliases for `--reranker`: `minilm` = `cross-encoder/ms-marco-MiniLM-L-6-v2`, `bge` = `BAAI/bge-reranker-base`, `medcpt` = `ncbi/MedCPT-Cross-Encoder`; a full model ID also works. The three named configs (`rerank_minilm`, `rerank_bge`, `rerank_medcpt`) are built on `prefix`.

## Result

| Acceptance criterion | Result | Tests |
|---|---|---|
| `RetrievalConfig` has `reranker`, `rerank_query`, `prefilter_threshold = 0.6`. `retrieve_ranked` pre-filters, reranks with `rerank_passage` text, sorts (ties keep dense order), returns the time | Pass | `TestRerankConfig` (5), `TestRerankedRetrieval` (12: `test_prefilters_reranks_and_sorts_by_the_reranker_score`, `test_the_reranker_reads_title_section_and_text`, `test_ties_keep_the_dense_order`, `test_returns_the_time_taken`, `test_original_question_can_be_the_reranker_text`, and more) in `tests/test_eval_target.py` |
| Named configs `rerank_minilm`, `rerank_bge`, `rerank_medcpt` (on `prefix`); CLI `--depth`, `--rerank-query`, `--reranker`, name required | Pass | `TestRerankConfigs` (10, incl. `test_each_flag_needs_an_explicit_name`, `test_cli_refuses_a_flag_without_a_name`), `TestNamedConfigs::test_the_plan_configs_exist` |
| Run meta has `score_kind`, model, load time, disk size. `rerank_ms` per question after one warm-up. Report shows median and 90th percentile | Pass | `TestRerankRun` (11: `test_run_records_rerank_scores_timing_and_meta`, `test_the_model_is_loaded_once_and_warmed_up_before_timing`, `test_report_shows_the_reranker_and_its_time`), `test_rerank_timing_median_and_90th_percentile`, `test_run_summary_report_shows_wrong_section_and_timing`, `test_rerank_ms_roundtrips_and_is_omitted_when_absent` |
| `compare` default cut-off follows `score_kind`. `curve --gate` sweeps the top-1 score. Refused and lost per t | Pass | `TestCompareAndGateByScoreKind` (4), `test_reranked_run_defaults_to_no_cut`, `test_summary_of_a_reranked_run_is_cut_at_zero_by_default`, `test_gate_curve_*` (5), `test_gate_curve_report_lists_counts_and_the_caveat` |
| New metric `wrong_section_share(hits, labels, k)` shown in the run report | Pass | 4 tests in `tests/test_eval_metrics.py`, `test_summary_reports_wrong_section_share` |
| `pool` records rejected candidates in `data/eval/adjudicated.jsonl` and skips them | Pass | `TestPoolAdjudicated` (7) |
| LangSmith experiment threshold follows `score_kind`. Fakes only. `NAMED_CONFIGS` test updated | Pass | `test_langsmith_experiment_follows_the_score_kind`, `test_langsmith_experiment_of_a_dense_run_keeps_0_6`, `test_the_plan_configs_exist` |

Gates:

| Gate | Result |
|---|---|
| `tests/test_eval_*.py` plus `tests/test_rerank.py` | 357 passed |
| `ruff check` and `ruff format --check` on all touched files | clean |
| `mypy app/` | no issues, 44 files |
| `bandit -r app/` | clean (see the cache note below) |
| Full `tests/` | not run on purpose: another agent was editing the workflow files at the same time |

- Details: new tests: 4 in `test_eval_metrics.py`, 17 in `test_eval_target.py`, 17 in `test_eval_compare.py`, about 32 in `test_eval_runner.py`. The harness gets a `RerankHarness` with a `FakeReranker` that returns fixed scores. Two existing pool tests now pass `--adjudicated <tmp file>`, so they cannot read the real file once it exists.
- Details: what the runner does for a reranked run, in order: load the reranker through `Services.reranker_factory` (timed), one warm-up call, search at depth and threshold 0, drop hits under the 0.6 cosine pre-filter, fetch title and text from Qdrant, call `rerank_passage(title, section, text)`, stable sort by score, record `rerank_ms`. A question with no hits left after the pre-filter makes no model call and has no `rerank_ms`.
- Details: run-header fields. All runs: `score_kind`, `reranker` (null for dense). Reranked runs also: `rerank_query`, `prefilter_threshold`, `reranker_load_s`, `reranker_disk_bytes`. Old run files without `score_kind` read as cosine, so the Phase 9 runs still load and compare.
- Details: the title-and-text fetch lives in `app.evaluation.target.fetch_chunk_payloads`, because `runner` imports `target` and the reverse would be circular. `runner.fetch_chunk_texts` keeps its signature and now calls it. This is a small move away from the plan text ("extend `fetch_chunk_texts` in `runner.py`"), same effect.
- Details: the gate curve uses 25 thresholds: 0 to 1 in steps of 0.05, plus 0.01, 0.02, 0.98 and 0.99. Reranker scores pass through a sigmoid, so many sit very close to 0 or 1. A question counts as refused at t when no hit has a score of at least t. A reranked run has already lost hits below the 0.6 cosine pre-filter, so those questions are refused at every t. The report says so.
- Details: `wrong_section_share` takes an optional fourth argument `threshold` (default 0). It is the share of the top-k chunks whose paper has a label for the question but whose section has none. The denominator is the chunks kept, as in the Phase 9 diagnosis. The report column uses the run's own cut-off (0.6 cosine, 0 rerank).
- Details: bandit flagged `snapshot_download` (B615, no revision pin) in my first version of the disk-size helper. I switched to `huggingface_hub.scan_cache_dir`, which only reads the local cache.

### Interpretation (please confirm)

The plan says CLI flags `--depth`, `--rerank-query` and `--reranker` (the name is required when used). I read "the name" as the run name `--name`. Reason: each flag changes what a named config measures, and without a name the run would overwrite that config's own run file (for example `rerank_bge.json`). T6 needs 9 reranked runs (3 models by 3 variants), so this fits. If you meant something else, this is a small change in `make_run_options` in `runner.py`.

## Not checked

- No real model, no Qdrant, no Ollama. Timing, disk size and load time were only tested with fakes. A first real run is in T6.
- Disk size: `reranker_disk_bytes` is the size of the whole cached Hugging Face repository for that model, all files and revisions. A quick check on this machine gives 890 MB for MiniLM, because the cache also holds the ONNX and OpenVINO copies that T3 downloaded by mistake. Delete those first (see the T3 report) or the MiniLM size will look 10 times too big. It is 1134 MB for bge and 877 MB for MedCPT.
- Warm-up: one warm-up call keeps first-call cost out of the times. It does not remove variation from other programs on the machine.
- The `curve` (cosine threshold sweep) on a reranked run is not blocked. It is meaningless there; use `--gate`.
- `pool` rejections are stored by question, paper and section only. They are not tied to a golden-set hash, so after T6 freezes v2 a rejection stays valid unless the section is renamed.
- A syntax error sat in `app/evaluation/report.py` for a few minutes while I was editing (a line break inside a string). It is fixed and all tests pass. If the T2 agent hit an import error from the runner in that window, that was the cause.
- Phase 9 reports and runs under `data/eval` show as modified in git. That is the T2 agent's work, not mine.

## Changed files

- `app/evaluation/metrics.py`, `target.py`, `compare.py`, `report.py`, `runner.py`, `authoring.py`, `langsmith_eval.py`
- `scripts/eval_retrieval.py`
- `tests/test_eval_metrics.py`, `tests/test_eval_target.py`, `tests/test_eval_compare.py`, `tests/test_eval_runner.py`
- `docs/phases/10-rerank-hybrid/3-execute-T5.md` (this file)
