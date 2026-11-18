# T8: Rerank step in the app

**Status:** done. All five acceptance criteria pass. The default `RERANKER_MODEL` stays empty (off) until T7 picks a model.

## What

The app now has a sixth workflow step, "rerank", between retrieve and generate. Retrieve fetches 20 hits. Rerank re-scores them with the reranker model (a second model that reads the question and a passage together) and keeps the best 5. With no reranker set, it only keeps the first 5, so the steps and events are the same in every setup.

- Details: new `app/workflow/rerank.py` with `make_rerank_node(reranker, *, keep=5)`. `RETRIEVE_DEPTH = 20` lives in `app/workflow/retrieve.py`. `build_workflow(client, provider, llm, reranker)` takes the reranker as a required argument (`None` = off). The reranker reads `rerank_passage(title, section, excerpt)` and the rewritten query (the original question if the rewrite is empty). Equal scores keep the vector order. A reranker error, or a wrong number of scores, is logged, keeps the vector order cut to 5 and sets `rerank_fallback`; the request still succeeds.

## Example

New SSE event order for one safe query (before: `citations` came right after `retrieve`):

```
step_start classify    -> step_end classify
step_start rewrite     -> step_end rewrite
step_start retrieve    -> step_end retrieve
step_start rerank      -> step_end rerank
citations              (the final 5, in reranked order)
step_start generate    -> token ... token -> step_end generate
step_start validate    -> step_end validate
final
```

`debug` in `final` and in `/query` now looks like:

```json
{"route": "retrieval", "rewritten": "...", "rerank_model": "fake-reranker", "rerank_top_score": 0.95, "rerank_fallback": false}
```

With the reranker off, `rerank_model` and `rerank_top_score` are `null`.

- Details: `confidence` is still the cosine score of the vector top hit. Rerank does not change it. The `citations` event takes its confidence from the workflow state, which retrieve set one step earlier (the rerank step returns only citations). `rerank_fallback` in `debug` is an extra field beyond the plan; it makes a silent fallback visible.
- Details: `WorkflowState` got `rerank_top_score: float | None` and `rerank_fallback: bool`. `_initial_state` sets them to `None` and `False`. `_to_response` now takes the reranker, because the model name is not a per-request value and is not in the state.

## Result

| Acceptance criterion | Result | Tests |
|---|---|---|
| `make_rerank_node(reranker, *, keep=5)`: off cuts to 5; error logs, keeps vector order, sets `rerank_fallback` | Pass | `test_rerank_node_off_cuts_to_five_in_vector_order`, `test_rerank_node_error_keeps_vector_order_and_flags_fallback`, `test_rerank_node_wrong_score_count_is_a_fallback`, `test_rerank_node_default_keeps_five`, `test_rerank_node_no_citations_skips_the_reranker` |
| Order and cut | Pass | `test_rerank_node_orders_by_score_and_cuts_to_keep`, `test_rerank_node_ties_keep_the_vector_order`, `test_rerank_node_sends_rewritten_query_and_shared_passage_text`, `test_rerank_node_falls_back_to_question_when_rewrite_is_empty` |
| Retrieve fetches `RETRIEVE_DEPTH` (20); `build_workflow(..., reranker)`; `rerank_top_score` in state; `_initial_state` updated | Pass | `test_retrieve_fetches_the_retrieve_depth` (checks the Qdrant `limit`), the updated `test_full_workflow_e2e_safe` and `test_full_workflow_e2e_unsafe`, every API test |
| SSE `citations` when `rerank` ends; debug has `rerank_model` and `rerank_top_score`; `confidence` stays cosine | Pass | `test_stream_event_order` (6 steps, `citations` after rerank), `test_stream_citations_come_after_rerank_ends`, `test_stream_rerank_reorders_and_keeps_cosine_confidence`, `test_stream_rerank_off_cuts_to_five_in_vector_order`, `test_stream_rerank_error_falls_back_and_still_answers`, `test_query_returns_rerank_debug_fields`, `test_stream_final_equals_query_response` |
| UI label "Reranking passages"; `RERANKER_MODEL` in Compose `api` service and `.env.example`; shared Hugging Face cache volume | Pass (static check only) | `ui/app.py` `_STEP_LABELS`; Compose YAML parses and `api.environment.RERANKER_MODEL` is `${RERANKER_MODEL:-}`; the `api` service already mounts the shared `huggingface_cache` volume |
| Tests: node unit tests, stream order with 6 steps, `get_reranker` overridden in every API test | Pass | `tests/test_query.py`, `tests/test_query_stream.py` (including the disconnect test), `tests/test_retrieval.py` fixture all set `app.dependency_overrides[get_reranker]` |

Gates:

| Gate | Result |
|---|---|
| `pytest tests/` without `tests/test_eval_*` | 253 passed |
| `ruff check` and `ruff format --check` on every touched Python file | clean |
| `mypy app/workflow app/api app/main.py` | no issues, 15 files |
| `mypy app/` (whole tree) | blocked: `app/evaluation/report.py:431` has a syntax error from the parallel T5 work. Not mine. |
| `bandit` on `app/workflow` and `app/api` | clean |
| Full `pytest tests/` and `bandit -r app/` | not clean for the same reason: T5's unfinished `report.py` breaks collection of `tests/test_eval_*` and bandit's parse of that file. Re-run once T5 settles. |

- Details: `tests/test_model_status.py` is in the plan's file list but needed no change. `/model/status` does not use the reranker, and its tests never call a route that depends on `get_reranker`.
- Details: Compose: `RERANKER_MODEL: ${RERANKER_MODEL:-}` was added to the `api` service only (the `ingest` service does not rerank). The `api` service already mounts `huggingface_cache` at `/root/.cache/huggingface` (the container runs as root and sets no `HF_HOME`), so a downloaded reranker lands in the same cache volume as the embedder. No Compose volume change was needed. `CLAUDE.md` got `RERANKER_MODEL` in the "Absolute constraints" env-var list (one word). The rest of the docs are for T12.
- Details: files keep their original line endings (many are CRLF). New file `app/workflow/rerank.py` is LF, like `graph.py`; git normalises on commit.

## Not checked

- The Docker path: `docker compose up --build`, the first-start download into the shared volume, and the live stream showing 6 steps. Docker was not run. That is for Verify.
- Real-model latency inside the request. All tests use `FakeReranker`. The time of a real model on 20 passages is measured in T5/T7.
- First-request cost. `get_reranker()` builds the model lazily on the first request that needs it (FastAPI runs the sync dependency in a worker thread, so the event loop is not blocked, but that first request waits for the load and, in a fresh container, the download). The plan does not ask for warm-up at startup. Worth a decision once T7 picks a model.
- A live Qdrant run. The tests use a mocked Qdrant client, so retrieving 20 and cutting to 5 against real data is only covered by the Phase 9 truncation check.
- Full-suite status. See the gate table: the eval tests and a whole-tree `mypy` and `bandit` could not run while T5's `report.py` is mid-edit.
