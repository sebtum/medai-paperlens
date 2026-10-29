# Phase 7: Streaming — Plan

## Decisions (from Plan, see ADR-0017/0018/0019)
- **SSE**: FastAPI 0.136 built-in `fastapi.sse.EventSourceResponse` / `ServerSentEvent`
  (endpoint is an async generator). Sets `Cache-Control: no-cache`,
  `X-Accel-Buffering: no`, keepalive comment every 15 s. No new dependency (U1).
- **Graph streaming**: `astream(stream_mode=["tasks", "custom", "values"])`. `tasks` gives
  node start + end (elapsed ms measured in the endpoint by task id), `custom` carries
  tokens/errors, last `values` chunk is the final state. Under `ainvoke` the custom writer
  is a no-op (A1 confirmed), so `/query` keeps `ainvoke`.
- **One code path**: `generate_node` always calls `llm.stream()`, writes each token as
  `{"type": "token", "text"}`, joins and strips for `answer`. LLM exception → writes
  `{"type": "error", "message"}` and returns `_ERROR_ANSWER`.
- **R1 reworded**: `/query` contract and test assertions unchanged; test *mock setup*
  moves to a shared `FakeLlm` (has `generate` + `stream`) in `tests/conftest.py`.
- **Event schema** (`event:` = type, `data:` = JSON): `step_start {node}`,
  `step_end {node, elapsed_ms}`, `citations {citations, confidence}` (on retrieve's task
  result), `token {text}`, `error {message}`, `final` = `QueryResponse`. Unexpected
  exception in the graph → `error`, then `final` with the error answer, `grounded: false`.
- **Ollama stream**: `self._http.stream("POST", …, stream=True)` + `aiter_lines()` NDJSON;
  yield `response` / `message.content` only, never `thinking` (R7, A2). Cancellation
  exits the context manager → connection closed → Ollama stops generating (R6).
- **Gemini stream**: `async for chunk in await client.aio.models.generate_content_stream(…)`,
  skip empty `chunk.text`. google-genai isn't installed locally → fake `_client` in tests.
- **UI** (U3): sync `httpx.Client.stream()` + `iter_lines()` + small SSE parser, rendering
  into `st.status` / `st.empty` placeholders. No `st.write_stream`, no `asyncio.run` on
  the stream path.
- **LangSmith** (U4, A4): langsmith 0.12.5 already installed via langchain-core.
  `@traceable` supports async generators; a join `reduce_fn` gives one run output.
  Gate is `LANGSMITH_TRACING == "true"`; unset → no client, no network (R9).
- **Testing SSE** (U5): httpx 0.28 `ASGITransport` buffers the whole body, which is fine
  for event-order tests. The disconnect test (R6) uses a raw ASGI harness; a scope without
  `spec_version` takes Starlette's `listen_for_disconnect` path, the same one uvicorn 0.46
  (spec 2.3) uses.

## Task graph
G1: T1 → G2: T2 ‖ T3 ‖ T7 → G3: T4 → G4: T5 → G5: T6 ‖ T8 → G6: T9

T2 and T7 both touch `app/llm/gemini.py`, so run them in sequence (T2 then T7) when
using subagents.

## Tasks

### [x] T1: `stream()` on Protocol + OllamaClient
- Files: `app/llm/base.py`, `app/llm/ollama.py`, `tests/test_llm_stream.py` (new)
- Depends on: —
- Group: G1
- Acceptance criteria:
  - [x] `LlmProvider` has `def stream(prompt, **kwargs) -> AsyncIterator[str]`
  - [x] Ollama streams `/api/generate` (str prompt) and `/api/chat` (messages) NDJSON, tested with `httpx.MockTransport`
  - [x] `thinking` fields are never yielded
  - [x] HTTP error status raises; `mypy app/` clean
- Note: Extracted `_build_request()` shared by `generate` and `stream` (behaviour of `generate` unchanged). Suite 52 passed, ruff + mypy clean. `GeminiClient` has no `stream` yet, so it doesn't satisfy `LlmProvider` until T2.

### [x] T2: GeminiClient.stream
- Files: `app/llm/gemini.py`, `tests/test_llm_stream.py`
- Depends on: T1
- Group: G2
- Acceptance criteria:
  - [x] Yields chunk texts from a fake `aio.models.generate_content_stream`
  - [x] Empty / `None` chunk texts are skipped
- Note: Extracted `_contents()` static helper shared by `generate` and `stream`. Suite 54 passed, ruff + mypy clean.

### [x] T3: Shared FakeLlm + migrate test mocks
- Files: `tests/conftest.py`, `tests/test_workflow.py`, `tests/test_query.py`, `tests/test_retrieval.py`
- Depends on: T1
- Group: G2
- Acceptance criteria:
  - [x] `FakeLlm(answer=…, tokens=…, error=…)` passes `isinstance(_, LlmProvider)`
  - [x] Existing tests use it; assertions unchanged except `generate.assert_called_once` → equivalent call count on the fake
  - [x] Full suite green
- Note: `FakeLlm` records every prompt from `generate` and `stream` in `calls`, so the call-count assertions (`len(fake_llm.calls)`) stay valid once T4 switches `generate_node` to `stream()`. Tests import it via `from conftest import FakeLlm`. Suite 55 passed, ruff + mypy clean.

### [x] T4: generate_node streams via writer
- Files: `app/workflow/generate.py`, `tests/test_workflow.py`
- Depends on: T1, T3
- Group: G3
- Acceptance criteria:
  - [x] `answer` = stripped join of tokens; empty → `_ERROR_ANSWER`
  - [x] LLM exception → `error` custom event + `_ERROR_ANSWER`
  - [x] Under `ainvoke` no writer side effects; under `astream(["custom"])` tokens arrive in order
- Note: `get_stream_writer()` raises `RuntimeError` when a node is called directly (outside a graph run), so `_stream_writer()` falls back to a no-op; this keeps direct-call node tests working. Suite 61 passed, ruff + mypy clean.

### [x] T5: `POST /query/stream` endpoint
- Files: `app/api/routes/query.py`, `app/models/query.py` (event payload models), `tests/test_query_stream.py` (new)
- Depends on: T4
- Group: G4
- Acceptance criteria:
  - [x] Shared `_initial_state(request)` / `_to_response(state)` used by both endpoints
  - [x] `content-type: text/event-stream`, `X-Accel-Buffering: no`
  - [x] Order: step_start/step_end per node → citations after retrieve → tokens → final
  - [x] Unsafe query: no `token` events; `final` has `grounded: false`, `route: unsafe_medical_advice` (R3)
  - [x] LLM error: `error` then `final` with the error answer (R5)
  - [x] `final` equals the `/query` JSON for the same input and fakes (R2)
  - [x] Empty question → 422; existing `/query` tests untouched (R1)
- Note: `citations` comes right after retrieve's `step_end` (both come from the same task-result chunk; `step_end` is emitted first). A `tasks` chunk is a start when it has `input`, an end when it has `result`. The graph stream runs in its own task via `_iterate_in_own_task` (see T6; a plain `aclosing` wrapper was not enough). Event payload models: `StepStartEvent`, `StepEndEvent`, `CitationsEvent`, `TokenEvent`, `ErrorEvent`. `query.py` imports the private `_ERROR_ANSWER` from `generate.py`. Also tested: an unexpected graph exception gives `error` then `final`. Suite 68 passed; ruff, mypy, bandit clean.

### [x] T6: Disconnect cancels graph + LLM (R6)
- Files: `tests/test_query_stream.py` (+ endpoint fix if needed)
- Depends on: T5
- Group: G5
- Acceptance criteria:
  - [x] Raw ASGI harness; FakeLlm stream yields one token, then blocks on an Event with a `finally` flag
  - [x] `receive` returns `http.disconnect` after the first token is sent → flag set within 2 s; the app call returns cleanly
- Note: The test found a real bug. The app returned cleanly on disconnect, but the `generate` node task and the LLM stream kept running. Cause: FastAPI cancels the SSE producer with an anyio cancel scope, which re-cancels the consumer while LangGraph awaits its own cleanup, so the node tasks are never cancelled (reproduced with a bare anyio scope; a plain `asyncio` cancel and `aclosing` both work). Fix: `_iterate_in_own_task()` in `app/api/routes/query.py` runs `workflow.astream` in a separate task feeding a queue and cancels that task when the response ends. Pump tasks are kept in a module-level set so they are not garbage collected. Suite 69 passed; ruff, mypy, bandit clean.

### [x] T7: Opt-in LangSmith tracing
- Files: `pyproject.toml` (+ `uv.lock`), `app/llm/ollama.py`, `app/llm/gemini.py`, `.env.example`, `tests/conftest.py`
- Depends on: T1, T2
- Group: G2
- Acceptance criteria:
  - [x] `langsmith` is a direct dependency; `pip-audit` clean (lock with `uv lock` only, never `uv run` / `uv sync`)
  - [x] `@traceable(run_type="llm")` on `generate` and `stream` (stream with a join reducer)
  - [x] Autouse fixture removes `LANGSMITH_*` env vars in tests (R8)
  - [x] Test: no `langsmith.Client` is constructed when tracing is unset (R9)
  - [x] `LANGSMITH_TRACING`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT` in `.env.example`
- Note: Extra files: `app/llm/base.py` (shared `join_chunks` reducer), `tests/test_llm_stream.py` (R9 test). The autouse fixture also clears `LANGCHAIN_*`. `uv.lock` gained 2 lines. Suite 56 passed; ruff, mypy, bandit, pip-audit clean. Not verified against a real LangSmith account (traces only checked as no-ops when off).

### [x] T8: Streamlit UI on the stream
- Files: `ui/app.py`, `tests/test_ui_client.py`
- Depends on: T5 (event schema)
- Group: G5
- Acceptance criteria:
  - [x] U2 spike first: does the `run_every=5` fragment rerun interrupt a 30 s+ stream? If yes, render the stream outside the fragment (fragment only gates on warm status) or skip polling while `st.session_state["querying"]`. Record the outcome in Note.
  - [x] `parse_sse_lines(lines) -> Iterator[tuple[str, dict]]` unit-tested (multi-line data, keepalive comments ignored)
  - [x] `stream_query_api()` sync generator tested with `httpx.MockTransport`
  - [x] Live step list with timings, citations, token text; replaced by `final.answer` at the end (R4)
  - [x] Connect / timeout / HTTP errors map to the existing `render_error`
  - [x] Final result kept in session state so reruns still show it
- Note: U2 spike not run in a browser (no way to drive one here). Avoided the question by design: the fragment polls (`run_every=5`) only while the model is not warm; once warm, `main()` builds it with `run_every=None`, so it reruns only on form submit and a timer rerun cannot interrupt a stream. Warm/cold transitions trigger a full `st.rerun()` (state in `session_state["model_warm"]`). Added `parse_sse_lines`, `stream_query_api` (sync; takes an optional `transport` for tests; transport errors come back as a final `client_error` event) and `run_streamed_query` (`st.status` step list with timings, citations line, live tokens; the final answer is rendered from `final` via the existing session-state path). `call_query_api` is kept (still tested). Suite 77 passed; ruff, mypy, bandit clean. Not checked manually against a live Streamlit session.

### [x] T9: Docs + demo path
- Files: `README.md`, `CLAUDE.md`, `docs/roadmap.md`
- Depends on: all
- Group: G6
- Acceptance criteria:
  - [x] `/query/stream` and the event schema are in the API contract
  - [x] `LANGSMITH_*` env vars documented (CLAUDE.md absolute constraints, README)
  - [x] Architecture section lists `stream()`, the stream endpoint and tracing
  - [x] Manual check: `curl -N` against uvicorn shows incremental events; Ctrl-C mid-stream stops Ollama generation
- Note: README gets the event table, a `curl -N` example and the LangSmith opt-in; CLAUDE.md architecture, API contract and phase status updated (`LANGSMITH_*` was already in the env var list); roadmap row 7 marked done. Manual check against the running stack (uvicorn, Ollama qwen3:0.6b, Qdrant): `curl -N` showed step_start/step_end, citations, then tokens arriving incrementally with ~100 ms spacing. Closing the pipe mid-stream ended the request and the server stayed healthy, but I did not observe on the Ollama side that generation actually stopped; that part is covered only by the R6 disconnect test (T6). Left for the Verify stage.

## ADRs
- [ADR-0017](../../adr/0017-sse-query-stream-endpoint.md): SSE `POST /query/stream` and its event schema
- [ADR-0018](../../adr/0018-stream-on-llm-provider.md): `stream()` on the LLM provider interface (amends ADR-0014)
- [ADR-0019](../../adr/0019-langsmith-opt-in-observability.md): LangSmith for observability and evaluation (opt-in)
