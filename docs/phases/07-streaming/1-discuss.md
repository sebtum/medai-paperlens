# Phase 7: Streaming — Discuss

## Goal
Show step status (rewriting, retrieving, generating) and answer tokens in the UI as
they happen. `POST /query` stays unchanged for tests and eval. Opt-in LangSmith
tracing shows per-node latency and the LLM prompts and outputs.

## Scope
### In
- New `POST /query/stream` endpoint using Server-Sent Events (SSE). Same request
  body as `/query`.
- `stream()` added to the `LlmProvider` Protocol. Native token streaming in both
  `OllamaClient` and `GeminiClient`.
- `generate_node` emits tokens through LangGraph's custom stream writer. The endpoint
  consumes `astream(stream_mode=["updates", "custom"])`.
- Event schema: `step_start` / `step_end` (node name, elapsed ms), `citations`
  (sent right after retrieve), `token`, `final` (the full `QueryResponse` payload)
  and `error`.
- The Streamlit UI always uses the stream: live step list with timings, citations,
  then tokens, then the final state.
- Opt-in LangSmith tracing: `LANGSMITH_*` env vars, plus `@traceable` on the provider
  `generate` / `stream` calls. Documented in `.env.example`, README and CLAUDE.md.

### Out
- Streaming the rewrite node's tokens (it only gets step status).
- LangSmith datasets and evaluation (Phases 9 and 11).
- WebSockets, resuming a stream, multi-turn chat.
- Changes to retrieval, prompts or the grounding check.

## Requirements
- R1 `POST /query` keeps its contract and behaviour. Existing tests pass unchanged.
- R2 `/query` and `/query/stream` return the same `final` result for the same input
  (one code path in the graph).
- R3 An unsafe query on the stream sends no tokens. The `final` event has
  `grounded: false` and `route: unsafe_medical_advice`.
- R4 The `final` event is authoritative. The UI replaces the streamed text with
  `final.answer`, which covers strip, empty answer and errors mid-stream.
- R5 An LLM error mid-stream sends an `error` event, then a `final` with the error
  answer. The connection closes cleanly.
- R6 When the client disconnects, the graph and the LLM request are cancelled. No
  orphaned Ollama generation.
- R7 Thinking tokens (`OLLAMA_THINK=true`) are never streamed as answer tokens.
- R8 Tests use mock providers only and need no LangSmith account.
  `LANGSMITH_TRACING` stays unset in CI.
- R9 With LangSmith off, the app runs fully offline. `@traceable` has no network
  side effects then.
- R10 The Docker Compose demo path still works. Proxies don't buffer SSE (correct
  headers).

## Assumptions
- A1 `get_stream_writer()` does nothing under `ainvoke` (or can be guarded), so
  `generate_node` can always call `llm.stream()` and collect the tokens.
- A2 Ollama `/api/generate` and `/api/chat` with `stream: true` return NDJSON. The
  answer text is in `response` / `message.content`, and thinking text is in a
  separate `thinking` field.
- A3 `google-genai` provides `aio.models.generate_content_stream`.
- A4 `langsmith` is already installed transitively. Making it a direct dependency
  adds no new package.

## Unknowns
- U1 SSE implementation: FastAPI's built-in SSE support vs `StreamingResponse` with
  hand-written framing vs `sse-starlette`. Resolved in Plan by checking the installed
  FastAPI version. Avoid a new dependency if possible.
- U2 Does a Streamlit fragment with `run_every=5` interrupt a long-running stream?
  Resolved by a spike at the start of Execute. Fallback: move the stream rendering
  out of the fragment, or pause `run_every` while a query is running.
- U3 Sync vs async in the UI: `st.write_stream` needs a sync generator (httpx sync
  `client.stream()`), but the UI currently uses `asyncio.run`. Resolved in Plan.
- U4 Do generator-based `@traceable` decorators aggregate stream chunks into one run
  output? Resolved by a short check in Plan.
- U5 Testing SSE in-process with `ASGITransport`: does `aiter_lines()` deliver
  events incrementally? Resolved in Plan.

## Risks
- Disconnect handling in Starlette streams is easy to get wrong (R6). Needs an
  explicit test with a mock provider that blocks.
- Changing the Protocol breaks the mock providers in the tests. Every fake needs
  `stream()`.
- LangSmith could leak prompts to an external service. That's acceptable: it's
  opt-in, the corpus is public papers and there is no PHI (ADR-0003). This goes in
  the ADR.

## ADRs to write
- ADR-0017: SSE `POST /query/stream` and its event schema.
- ADR-0018: `stream()` on the LLM provider interface (amends ADR-0014).
- ADR-0019: LangSmith for observability and evaluation (opt-in, scope per phase).
