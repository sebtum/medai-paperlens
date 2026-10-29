# ADR-0017: SSE `POST /query/stream` and its event schema

## Status

Accepted (Phase 7).

## Context

`POST /query` returns only after the whole workflow finishes. With a local model
that can take tens of seconds, and the UI shows nothing but a spinner. Phase 7 wants
the UI to show step status (rewriting, retrieving, generating) and answer tokens as
they happen. `/query` must stay unchanged because the tests and the upcoming eval
phases (9, 11) rely on it.

## Decision

Add `POST /query/stream`. It takes the same `QueryRequest` body and returns
Server-Sent Events. It uses FastAPI's built-in `EventSourceResponse` /
`ServerSentEvent` (FastAPI ≥ 0.136), which sets `Cache-Control: no-cache` and
`X-Accel-Buffering: no` and sends a keepalive comment every 15 s.

The endpoint runs the same compiled graph as `/query`, using
`astream(stream_mode=["tasks", "custom", "values"])`:

- `tasks` → `step_start {node}` and `step_end {node, elapsed_ms}`
- the retrieve task result → `citations {citations, confidence}`
- `custom` → `token {text}` and `error {message}`, written by `generate_node`
- the last `values` chunk → `final`, which carries the full `QueryResponse` payload

Each event's SSE `event:` field holds its type, and `data:` holds its JSON payload.
`final` is authoritative: clients replace streamed text with `final.answer`. An
unexpected exception sends `error` and then a `final` with the error answer and
`grounded: false`. Client disconnect cancels the graph, which also cancels the LLM
request. To make that reliable the endpoint iterates `astream` in its own asyncio
task and cancels that task when the response ends. FastAPI cancels the SSE
producer with an anyio cancel scope that keeps re-cancelling the consumer while
LangGraph cleans up, which left the node task and the LLM request running.

## Alternatives Considered

### `StreamingResponse` with hand-written SSE framing

Pros:
- Full control of framing.

Cons:
- Re-implements headers, keepalive and encoding that FastAPI already provides.

### `sse-starlette`

Pros:
- Mature, widely used.

Cons:
- New dependency for something FastAPI now has built in.

### WebSockets

Pros:
- Bidirectional.

Cons:
- Not needed for one-way, single-answer streaming. Harder to proxy and test.

### `stream_mode=["updates", "custom"]` (as drafted in Discuss)

Pros:
- Smaller payloads.

Cons:
- `updates` only reports node completion, so step start events would need
  writer calls in every node. `tasks` reports both start and end.

## Consequences

Positive:
- One graph, one code path: `/query` and `/query/stream` return the same final result.
- No new dependency. The proxy-safe headers are set by the framework.

Negative:
- Two endpoints to keep in sync, which the shared state and response builders mitigate.
- httpx's `ASGITransport` buffers responses, so the disconnect test needs a raw ASGI harness.

## Follow-up

- Revisit event ids and resume (`Last-Event-ID`) only if multi-turn chat is added.
