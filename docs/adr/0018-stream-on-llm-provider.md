# ADR-0018: `stream()` on the LLM provider interface

## Status

Accepted (Phase 7). Amends ADR-0014.

## Context

ADR-0014 defined `LlmProvider` with `generate()` and `generate_structured()`. Token
streaming for `/query/stream` (ADR-0017) needs incremental output from the provider.
Both providers, Ollama and Gemini (ADR-0016), support native streaming.

## Decision

Add `def stream(prompt, **kwargs) -> AsyncIterator[str]` to `LlmProvider`, and
implement it as an async generator in both providers:

- `OllamaClient`: `stream: true` on `/api/generate` (string prompt) or `/api/chat`
  (messages), reading NDJSON lines. It yields only `response` / `message.content`.
  `thinking` text is never yielded as answer tokens.
- `GeminiClient`: `aio.models.generate_content_stream`, skipping empty chunks.

`generate_node` always calls `stream()`. It forwards each token to LangGraph's custom
stream writer and joins the tokens into the answer. Under `ainvoke`, used by
`/query`, the writer does nothing, so both endpoints share one code path.
`rewrite_node` keeps using `generate()`.

## Alternatives Considered

### Separate `StreamingLlmProvider` Protocol, with `generate_node` falling back to `generate()`

Pros:
- Existing test fakes and AsyncMocks keep working.

Cons:
- Two code paths in `generate_node`, plus duck-typing to pick one.
  `/query` and `/query/stream` could drift apart.

### Stream via LangGraph `messages` mode

Pros:
- Built into LangGraph.

Cons:
- Needs LangChain chat model objects. The project avoids LangChain (CLAUDE.md).

## Consequences

Positive:
- Native streaming in both providers. One generate path.
- Cancelling the async generator closes the HTTP stream, so Ollama stops generating.

Negative:
- Every test fake must implement `stream()`. A shared `FakeLlm` in
  `tests/conftest.py` replaces bare `AsyncMock` LLMs.

## Follow-up

- Stream the rewrite node's tokens only if the UI ever needs them (out of scope for Phase 7).
