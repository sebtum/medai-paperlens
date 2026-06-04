# ADR-013: LlmProvider Protocol for decoupled LLM inference

## Status

Accepted

## Context

Phase 4 introduces two workflow nodes that call an LLM: `rewrite` (query rewriting)
and `generate` (answer synthesis). Both nodes originally depended directly on
`OllamaClient`, coupling the workflow to a concrete HTTP client.

Additionally, a plain `generate(prompt: str) -> str` interface is too narrow for
a production RAG system:

- Modern LLMs use chat message format `[{"role": "user", "content": "..."}]`
  in addition to plain string completion
- Structured output (returning a Pydantic model) is needed for future evidence
  extraction nodes that must parse citations, metrics, and limitations from paper text
- LLM parameters (temperature, max_tokens) must be passable without changing
  the interface signature

## Decision

Introduce an `LlmProvider` Protocol in `app/llm/base.py`:

```python
class LlmProvider(Protocol):
    async def generate(
        self,
        prompt: str | Sequence[Mapping[str, str]],
        **kwargs: Any,
    ) -> str: ...

    async def generate_structured(
        self,
        prompt: str | Sequence[Mapping[str, str]],
        response_model: type[T],
        **kwargs: Any,
    ) -> T: ...
```

`OllamaClient` satisfies this protocol through structural subtyping. Internally it
dispatches on prompt type: strings go to `/api/generate`, chat message sequences
go to `/api/chat`. LLM parameters in `**kwargs` are forwarded via Ollama's
`options` dict.

`generate_structured` is stubbed as `NotImplementedError` in `OllamaClient` for
Phase 4 and will be implemented in Phase 5 using Ollama's `format: json-schema`
structured output capability.

## Alternatives Considered

### Single `generate(prompt: str) -> str`

Pros:
- Minimal surface area

Cons:
- Cannot express chat message format without changing the interface later
- No path to structured output without breaking changes
- Insufficient for a production agentic RAG system

### Abstract base class (ABC)

Pros:
- Explicit contract enforcement via inheritance

Cons:
- Requires `OllamaClient` to inherit from the ABC, coupling implementation to
  the abstract layer
- Protocol-based structural subtyping is the idiomatic Python approach

### Separate `TextProvider` and `StructuredProvider` protocols

Pros:
- Nodes that only need text generation declare a narrower dependency

Cons:
- Premature splitting; all planned providers will implement both methods
- Extra protocols add cognitive overhead

## Consequences

Positive:
- Workflow nodes depend on `LlmProvider`, not on Ollama's HTTP semantics
- `OllamaClient` dispatches between `/api/generate` and `/api/chat` transparently
- `generate_structured` is in the interface now; Phase 5 can implement it without
  any interface changes
- Tests mock with `MagicMock(generate=AsyncMock(...))` — no HTTP client setup needed
- Consistent with `EmbeddingProvider` pattern from ADR-012

Negative:
- `**kwargs` are passed to Ollama's `options` dict; implementations may silently
  ignore unknown kwargs — callers cannot verify that a parameter was honoured
- `generate_structured` in `OllamaClient` raises `NotImplementedError` until
  Phase 5; calling it at runtime will fail

## Follow-up

- Phase 5: implement `generate_structured` in `OllamaClient` using
  `format: {"type": "json_schema", ...}` (requires Ollama ≥ 0.5).
- See ADR-007 for the decision to use Ollama as the default provider.
- See ADR-012 for the parallel `EmbeddingProvider` abstraction.
