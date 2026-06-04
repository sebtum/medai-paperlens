# ADR-012: EmbeddingProvider Protocol for decoupled embedding inference

## Status

Accepted

## Context

Phase 4 introduces the `retrieve` workflow node, which must produce vector embeddings
for a user query before searching Qdrant. Two fundamentally different execution models
exist for computing embeddings:

1. **Local CPU inference** — a `SentenceTransformer` model runs in a thread pool
   (`asyncio.to_thread`) because `model.encode()` is CPU-bound and would block the
   event loop if called directly from async code.

2. **Remote API inference** — an external embedding endpoint (e.g., Ollama's
   `/api/embeddings`, OpenAI Embeddings) is I/O-bound and should use native async
   HTTP, not a thread pool.

Injecting `SentenceTransformer` directly into `make_retrieve_node` couples the
workflow to a single library and forces the threading concern into the node itself.
It also prevents testing the node without loading a real model.

## Decision

Introduce an `EmbeddingProvider` Protocol in `app/retrieval/embedding.py`:

```python
class EmbeddingProvider(Protocol):
    async def embed(self, text: str) -> npt.NDArray[np.float32]: ...
```

The concrete `SentenceTransformerProvider` wraps the existing `embed_async` helper,
which already handles `asyncio.to_thread` for CPU-bound encoding:

```python
class SentenceTransformerProvider:
    def __init__(self, model: SentenceTransformer) -> None: ...
    async def embed(self, text: str) -> npt.NDArray[np.float32]:
        return await embed_async(text, self._model)
```

A `get_embedding_provider()` factory (using `@lru_cache`) is the FastAPI dependency
for the web API path. The ingestion script continues to use `get_model()` directly,
since `ingest()` is synchronous and runs outside the event loop.

## Alternatives Considered

### Inject SentenceTransformer directly

Pros:
- No additional abstraction layer
- Simpler for the current single-provider case

Cons:
- Binds the workflow to `sentence-transformers`
- Forces threading logic into the node
- Cannot be swapped without changing `retrieve.py` signatures
- Cannot be mocked as an async provider in tests without extra boilerplate

### Use Ollama for embeddings (single provider for both LLM and embeddings)

Pros:
- Single runtime dependency (Ollama)
- Consistent async HTTP pattern with `OllamaClient`

Cons:
- Ollama embedding models are larger and slower than dedicated sentence-transformers
- Adds a second Ollama round-trip before every retrieval
- Creates a hard dependency on Ollama even for the retrieval path

## Consequences

Positive:
- `make_retrieve_node` depends only on the `EmbeddingProvider` protocol, not on
  any concrete library
- Future providers (Ollama, OpenAI, Cohere) require only a new class, no node changes
- Tests mock the protocol directly with `AsyncMock`, no real model needed
- CPU vs. I/O threading strategy is encapsulated per implementation

Negative:
- Adds a thin wrapper class and one extra factory function
- Two paths for model access: `get_model()` for ingestion (sync),
  `get_embedding_provider()` for web API (async)

## Follow-up

- If a second embedding provider is added, evaluate whether `make_ollama_client`
  should expose an `EmbeddingProvider` implementation for unified config.
- The `embed_async` helper can be removed if `SentenceTransformerProvider` becomes
  the only caller.
