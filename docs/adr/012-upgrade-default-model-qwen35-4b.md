# ADR-012: Upgrade default Ollama model to qwen3.5:4b

## Status

Accepted

## Context

ADR-011 selected `qwen2.5:3b` as the default model and included a follow-up:
> Re-evaluate when the project reaches Docker Compose phase: if running in a container on a machine with a discrete GPU, `qwen2.5:7b` or `llama3.1:8b` may become viable defaults.

Phase 5 (Docker Compose) is now underway, making this the natural re-evaluation checkpoint.

A survey of available Ollama models in June 2026 shows that `qwen3.5:4b` is available at [ollama.com/library/qwen3.5](https://ollama.com/library/qwen3.5) and represents a direct generational improvement over `qwen2.5:3b` in the instruction-following workloads this project uses (query rewriting, RAG synthesis).

Hardware constraints of the primary development machine remain unchanged from ADR-011:
- CPU-only inference (no dedicated GPU)
- Intel Iris Xe integrated graphics, ~8 GB available RAM

At Q4_K_M quantisation `qwen3.5:4b` requires approximately 2.5–3 GB RAM, leaving ample headroom for Qdrant, the embedding model, and the FastAPI/Streamlit processes.

## Decision

Upgrade the default model from `qwen2.5:3b` to `qwen3.5:4b`.

Key improvements over qwen2.5:3b:
- Better instruction-following (directly benefits the rewrite and generate nodes)
- 256 K token context window (vs. ~32 K in 2.5 series)
- Native tool-calling support (useful for future structured-output work in Phase 5)
- Same Ollama pull/run interface — no code changes beyond the default string

## Alternatives Considered

### qwen3.5:1.7b

Pros:
- ~1.2 GB RAM, faster than 4b on CPU

Cons:
- Noticeably weaker synthesis quality than 4b; not worth the regression

Available as `OLLAMA_MODEL=qwen3.5:1.7b` override on very memory-constrained machines.

### gemma4:e4b (Google, released April 2026)

Pros:
- Strong agent/tool-calling capabilities
- Apache 2.0 license

Cons:
- ~3 GB RAM (slightly larger than qwen3.5:4b)
- Community benchmarks show qwen3.5 superior on RAG instruction-following tasks

### qwen3.5:9b / qwen2.5:7b

Pros:
- Higher answer quality

Cons:
- ~5–6 GB RAM; leaves insufficient headroom on the primary dev machine
- 3–6 tok/s on CPU — responses would exceed 60 seconds

## Thinking Mode Note

The Qwen3 family supports an optional "thinking mode" (`<think>` tags in output). In `qwen3.5:4b` this mode is **disabled by default** when called via `/api/generate` or `/api/chat` without explicit `think: true`. No code changes are required, but smoke-test responses should be inspected for stray `<think>` blocks on first run.

## Consequences

Positive:
- Better query rewriting and answer synthesis quality
- Larger context window reduces risk of prompt truncation on longer paper excerpts
- `OLLAMA_MODEL` env var still allows per-machine override

Negative:
- ~0.5–1 GB additional RAM usage compared to qwen2.5:3b
- Slightly slower token generation on CPU (~10–14 tok/s vs 12–18 tok/s)
- Users must run `ollama pull qwen3.5:4b` before first use

## Follow-up

- Add a startup log line that prints the active model name (carry-over from ADR-011).
- Re-evaluate with a discrete GPU setup once Docker Compose is stable: `qwen3.5:9b` or larger may become viable defaults in that configuration.
