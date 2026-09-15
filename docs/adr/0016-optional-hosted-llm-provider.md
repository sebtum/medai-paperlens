# ADR-0016: Optional hosted LLM provider (Gemini) behind LLM_PROVIDER

## Status

Accepted. Extends, does not supersede, ADR-0007.

## Context

ADR-0007 established Ollama as the default local LLM provider and explicitly
left the door open for more:

> External providers such as OpenRouter may be added later as optional
> integrations, but they must be disabled by default.

`scripts/llm_bench/providers.py` already had a `GeminiClient` adapter built for
benchmarking (`scripts/benchmark_llm.py`, `scripts/eval_llm.py`), structurally
matching `LlmProvider`. Promoting it to an application-usable provider needed
only a dispatch point, not new client code.

Groq was considered as an alternative hosted provider — it serves open weights
(Llama, Qwen3, gpt-oss) at 300–1200 tok/s and does not train on submitted data —
but was deferred because it would require new client code from scratch, while
the Gemini scaffolding already existed and was already exercised by the
benchmark harness.

## Decision

Add an `LLM_PROVIDER` environment variable, default `"ollama"`, accepted values
`ollama` | `gemini`. `app/llm/factory.py`'s `make_llm_provider()` dispatches on
it; an unknown value raises a clear `ValueError` rather than silently falling
back to Ollama.

Gemini access lives in `app/llm/gemini.py` (`GeminiClient`), constructed via
`make_gemini_client()` reading `GEMINI_MODEL` (default `gemini-2.5-flash`) and
`GEMINI_API_KEY`. The `google-genai` SDK stays an optional dependency
(`pyproject.toml`'s `gemini` extra) and is imported lazily inside
`GeminiClient.__aenter__`, so `app/` still imports cleanly without it installed.
Setting `LLM_PROVIDER=gemini` without `GEMINI_API_KEY` set raises `RuntimeError`
at startup — opt-in must fail loudly, not fall back silently to a provider the
operator didn't ask for.

## Alternatives Considered

### Groq

Pros:
- Open weights (same "weights, two runtimes" story as local Ollama)
- Does not train on submitted data
- 300–1200 tok/s — fastest hosted option surveyed

Cons:
- No existing client code; would need a new adapter built from scratch
- Per-model daily token cap is the binding limit for this project's RAG usage
  (each `generate` prompt carries several paper abstracts)

### Support both Gemini and Groq now

Pros:
- More choice for users with API keys already

Cons:
- Twice the new surface area (two adapters, two env-var sets, two test paths)
  for a feature that is opt-in and not on the critical path
- The existing Gemini scaffolding is enough to prove the `LLM_PROVIDER`
  dispatch pattern; a second provider can reuse it later without redesign

### Local-only (no hosted provider)

Pros:
- Smallest surface area; stays fully within ADR-0007's original scope

Cons:
- No way to compare local-model synthesis quality against a stronger hosted
  model without leaving the app, which the benchmark/eval scripts already do
  informally — wiring it into the app itself costs little given the existing
  `GeminiClient`

## Consequences

Positive:
- Ollama stays the required, always-available default — no behavior change
  for the common path
- Gemini is available for ad hoc quality comparison without leaving the app
- `ModelStatusProvider` protocol (`app/llm/base.py`) lets `/model/status` work
  against either provider: `GeminiClient.warmup()` is a no-op and
  `is_model_warm()` always returns `True` since a hosted API has no
  local cold-start concept

Negative:
- Free-tier Gemini content may be used by Google to improve its products, and
  Gemini's weights are closed. This is acceptable only because the corpus is
  public medical AI papers and ADR-0003 already forbids patient data / PHI from
  ever reaching this system — this constraint must not be silently assumed if
  the corpus scope ever changes
- Two LLM code paths now exist in `app/`, both must be kept passing the same
  `LlmProvider` protocol tests

## Follow-up

- If a second hosted provider becomes worth adding (e.g. Groq, once its
  per-model token cap is confirmed workable for this project's prompt sizes),
  extend `make_llm_provider()`'s dispatch rather than inventing a registry.
- Re-verify the "free-tier data use" and rate-limit claims above against each
  provider's current terms before relying on them for anything beyond
  development-time comparison — these change every few months.
