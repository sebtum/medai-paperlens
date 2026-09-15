# ADR-0015: Default model qwen3:0.6b with thinking disabled

## Status

Accepted. Supersedes ADR-0011 and ADR-0013.

## Context

ADR-0013 set the default model to `qwen3.5:4b`, estimating 10–14 tok/s on the
primary development machine (CPU-only, Intel Iris Xe iGPU, ~8 GB available RAM).

`scripts/benchmark_llm.py` was built to verify that estimate against real
Ollama-reported throughput (`eval_count` / `eval_duration`, not a character-count
proxy) rather than continuing to plan around an assumption. Results, recorded in
`docs/benchmarks.md` and `docs/benchmarks/2026-09-15-ollama-qwen3.csv`:

| model | mode | avg rewrite | avg generate | avg tok/s |
|---|---|---|---|---|
| qwen3:0.6b | no_think | 2.2 s | 4.8 s | 47–55 |
| qwen3:1.7b | no_think | 4.3 s | 20.3 s | 21–24 |
| qwen3.5:4b | no_think | 10.4 s | 38.0 s | 8–9 |
| qwen3.5:4b | think | 440.4 s | 215.9 s | 8–9 |

`qwen3.5:4b` measures at 8–9 tok/s, below ADR-0013's 10–14 tok/s estimate, and a
single `generate` call takes 38 s — unusable for an interactive dev loop.
`qwen3:0.6b` in `no_think` mode answers in well under 10 s round-trip.

The benchmark also surfaced a separate latency bug (see ADR-0016's sibling fix
in `app/llm/ollama.py`, tracked alongside this ADR): the prompts' literal
`/no_think` text hint does nothing — only Ollama's real `think` API parameter
controls thinking mode, and `OllamaClient.generate()` never set it. Every model
above paid the full thinking-mode cost on every call regardless of the text
hint. Fixing that plumbing is a precondition for this model choice being
meaningful: without it, `qwen3:0.6b` would default into `think` mode and run at
37–38 tok/s with 3–4x the latency, comparable to `qwen3:1.7b`'s `no_think`
numbers.

## Decision

Set `OLLAMA_MODEL=qwen3:0.6b` and `OLLAMA_THINK=false`.

Ollama remains the required default provider (ADR-0007) — acquiring hands-on
Ollama experience is an explicit project goal — so the fix is a smaller model
plus a real fix to the `think` plumbing, not a switch away from Ollama.

## Alternatives Considered

### Stay on qwen3.5:4b

Pros:
- Better raw answer quality per ADR-0013's reasoning

Cons:
- 8–9 tok/s measured, not the 10–14 tok/s assumed; 38 s per `generate` call is
  not interactive

### Drop to qwen3:1.7b

Pros:
- Meaningfully stronger than 0.6b at long-form synthesis
- Still usable: 20.3 s avg `generate` in `no_think` mode

Cons:
- 4–5x slower than 0.6b for a synthesis quality gain that has not been measured
  (no quality eval exists yet — see Follow-up)

### gemma3:1b / llama3.2:1b

Pros:
- Candidates surveyed in `docs/benchmarks.md`; gemma3:1b in particular has
  strong instruction-following per parameter

Cons:
- Not benchmarked on this hardware yet (explicitly out of scope for this
  round — see `plan_llm_model_decision.md`); no thinking-mode toggle
  (gemma3:1b) to compare against qwen3's hybrid mode

### Hosted-only (drop local Ollama requirement)

Pros:
- Sidesteps local hardware constraints entirely

Cons:
- Contradicts the explicit project goal of hands-on Ollama experience
- Ollama must remain the required default per ADR-0007; a hosted provider is
  opt-in only (see ADR-0016)

## Consequences

Positive:
- Real interactive dev loop: ~4.8 s avg `generate`, ~2.2 s avg `rewrite`
- `OLLAMA_THINK=false` plumbed correctly through `OllamaClient` at the Ollama
  payload's top level, not silently swallowed into `options`
- Frees the `/no_think` prompt-text hint, which was Qwen-specific and would
  have been actively wrong once Gemini (ADR-0016) is a supported provider

Negative:
- `qwen3:0.6b` is the weakest model in the surveyed corpus at long-form
  grounded synthesis (see `docs/benchmarks.md`'s alternatives table); the
  generate prompt needs to stay tight and evidence-focused or output quality
  may drift
- Answer *quality* has not been measured yet, only speed — `scripts/eval_llm.py`
  exists for a human-scored groundedness pass but running it is deferred (see
  Follow-up)

## Follow-up

- Run `scripts/eval_llm.py` against qwen3:0.6b once the prototype (through
  Phase 6, Docker Compose) is stable, and re-evaluate against gemma3:1b and
  llama3.2:1b if groundedness looks weak.
- Add a startup log line that prints the active model name (carried over from
  ADR-0011 and ADR-0013 — still not implemented).
