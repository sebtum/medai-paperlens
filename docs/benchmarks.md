# LLM Benchmarks

Speed/stability measurements behind [ADR-0015](adr/0015-default-model-qwen3-06b-no-thinking.md)
(default model) and [ADR-0016](adr/0016-optional-hosted-llm-provider.md) (opt-in
Gemini provider). This covers speed only — see [Follow-up](#follow-up) for why
answer quality is a separate, deferred effort.

## Hardware & method

- **CPU:** 11th Gen Intel Core i7-1165G7 @ 2.80GHz (4 cores / 8 threads),
  integrated Iris Xe graphics, no dedicated GPU
- **RAM:** 16 GB total, ~8 GB available at runtime
- **Ollama:** 0.34.0, `keep_alive=0` between calls so only one model is
  resident in RAM at a time — the reload cost lands in `latency_s`, not in the
  reported tok/s (which comes from Ollama's own generation-only
  `eval_duration`, not wall-clock)
- **Corpus:** 12 fixture questions (`scripts/llm_bench/fixtures.py`) × 2
  stages (`rewrite`, `generate`), run against the exact prompts the LangGraph
  pipeline uses (`app.workflow.rewrite._PROMPT`, `app.workflow.generate._PROMPT`)
- **Retrieval faked:** fixture citations are hardcoded, so no Qdrant/embedding
  variance enters the measurement — this harness isolates LLM speed only
- **Real tokens/sec:** Ollama's `/api/generate` response includes `eval_count`
  (tokens generated) and `eval_duration` (generation-only time in nanoseconds);
  both are read directly rather than estimated from character counts
- **Reproduce:**
  ```powershell
  .\.venv\Scripts\python.exe -m scripts.benchmark_llm `
    --models ollama:qwen3:0.6b,ollama:qwen3:1.7b,ollama:qwen3.5:4b --modes no_think,think
  ```
  Raw data: [`docs/benchmarks/2026-09-15-ollama-qwen3.csv`](benchmarks/2026-09-15-ollama-qwen3.csv)
  (144 rows: 3 models × 2 modes × 12 questions × 2 stages, 0 failures).

## Results

Averaged over 12 questions per (model, mode, stage):

| model | mode | stage | avg latency | avg tok/s | avg thinking chars |
|---|---|---|---:|---:|---:|
| qwen3:0.6b | no_think | rewrite | 2.2 s | 54.7 | 0 |
| qwen3:0.6b | no_think | generate | 4.8 s | 47.0 | 0 |
| qwen3:0.6b | think | rewrite | 8.4 s | 37.6 | 1,072 |
| qwen3:0.6b | think | generate | 14.5 s | 38.1 | 1,480 |
| qwen3:1.7b | no_think | rewrite | 4.3 s | 23.5 | 0 |
| qwen3:1.7b | no_think | generate | 20.3 s | 20.8 | 0 |
| qwen3:1.7b | think | rewrite | 20.4 s | 20.2 | 1,612 |
| qwen3:1.7b | think | generate | 45.6 s | 18.4 | 2,019 |
| qwen3.5:4b | no_think | rewrite | 10.4 s | 9.1 | 0 |
| qwen3.5:4b | no_think | generate | 38.0 s | 8.3 | 0 |
| qwen3.5:4b | think | rewrite | 440.4 s | 8.5 | 17,273 |
| qwen3.5:4b | think | generate | 215.9 s | 9.4 | 8,131 |

## Finding 1: thinking mode is the dominant cost — and the prompt-text hint does nothing

Every `no_think`-mode row has `thinking_chars = 0`; every `think`-mode row has
thousands of thinking characters — for the *same* prompt, which still carried
the literal text `/no_think` in both cases (this benchmark predates the fix
below). The text hint has zero effect; only Ollama's real `think` API
parameter (a top-level payload field, set explicitly per row by this harness)
controls it.

The clearest evidence is `qwen3.5:4b`'s `rewrite` stage in `think` mode:
**440 s average** — slower than its own `generate` stage (216 s), even though
rewrite's *job* is to return one short line. Individual runs ranged
293–565 s and 10,500–19,900 thinking characters for a task that should take
under a second. `qwen3.5:4b` in particular seems to reason far more than the
smaller models before answering even a trivial prompt.

`app/llm/ollama.py`'s `OllamaClient.generate()` never set Ollama's `think`
field — it forwarded `**kwargs` into the payload's `options` dict, but `think`
is a top-level field, so no caller could have set it even by passing a kwarg.
Every call the running app made paid the full thinking-mode cost. This is
fixed as part of this work: `OllamaClient.__init__` now accepts `think: bool |
None`, and `generate()` sets `payload["think"]` at the top level in both the
`/api/generate` and `/api/chat` branches. The now-provably-inert `/no_think`
text hint has been removed from both prompts (it was Qwen-specific and would
have been actively wrong for Gemini).

## Finding 2: qwen3.5:4b is not viable on this hardware

8.3–9.1 tok/s measured (`no_think` mode) versus the 10–14 tok/s
[ADR-0013](adr/0013-upgrade-default-model-qwen35-4b.md) assumed when it chose
this model — a 38 s `generate` call is not usable for an interactive dev loop.
`qwen3:0.6b` in `no_think` mode measures 4.8 s for the same stage: an order of
magnitude faster. See [ADR-0015](adr/0015-default-model-qwen3-06b-no-thinking.md)
for the resulting default-model change.

## Token budget for hosted providers

For RAG, the per-day **token** cap binds before the per-day **request** cap,
because each `generate` call's prompt carries several paper-abstract excerpts
as evidence, not just the user's question. Measured `prompt_tokens` on the
`generate` stage across all three local models (same fixture evidence, so
comparable): **154–197 tokens per request** (avg ~174). Gemini was not
included in this run (see Reproduce above — add a `gemini:` spec and a
`GEMINI_API_KEY` to extend this table), so budget any hosted free tier against
this per-request range plus normal daily request volume rather than assuming
it from request count alone.

## Alternatives surveyed

No new benchmarking was run for these — recorded here as ADR-0015 follow-up
candidates, not measured results.

### Small local models

| model | params | pros | cons |
|---|---|---|---|
| qwen3:0.6b | 0.6B | measured baseline, 55 tok/s, hybrid thinking toggle, 32k ctx, strong multilingual | weakest at long-form synthesis; needs a tight prompt or it drifts off the evidence |
| gemma3:1b | 1B | best instruction-following per parameter in this class; strong at extraction/summarization; 32k ctx | no thinking mode; ~2× slower than 0.6b |
| llama3.2:1b | 1B | very stable output-format adherence (good for rewrite), large nominal context | 2024-era, weaker reasoning |
| qwen2.5:1.5b | 1.5B | no thinking overhead, faster than qwen3:1.7b | superseded by qwen3 on quality |
| smollm2:1.7b | 1.7B | Apache-2.0, built for on-device | weaker than qwen3:1.7b at same size |
| deepseek-r1:1.5b | 1.5B | — | verbose math-reasoning distill, wrong tool for RAG |
| tinyllama:1.1b | 1.1B | — | obsolete |

### Free hosted APIs

Speed, free-tier shape, card requirement, and training-data use vary by
provider and change every few months — **verify against each provider's
current rate-limit page before relying on any number here.**

| provider | notes |
|---|---|
| Gemini | free tier may use submitted content to improve Google products; closed weights. Scaffolding already exists (`app/llm/gemini.py`); see ADR-0016 |
| Groq | does not train on submitted data; serves open weights (Llama, Qwen3, gpt-oss) at 300–1200 tok/s; per-model daily token cap is the binding limit |
| Cerebras | — |
| OpenRouter `:free` tier | — |
| GitHub Models | — |
| Mistral | — |
| Cloudflare Workers AI | — |

### Chinese models

Two distinct cases: open weights run **locally via Ollama** (Qwen3, already in
use — no privacy exposure, the weights are on disk) versus **hosted Chinese
APIs** (DeepSeek, Moonshot, Zhipu GLM-Flash, Alibaba DashScope), which are
cheap-to-free but raise a data-residency question that conflicts with
[ADR-0003](adr/0003-avoid-patient-data-and-clinical-decision-support.md)'s
privacy posture. The hosted case is not recommended for this project.

## Follow-up

Quality evaluation with `scripts/eval_llm.py` (a human-scorable groundedness
review sheet — no eval framework; at ~12 questions × N models, eyeballing beats
standing up ragas/deepeval/promptfoo) belongs **after** the prototype is
complete. Speed is now settled by this document; groundedness and
hallucination rate are not. See ADR-0015's Follow-up section.

No further speed benchmarking of gemma3:1b / llama3.2:1b is planned until that
quality eval exists and gives a reason to revisit the model choice.
