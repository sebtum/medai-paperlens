# ADR-011: Use qwen2.5:3b as default Ollama model

## Status

Accepted

## Context

Phase 4 adds Ollama-based LLM calls for query rewriting and answer synthesis.
The original plan used `llama3.2` as the default model name. Before writing it
into code and documentation, the hardware constraints of the primary development
machine were evaluated:

- CPU-only inference (no dedicated GPU)
- Intel Iris Xe integrated graphics (shares system RAM, no dedicated VRAM)
- 15.6 GB total RAM, ~8 GB available at runtime

Intel oneAPI / Level Zero is not installed, so no SYCL-based GPU offloading is
possible. Ollama's standard Windows binary uses Vulkan for iGPU acceleration
when available. Because the Iris Xe shares the system memory bus with the CPU,
the Vulkan speedup is approximately 1.3–1.5x (not the 4–8x seen on discrete
GPUs). The memory-bandwidth bottleneck is the same for both CPU and iGPU paths.

For this project the LLM must handle two tasks: (1) rewriting a research
question for better retrieval and (2) synthesising an answer from provided
paper excerpts (RAG). Both are instruction-following tasks over short,
structured prompts — raw factual knowledge matters less than prompt adherence.

## Decision

Use `qwen2.5:3b` as the default model via the `OLLAMA_MODEL` environment
variable.

At the 3B parameter class and Q4_K_M quantisation, `qwen2.5:3b` requires
approximately 2 GB of RAM and runs at 12–18 tokens/second on CPU (or
18–25 tok/s with Vulkan iGPU offloading). This leaves ample headroom in the
available ~8 GB.

## Alternatives Considered

### llama3.2:3b (original plan default)

Pros:
- Well-known, widely documented
- Same RAM footprint (~2 GB) and CPU speed as qwen2.5:3b

Cons:
- Consistently below qwen2.5:3b on instruction-following benchmarks
  (MT-Bench, IFEval) at the same parameter count
- Less suited to structured RAG prompts

### gemma2:2b

Pros:
- Smaller (~1.6 GB) and faster (15–25 tok/s CPU)
- Google efficiency-optimised; strong for its size

Cons:
- Slightly weaker on complex instruction tasks than qwen2.5:3b
- Less headroom if the synthesis prompt grows

### qwen2.5:7b

Pros:
- Noticeably stronger answer quality

Cons:
- ~4.5 GB RAM; leaves only ~3.5 GB headroom
- 4–7 tok/s on CPU — a 200-token response takes ~40 seconds, too slow for
  interactive use on this hardware

### llama3.1:8b

Pros:
- High benchmark scores

Cons:
- ~5 GB RAM; borderline fit
- 3–6 tok/s on CPU; ~90-second responses at 500 tokens — unusable interactively

## Consequences

Positive:
- Best instruction-following quality in the ≤2 GB RAM class
- Fits comfortably within available RAM with headroom for Qdrant and embeddings
- Interactive response times (~1–2 seconds for rewrite, ~10–15 seconds for synthesis)
- OLLAMA_MODEL env var lets any user override to a larger model on better hardware

Negative:
- 3B models occasionally produce vague or incomplete synthesis; the grounding
  validator mitigates this by checking citation coverage
- Users with discrete GPUs are not automatically upgraded to a stronger model

## Follow-up

- Re-evaluate when the project reaches Docker Compose phase: if running in a
  container on a machine with a discrete GPU, `qwen2.5:7b` or `llama3.1:8b`
  may become viable defaults.
- Add a startup log line that prints the active model name so users know which
  model is running without reading `.env`.
