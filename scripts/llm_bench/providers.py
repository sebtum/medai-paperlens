"""LLM provider construction for benchmark/eval scripts.

GeminiClient itself now lives in app/llm/gemini.py (ADR-0016: Gemini is an
opt-in *application* provider, not benchmark-only). BenchGeminiClient here
adds only the benchmark-specific generate_with_metrics() on top of it, so
that method stays out of app/.
"""

import os
from typing import Any

import httpx

from app.llm.gemini import GeminiClient
from app.llm.ollama import OllamaClient


class BenchGeminiClient(GeminiClient):
    """GeminiClient plus a metrics-surfacing generate call, for benchmarking only."""

    async def generate_with_metrics(self, prompt: str) -> dict[str, Any]:
        """Like generate(), but also surfaces token usage for benchmarking.

        Gemini doesn't report a generation-only duration separate from the
        network round trip, so gen_duration_s is left None - tokens/sec for
        Gemini has to be derived from wall-clock latency instead.
        """
        if self._client is None:
            raise RuntimeError("GeminiClient must be used as async context manager")
        response = await self._client.aio.models.generate_content(
            model=self._model, contents=prompt
        )
        usage = getattr(response, "usage_metadata", None)
        return {
            "output": str(response.text),
            "tokens_generated": getattr(usage, "candidates_token_count", None),
            "gen_duration_s": None,
            "prompt_tokens": getattr(usage, "prompt_token_count", None),
            "thinking_chars": None,
            "thinking_tokens": getattr(usage, "thoughts_token_count", None),
        }


async def ollama_generate_with_metrics(
    http: httpx.AsyncClient,
    base_url: str,
    model: str,
    prompt: str,
    think: bool | None = None,
    keep_alive: int = 0,
) -> dict[str, Any]:
    """Direct /api/generate call preserving fields OllamaClient.generate()
    discards: eval_count/eval_duration (real tokens/sec, not a char-count
    proxy) and thinking (non-empty when the model reasoned before answering,
    even though the prompt asks it not to via a plain-text '/no_think' hint
    rather than Ollama's `think` API parameter).

    `think` sets Ollama's actual `think` API parameter explicitly (True/False)
    so callers can compare it against the prompt-text '/no_think' hint alone;
    leaving it None omits the field and falls back to the model's default.

    `keep_alive` defaults to 0 (unload the model immediately after this
    call) so benchmarking multiple models never keeps more than one
    resident in RAM at a time - important on memory-constrained machines,
    at the cost of a reload (counted in `latency_s`, not in the reported
    tokens/sec, which is derived from Ollama's own generation-only
    eval_duration) on every single call.
    """
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "keep_alive": keep_alive,
    }
    if think is not None:
        payload["think"] = think
    response = await http.post(f"{base_url}/api/generate", json=payload)
    response.raise_for_status()
    data = response.json()
    thinking = data.get("thinking") or ""
    eval_duration = data.get("eval_duration")
    return {
        "output": str(data.get("response", "")),
        "tokens_generated": data.get("eval_count"),
        "gen_duration_s": (eval_duration / 1e9) if eval_duration else None,
        "prompt_tokens": data.get("prompt_eval_count"),
        "thinking_chars": len(thinking),
        "thinking_tokens": None,
    }


async def get_ollama_model_size(
    http: httpx.AsyncClient, base_url: str, model: str
) -> int | None:
    """Reported resident size (bytes) of a currently-loaded Ollama model, or
    None if it isn't loaded (e.g. queried before the first request warms it).
    """
    target = model if ":" in model else f"{model}:latest"
    try:
        response = await http.get(f"{base_url}/api/ps")
        response.raise_for_status()
        for entry in response.json().get("models", []):
            name = entry.get("name", "")
            normalized = name if ":" in name else f"{name}:latest"
            if normalized == target:
                return entry.get("size")
    except (httpx.HTTPError, ValueError):
        return None
    return None


def build_provider(spec: str) -> tuple[Any, str, str]:
    """Parse 'provider:model' (e.g. 'ollama:qwen3.5:4b', 'gemini:gemini-2.0-flash')
    into (client, provider_name, model_name). The client is an async context
    manager satisfying LlmProvider - callers must `async with` it.
    """
    provider_name, sep, model_name = spec.partition(":")
    if not sep:
        raise ValueError(f"Invalid provider spec {spec!r}, expected 'provider:model'")

    if provider_name == "ollama":
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        return OllamaClient(base_url, model_name), provider_name, model_name
    if provider_name == "gemini":
        return BenchGeminiClient(model_name), provider_name, model_name

    raise ValueError(f"Unknown provider {provider_name!r} in spec {spec!r}")
