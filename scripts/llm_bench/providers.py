"""LLM provider construction for benchmark/eval scripts.

GeminiClient is a benchmarking-only adapter - it is never imported by app/,
keeping Ollama the sole default provider (ADR-007). The google-genai package
is an optional extra (see pyproject.toml `gemini` extra) and is only imported
lazily, so this module loads fine even when it isn't installed.
"""

import os
from collections.abc import Mapping, Sequence
from types import TracebackType
from typing import Any

import httpx

from app.llm.base import T
from app.llm.ollama import OllamaClient


class GeminiClient:
    """Thin adapter over the google-genai SDK, structurally matching LlmProvider."""

    def __init__(self, model: str, api_key: str | None = None) -> None:
        self._model = model
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not self._api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        self._client: Any = None

    @property
    def model(self) -> str:
        return self._model

    async def __aenter__(self) -> GeminiClient:
        from google import genai  # optional dep, imported lazily

        self._client = genai.Client(api_key=self._api_key)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self._client = None

    async def generate(
        self,
        prompt: str | Sequence[Mapping[str, str]],
        **kwargs: Any,
    ) -> str:
        if self._client is None:
            raise RuntimeError("GeminiClient must be used as async context manager")
        contents = (
            prompt
            if isinstance(prompt, str)
            else "\n\n".join(m["content"] for m in prompt)
        )
        response = await self._client.aio.models.generate_content(
            model=self._model, contents=contents
        )
        return str(response.text)

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

    async def generate_structured(
        self,
        prompt: str | Sequence[Mapping[str, str]],
        response_model: type[T],
        **kwargs: Any,
    ) -> T:
        raise NotImplementedError(
            "generate_structured is not needed for benchmark/eval scripts."
        )


async def ollama_generate_with_metrics(
    http: httpx.AsyncClient, base_url: str, model: str, prompt: str
) -> dict[str, Any]:
    """Direct /api/generate call preserving fields OllamaClient.generate()
    discards: eval_count/eval_duration (real tokens/sec, not a char-count
    proxy) and thinking (non-empty when the model reasoned before answering,
    even though the prompt asks it not to via a plain-text '/no_think' hint
    rather than Ollama's `think` API parameter).
    """
    response = await http.post(
        f"{base_url}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
    )
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
        return GeminiClient(model_name), provider_name, model_name

    raise ValueError(f"Unknown provider {provider_name!r} in spec {spec!r}")
