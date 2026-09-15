"""Opt-in Gemini provider (ADR-0007 extension, ADR-0016).

Ollama remains the required default; this provider is only constructed when
LLM_PROVIDER=gemini is set explicitly (see app/llm/factory.py). The
google-genai package is an optional extra (see pyproject.toml `gemini`
extra) and is only imported lazily, so this module loads fine even when it
isn't installed.
"""

import os
from collections.abc import Mapping, Sequence
from types import TracebackType
from typing import Any

from app.llm.base import T


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

    async def generate_structured(
        self,
        prompt: str | Sequence[Mapping[str, str]],
        response_model: type[T],
        **kwargs: Any,
    ) -> T:
        raise NotImplementedError(
            "generate_structured is not implemented for GeminiClient."
        )

    async def warmup(self) -> None:
        """No-op: Gemini is a hosted API with no local model to warm."""
        return None

    async def is_model_warm(self) -> bool:
        """Always True: there is no cold-start concept for a hosted API."""
        return True
