import os
from collections.abc import Mapping, Sequence
from types import TracebackType
from typing import Any, cast

import httpx
from fastapi import Request

from app.llm.base import T


class OllamaClient:
    def __init__(self, base_url: str, model: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._http: httpx.AsyncClient | None = None

    @property
    def model(self) -> str:
        return self._model

    async def __aenter__(self) -> OllamaClient:
        self._http = httpx.AsyncClient(timeout=60.0)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    async def generate(
        self,
        prompt: str | Sequence[Mapping[str, str]],
        **kwargs: Any,
    ) -> str:
        if self._http is None:
            raise RuntimeError("OllamaClient must be used as async context manager")
        if isinstance(prompt, str):
            payload: dict[str, Any] = {
                "model": self._model,
                "prompt": prompt,
                "stream": False,
                "options": kwargs,
            }
            response = await self._http.post(
                f"{self._base_url}/api/generate", json=payload
            )
            response.raise_for_status()
            return str(response.json()["response"])
        else:
            payload = {
                "model": self._model,
                "messages": list(prompt),
                "stream": False,
                "options": kwargs,
            }
            response = await self._http.post(
                f"{self._base_url}/api/chat", json=payload
            )
            response.raise_for_status()
            return str(response.json()["message"]["content"])

    async def is_model_warm(self) -> bool:
        if self._http is None:
            return False
        try:
            response = await self._http.get(f"{self._base_url}/api/ps")
            response.raise_for_status()
            models: list[dict[str, Any]] = response.json().get("models", [])
            target = (
                self._model if ":" in self._model else f"{self._model}:latest"
            )
            return any(
                (n if ":" in n else f"{n}:latest") == target
                for m in models
                if (n := m.get("name", ""))
            )
        except (httpx.HTTPError, ValueError):
            return False

    async def generate_structured(
        self,
        prompt: str | Sequence[Mapping[str, str]],
        response_model: type[T],
        **kwargs: Any,
    ) -> T:
        raise NotImplementedError(
            "generate_structured will be implemented in Phase 5 "
            "using Ollama structured output (format: json-schema)."
        )


def get_ollama_client(request: Request) -> OllamaClient:
    return cast(OllamaClient, request.app.state.ollama)


def make_ollama_client() -> OllamaClient:
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    model = os.environ.get("OLLAMA_MODEL", "qwen3.5:4b")
    return OllamaClient(base_url, model)
