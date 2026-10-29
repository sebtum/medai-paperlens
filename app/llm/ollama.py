import contextlib
import json
import os
from collections.abc import AsyncIterator, Mapping, Sequence
from types import TracebackType
from typing import Any

import httpx
from langsmith import traceable

from app.llm.base import T, join_chunks


class OllamaClient:
    def __init__(self, base_url: str, model: str, think: bool | None = None) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._think = think
        self._http: httpx.AsyncClient | None = None

    @property
    def model(self) -> str:
        return self._model

    async def __aenter__(self) -> OllamaClient:
        self._http = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=30.0, read=None, write=30.0, pool=30.0)
        )
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

    def _build_request(
        self,
        prompt: str | Sequence[Mapping[str, str]],
        *,
        stream: bool,
        options: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        payload: dict[str, Any] = {
            "model": self._model,
            "stream": stream,
            "options": options,
        }
        if isinstance(prompt, str):
            path = "/api/generate"
            payload["prompt"] = prompt
        else:
            path = "/api/chat"
            payload["messages"] = list(prompt)
        if self._think is not None:
            payload["think"] = self._think
        return f"{self._base_url}{path}", payload

    @traceable(run_type="llm", name="ollama.generate")
    async def generate(
        self,
        prompt: str | Sequence[Mapping[str, str]],
        **kwargs: Any,
    ) -> str:
        if self._http is None:
            raise RuntimeError("OllamaClient must be used as async context manager")
        url, payload = self._build_request(prompt, stream=False, options=kwargs)
        response = await self._http.post(url, json=payload)
        response.raise_for_status()
        body = response.json()
        if isinstance(prompt, str):
            return str(body["response"])
        return str(body["message"]["content"])

    @traceable(run_type="llm", name="ollama.stream", reduce_fn=join_chunks)
    async def stream(
        self,
        prompt: str | Sequence[Mapping[str, str]],
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Yield answer text chunks; ``thinking`` output is never yielded."""
        if self._http is None:
            raise RuntimeError("OllamaClient must be used as async context manager")
        url, payload = self._build_request(prompt, stream=True, options=kwargs)
        async with self._http.stream("POST", url, json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                chunk = json.loads(line)
                if isinstance(prompt, str):
                    text = chunk.get("response", "")
                else:
                    text = chunk.get("message", {}).get("content", "")
                if text:
                    yield str(text)

    async def warmup(self) -> None:
        if self._http is None:
            return
        with contextlib.suppress(httpx.HTTPError):
            await self._http.post(
                f"{self._base_url}/api/generate",
                json={
                    "model": self._model,
                    "prompt": "",
                    "stream": False,
                    "keep_alive": -1,
                },
                timeout=300.0,
            )

    async def is_model_warm(self) -> bool:
        if self._http is None:
            return False
        try:
            response = await self._http.get(f"{self._base_url}/api/ps")
            response.raise_for_status()
            models: list[dict[str, Any]] = response.json().get("models", [])
            target = self._model if ":" in self._model else f"{self._model}:latest"
            return any(
                (n if ":" in n else f"{n}:latest") == target
                for m in models
                if (n := m.get("name", ""))
            )
        except httpx.HTTPError, ValueError:
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


def make_ollama_client() -> OllamaClient:
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    model = os.environ.get("OLLAMA_MODEL", "qwen3:0.6b")
    think = os.environ.get("OLLAMA_THINK", "false").strip().lower() == "true"
    return OllamaClient(base_url, model, think=think)
