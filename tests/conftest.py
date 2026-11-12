import os
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any

import pytest
from asgi_lifespan import LifespanManager

from app.main import app


class FakeLlm:
    """Shared LLM test double satisfying ``LlmProvider`` (no network).

    ``generate`` returns ``answer`` and ``stream`` yields ``tokens`` (default:
    ``answer`` split on whitespace, keeping the separators). With ``error`` set,
    both raise it. ``calls`` records every prompt passed to either method.
    """

    def __init__(
        self,
        answer: str = "Mocked answer.",
        tokens: Sequence[str] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.answer = answer
        self.tokens = (
            list(tokens) if tokens is not None else [w + " " for w in answer.split()]
        )
        self.error = error
        self.calls: list[str | Sequence[Mapping[str, str]]] = []

    async def generate(
        self, prompt: str | Sequence[Mapping[str, str]], **kwargs: Any
    ) -> str:
        self.calls.append(prompt)
        if self.error is not None:
            raise self.error
        return self.answer

    async def stream(
        self, prompt: str | Sequence[Mapping[str, str]], **kwargs: Any
    ) -> AsyncIterator[str]:
        self.calls.append(prompt)
        if self.error is not None:
            raise self.error
        for token in self.tokens:
            yield token

    async def generate_structured(
        self,
        prompt: str | Sequence[Mapping[str, str]],
        response_model: type,
        **kwargs: Any,
    ) -> Any:
        raise NotImplementedError


@pytest.fixture(autouse=True)
def _no_langsmith_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests never trace or touch LangSmith, whatever the developer's shell has set."""
    for name in list(os.environ):
        if name.startswith(("LANGSMITH_", "LANGCHAIN_")):
            monkeypatch.delenv(name)


@pytest.fixture
async def managed_app():
    """App with ASGI lifespan (startup/shutdown) properly executed."""
    async with LifespanManager(app) as manager:
        yield manager.app
