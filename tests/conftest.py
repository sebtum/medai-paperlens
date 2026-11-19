import os
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
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


class FakeReranker:
    """Shared reranker test double satisfying ``Reranker`` (no model, no network).

    ``score`` returns ``scores`` when given (one per passage, in order). Otherwise
    each passage scores the share of the query's words it contains, so a passage that
    repeats the query ranks high. With ``error`` set, ``score`` raises it. ``calls``
    records every ``(query, passages)`` pair.
    """

    def __init__(
        self,
        scores: Sequence[float] | None = None,
        error: Exception | None = None,
        model_name: str = "fake-reranker",
    ) -> None:
        self.scores = list(scores) if scores is not None else None
        self.error = error
        self._model_name = model_name
        self.calls: list[tuple[str, list[str]]] = []

    @property
    def model_name(self) -> str:
        return self._model_name

    async def score(self, query: str, passages: Sequence[str]) -> list[float]:
        self.calls.append((query, list(passages)))
        if self.error is not None:
            raise self.error
        if self.scores is not None:
            return self.scores[: len(passages)]
        words = set(query.lower().split())
        if not words:
            return [0.0 for _ in passages]
        return [len(words & set(p.lower().split())) / len(words) for p in passages]


def _refuse_real_cross_encoder(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError(
        "A test tried to load a real CrossEncoder model. Use FakeReranker."
    )


@pytest.fixture(autouse=True)
def _no_real_reranker(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """No test loads a real reranker model, whatever the developer's shell has set.

    Unset `RERANKER_MODEL` now means the default MiniLM model (ADR-0026), so the
    variable is set to `none` (off) for every test; a test that wants the default or a
    model name sets or clears it itself. On top of that the `CrossEncoder` constructor
    used by `get_reranker` raises, so even a model name never reaches a download. The
    `get_reranker` singleton cache is emptied before and after.
    """
    from app.retrieval import rerank

    monkeypatch.setenv("RERANKER_MODEL", "none")
    monkeypatch.setattr(rerank, "CrossEncoder", _refuse_real_cross_encoder)
    rerank.get_reranker.cache_clear()
    yield
    rerank.get_reranker.cache_clear()


@pytest.fixture
async def managed_app():
    """App with ASGI lifespan (startup/shutdown) properly executed."""
    async with LifespanManager(app) as manager:
        yield manager.app


# Vector size of the mocked embedder; ingestion takes the real one from the model.
TEST_VECTOR_DIM = 384
