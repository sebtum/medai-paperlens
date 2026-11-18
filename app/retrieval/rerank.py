import asyncio
import os
from collections.abc import Sequence
from functools import lru_cache
from typing import Any, Protocol, runtime_checkable

import torch
from sentence_transformers import CrossEncoder

RERANKER_MAX_LENGTH = 512
"""Token limit per (query, passage) pair, the same for all candidate models."""

_OFF_VALUES = frozenset({"", "none"})


@runtime_checkable
class Reranker(Protocol):
    """Scores passages against a query, jointly. Higher means more relevant.

    Scores are in 0..1 (sigmoid), so every model gives comparable numbers.
    """

    @property
    def model_name(self) -> str: ...

    async def score(self, query: str, passages: Sequence[str]) -> list[float]: ...


class CrossEncoderReranker:
    """`Reranker` over a sentence-transformers `CrossEncoder` (Phase 10).

    The model reads the query and each passage together. `score()` runs it in a
    worker thread so the event loop is never blocked, and squeezes the raw output
    through a sigmoid.
    """

    def __init__(self, model: CrossEncoder, model_name: str) -> None:
        self._model = model
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    async def score(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        return await asyncio.to_thread(self._score_sync, query, list(passages))

    def _score_sync(self, query: str, passages: list[str]) -> list[float]:
        # `list[Any]`: the library types its pairs as an invariant union.
        pairs: list[Any] = [(query, passage) for passage in passages]
        scores = self._model.predict(pairs, activation_fn=torch.nn.Sigmoid())
        return [float(s) for s in scores]


def get_reranker_model_name() -> str | None:
    """Reranker model name from `RERANKER_MODEL`; `None` when the reranker is off.

    Unset, empty and `none` (any case) all mean off. The default model is chosen
    after the Phase 10 comparison, so until then the reranker is off by default.
    """
    value = os.getenv("RERANKER_MODEL", "").strip()
    return None if value.lower() in _OFF_VALUES else value


def rerank_passage(title: str, section: str, text: str) -> str:
    """The text a reranker reads for one chunk. The app and the eval both use it."""
    return f"{title}\n{section}\n{text}"


@lru_cache(maxsize=1)
def get_reranker() -> Reranker | None:
    """Singleton reranker; `None` when `RERANKER_MODEL` is off. FastAPI dependency."""
    name = get_reranker_model_name()
    if name is None:
        return None
    return CrossEncoderReranker(
        CrossEncoder(name, max_length=RERANKER_MAX_LENGTH), model_name=name
    )
