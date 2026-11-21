import asyncio
import logging
import os
import threading
import time
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

import torch
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

RERANKER_MAX_LENGTH = 512
"""Token limit per (query, passage) pair, the same for all candidate models."""

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
"""Default reranker (ADR-0026): small, about 1.7 s per request at depth 20."""

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

    Unset means `DEFAULT_RERANKER_MODEL` (ADR-0026). An empty value or `none` (any
    case) switches the reranker off.
    """
    value = os.getenv("RERANKER_MODEL", DEFAULT_RERANKER_MODEL).strip()
    return None if value.lower() in _OFF_VALUES else value


def rerank_passage(title: str, section: str, text: str) -> str:
    """The text a reranker reads for one chunk. The app and the eval both use it."""
    return f"{title}\n{section}\n{text}"


class UnavailableReranker:
    """Stands in for a reranker whose model could not be loaded.

    `score()` always raises, so the rerank step takes its normal fallback (vector
    order, `rerank_fallback` set) and the request still succeeds.
    """

    def __init__(self, model_name: str, error: Exception) -> None:
        self._model_name = model_name
        self._error = error

    @property
    def model_name(self) -> str:
        return self._model_name

    async def score(self, query: str, passages: Sequence[str]) -> list[float]:
        raise RuntimeError(
            f"reranker model {self._model_name!r} could not be loaded"
        ) from self._error


RERANKER_RETRY_SECONDS = 60.0
"""How long a failed model load is remembered before the next request retries it."""

_LOAD_ERRORS = (OSError, ValueError)
"""Errors that mean "this model cannot be loaded now": no network or cache, a bad model
name or file (Hugging Face Hub errors are `OSError`s). Anything else is a bug and is
not turned into a fallback."""

_state_lock = threading.Lock()
"""Guards `_loaded` and `_failed`. Held for an instant, never during a model load."""
_load_lock = threading.Lock()
"""Held by the one request that loads the model; other requests do not wait for it."""
_loaded: Reranker | None = None
_failed: tuple[float, Reranker] | None = None


def reset_reranker_cache() -> None:
    """Forget the loaded reranker and any remembered load failure (tests)."""
    global _loaded, _failed
    with _state_lock:
        _loaded = None
        _failed = None


def _cached_reranker(name: str) -> Reranker | None:
    """The loaded reranker or a remembered failure for `name`. Hold `_state_lock`."""
    if _loaded is not None and _loaded.model_name == name:
        return _loaded
    if (
        _failed is not None
        and _failed[1].model_name == name
        and time.monotonic() - _failed[0] < RERANKER_RETRY_SECONDS
    ):
        return _failed[1]
    return None


def get_reranker() -> Reranker | None:
    """Singleton reranker; `None` when `RERANKER_MODEL` is off. FastAPI dependency.

    A model that cannot be loaded (offline first start, bad name) never fails the
    request: an `UnavailableReranker` is returned instead, and the failure is
    remembered for `RERANKER_RETRY_SECONDS` so a broken model is not retried on every
    request. A successful load is kept for the life of the process.

    Only one request loads the model at a time. A request that arrives while another
    one is loading gets an `UnavailableReranker` at once (vector order for that
    request) instead of waiting for the download. Errors that are not load errors
    (see `_LOAD_ERRORS`) are bugs and propagate.
    """
    global _loaded, _failed
    name = get_reranker_model_name()
    if name is None:
        return None
    with _state_lock:
        cached = _cached_reranker(name)
    if cached is not None:
        return cached
    if not _load_lock.acquire(blocking=False):
        return UnavailableReranker(
            name, RuntimeError("the model is still loading in another request")
        )
    try:
        with _state_lock:  # another request may have finished while this one waited
            cached = _cached_reranker(name)
        if cached is not None:
            return cached
        try:
            reranker = CrossEncoderReranker(
                CrossEncoder(name, max_length=RERANKER_MAX_LENGTH), model_name=name
            )
        except _LOAD_ERRORS as exc:
            logger.exception(
                "get_reranker: could not load %r, reranking falls back to vector order",
                name,
            )
            stand_in = UnavailableReranker(name, exc)
            with _state_lock:
                _failed = (time.monotonic(), stand_in)
            return stand_in
        with _state_lock:
            _loaded = reranker
            _failed = None
        return reranker
    finally:
        _load_lock.release()
