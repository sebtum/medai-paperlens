import asyncio
import os
from functools import lru_cache
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
from sentence_transformers import SentenceTransformer

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
"""The BGE retrieval instruction. Put in front of queries only, never documents."""


@runtime_checkable
class EmbeddingProvider(Protocol):
    async def embed(self, text: str) -> npt.NDArray[np.float32]: ...


class SentenceTransformerProvider:
    """Query embedder over a `SentenceTransformer` (ADR-0012, ADR-0025).

    `embed()` is the query side: `query_prefix` is put in front of the text. Documents
    are embedded by ingestion straight through `model.encode()` and never see the
    prefix, so changing the prefix needs no re-ingest.
    """

    def __init__(self, model: SentenceTransformer, query_prefix: str = "") -> None:
        self._model = model
        self._query_prefix = query_prefix

    async def embed(self, text: str) -> npt.NDArray[np.float32]:
        return await embed_async(self._query_prefix + text, self._model)


def get_model_name() -> str:
    """Embedding model name from `EMBEDDING_MODEL`, else the default (ADR-0022)."""
    return os.getenv("EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL


def get_query_prefix(model_name: str | None = None) -> str:
    """Query prefix from `EMBEDDING_QUERY_PREFIX` (ADR-0025).

    A set variable is used as is, even when empty (empty switches the prefix off). When
    it is unset, BGE models (name starts with `BAAI/bge-`) get `BGE_QUERY_PREFIX` and
    any other model gets none. `model_name` defaults to the configured embedding model.
    """
    explicit = os.environ.get("EMBEDDING_QUERY_PREFIX")
    if explicit is not None:
        return explicit
    name = get_model_name() if model_name is None else model_name
    return BGE_QUERY_PREFIX if name.startswith("BAAI/bge-") else ""


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    return SentenceTransformer(get_model_name())


@lru_cache(maxsize=1)
def get_embedding_provider() -> EmbeddingProvider:
    return SentenceTransformerProvider(get_model(), query_prefix=get_query_prefix())


def embed(text: str, model: SentenceTransformer) -> npt.NDArray[np.float32]:
    return np.asarray(model.encode(text), dtype=np.float32)


async def embed_async(text: str, model: SentenceTransformer) -> npt.NDArray[np.float32]:
    return await asyncio.to_thread(embed, text, model)
