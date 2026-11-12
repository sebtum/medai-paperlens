import asyncio
import os
from functools import lru_cache
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
from sentence_transformers import SentenceTransformer

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


@runtime_checkable
class EmbeddingProvider(Protocol):
    async def embed(self, text: str) -> npt.NDArray[np.float32]: ...


class SentenceTransformerProvider:
    def __init__(self, model: SentenceTransformer) -> None:
        self._model = model

    async def embed(self, text: str) -> npt.NDArray[np.float32]:
        return await embed_async(text, self._model)


def get_model_name() -> str:
    """Embedding model name from `EMBEDDING_MODEL`, else the default (ADR-0022)."""
    return os.getenv("EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    return SentenceTransformer(get_model_name())


@lru_cache(maxsize=1)
def get_embedding_provider() -> EmbeddingProvider:
    return SentenceTransformerProvider(get_model())


def embed(text: str, model: SentenceTransformer) -> npt.NDArray[np.float32]:
    return np.asarray(model.encode(text), dtype=np.float32)


async def embed_async(text: str, model: SentenceTransformer) -> npt.NDArray[np.float32]:
    return await asyncio.to_thread(embed, text, model)
