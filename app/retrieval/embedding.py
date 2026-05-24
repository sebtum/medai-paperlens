import asyncio
import os
from functools import lru_cache

import numpy as np
import numpy.typing as npt
from sentence_transformers import SentenceTransformer

from app.retrieval.ingestion import EMBEDDING_MODEL


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    model_name = os.getenv("EMBEDDING_MODEL", EMBEDDING_MODEL)
    return SentenceTransformer(model_name)


def embed(text: str, model: SentenceTransformer) -> npt.NDArray[np.float32]:
    return np.asarray(model.encode(text), dtype=np.float32)


async def embed_async(text: str, model: SentenceTransformer) -> npt.NDArray[np.float32]:
    """Offloads CPU-bound encoding to a thread pool to avoid blocking the event loop."""
    return await asyncio.to_thread(embed, text, model)
