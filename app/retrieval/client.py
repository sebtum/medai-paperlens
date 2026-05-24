import os
from functools import lru_cache

from qdrant_client import QdrantClient


@lru_cache(maxsize=1)
def get_client() -> QdrantClient:
    url = os.getenv("QDRANT_URL", "http://localhost:6333")
    return QdrantClient(url=url)
