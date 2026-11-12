from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def join_chunks(chunks: Sequence[str]) -> str:
    """Reducer for traced streams: record the full answer as one run output."""
    return "".join(chunks)


@runtime_checkable
class LlmProvider(Protocol):
    async def generate(
        self,
        prompt: str | Sequence[Mapping[str, str]],
        **kwargs: Any,
    ) -> str: ...

    def stream(
        self,
        prompt: str | Sequence[Mapping[str, str]],
        **kwargs: Any,
    ) -> AsyncIterator[str]: ...

    async def generate_structured(
        self,
        prompt: str | Sequence[Mapping[str, str]],
        response_model: type[T],
        **kwargs: Any,
    ) -> T: ...


@runtime_checkable
class ModelStatusProvider(Protocol):
    model: str

    async def warmup(self) -> None: ...

    async def is_model_warm(self) -> bool: ...
