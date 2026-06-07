from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.llm.ollama import OllamaClient, get_ollama_client

router = APIRouter()

_COLD_START_ESTIMATE_S: int = 120


class ModelStatusResponse(BaseModel):
    warm: bool
    model: str
    estimated_warmup_seconds: int | None


@router.get("/status", response_model=ModelStatusResponse)
async def model_status(
    ollama: Annotated[OllamaClient, Depends(get_ollama_client)],
) -> ModelStatusResponse:
    warm = await ollama.is_model_warm()
    return ModelStatusResponse(
        warm=warm,
        model=ollama._model,
        estimated_warmup_seconds=None if warm else _COLD_START_ESTIMATE_S,
    )
