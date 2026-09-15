from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.llm.base import ModelStatusProvider
from app.llm.factory import get_llm_provider

router = APIRouter()

_COLD_START_ESTIMATE_S: int = 120


class ModelStatusResponse(BaseModel):
    warm: bool
    model: str
    estimated_warmup_seconds: int | None


@router.get("/status", response_model=ModelStatusResponse)
async def model_status(
    llm: Annotated[ModelStatusProvider, Depends(get_llm_provider)],
) -> ModelStatusResponse:
    warm = await llm.is_model_warm()
    return ModelStatusResponse(
        warm=warm,
        model=llm.model,
        estimated_warmup_seconds=None if warm else _COLD_START_ESTIMATE_S,
    )
