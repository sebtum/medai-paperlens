from typing import Any

from pydantic import BaseModel, Field, field_validator


class QueryRequest(BaseModel):
    question: str
    score_threshold: float = Field(default=0.4, ge=0.0, le=1.0)

    @field_validator("question")
    @classmethod
    def question_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question must not be empty")
        return v.strip()


class Citation(BaseModel):
    title: str
    authors: list[str]
    year: int
    doi: str | None = None
    excerpt: str


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    confidence: float
    grounded: bool
    debug: dict[str, Any]


# Payloads of the `POST /query/stream` SSE events (ADR-0017). The `final` event
# carries a QueryResponse; every other event type has a model below.


class StepStartEvent(BaseModel):
    node: str


class StepEndEvent(BaseModel):
    node: str
    elapsed_ms: int


class CitationsEvent(BaseModel):
    citations: list[Citation]
    confidence: float


class TokenEvent(BaseModel):
    text: str


class ErrorEvent(BaseModel):
    message: str
