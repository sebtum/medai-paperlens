from typing import Any

from pydantic import BaseModel, Field, field_validator

# Provisional, set in Phase 8 T12 for bge-small: off-topic top-1 scores were
# 0.44-0.53, on-topic top-5 scores 0.74-0.92. Phase 9 tunes it.
DEFAULT_SCORE_THRESHOLD = 0.6


class QueryRequest(BaseModel):
    question: str
    score_threshold: float = Field(default=DEFAULT_SCORE_THRESHOLD, ge=0.0, le=1.0)

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
    # Chunk provenance (ADR-0021); optional, as abstract-only payloads lack them.
    paper_id: str | None = None
    section: str | None = None
    page: int | None = None
    chunk_index: int | None = None


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
