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
