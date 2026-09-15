import os
from typing import cast

from fastapi import Request

from app.llm.base import LlmProvider
from app.llm.gemini import GeminiClient
from app.llm.ollama import OllamaClient, make_ollama_client


def make_gemini_client() -> GeminiClient:
    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    return GeminiClient(model)


def make_llm_provider() -> OllamaClient | GeminiClient:
    provider = os.environ.get("LLM_PROVIDER", "ollama").strip().lower()
    if provider == "ollama":
        return make_ollama_client()
    if provider == "gemini":
        return make_gemini_client()
    raise ValueError(
        f"Unknown LLM_PROVIDER {provider!r}, expected 'ollama' or 'gemini'"
    )


def get_llm_provider(request: Request) -> LlmProvider:
    return cast(LlmProvider, request.app.state.llm)
