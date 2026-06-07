import asyncio
import logging
from contextlib import asynccontextmanager

import dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routes.health import router as health_router
from app.api.routes.model import router as model_router
from app.api.routes.query import router as query_router
from app.core.exceptions import UnsafeQueryError
from app.llm.ollama import make_ollama_client

dotenv.load_dotenv()
logging.getLogger("app").setLevel(logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with make_ollama_client() as ollama:
        app.state.ollama = ollama
        app.state.warmup_task = asyncio.create_task(ollama.warmup())
        yield


app = FastAPI(title="MedAI PaperLens", version="0.1.0", lifespan=lifespan)

app.include_router(health_router)
app.include_router(model_router, prefix="/model")
app.include_router(query_router)


@app.exception_handler(UnsafeQueryError)
async def unsafe_query_handler(request: Request, exc: UnsafeQueryError) -> JSONResponse:
    return JSONResponse(
        status_code=200,
        content={
            "answer": "Literature summaries only — no medical advice.",
            "citations": [],
            "confidence": 0.0,
            "grounded": False,
            "debug": {"route": "unsafe_medical_advice"},
        },
    )
