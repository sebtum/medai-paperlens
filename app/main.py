from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routes.health import router as health_router
from app.api.routes.query import router as query_router
from app.core.exceptions import UnsafeQueryError

app = FastAPI(title="MedAI PaperLens", version="0.1.0")

app.include_router(health_router)
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
