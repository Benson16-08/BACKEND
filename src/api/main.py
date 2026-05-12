
from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.core.config import settings
from src.core.exceptions import InternalServerError
from src.core.logger import logger

from src.api.routes import auth, consultations, diagnosis, health, query



@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("=" * 60)
    logger.info("MediAssist API starting up")
    logger.info(f"  LLM provider    : {settings.LLM_PROVIDER}")
    logger.info(f"  ChromaDB path   : {settings.CHROMA_DB_PATH}")
    logger.info(f"  Collection name : {settings.CHROMA_COLLECTION_NAME}")
    logger.info(f"  Allowed origins : {settings.allowed_origins_list}")
    logger.info(f"  Host / Port     : {settings.HOST}:{settings.PORT}")
    logger.info("MediAssist API ready ✓")
    logger.info("=" * 60)

    yield  # ← application runs here

    logger.info("MediAssist API shutting down")



def create_app() -> FastAPI:
    app = FastAPI(
        title="MediAssist API",
        description=(
            "Clinical decision support API for MediAssist. "
            "Provides AI-assisted differential diagnosis using RAG over "
            "Tanzania Standard Treatment Guidelines (STG)."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins_list,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization", "X-Request-ID"],
        expose_headers=["X-Retrieval-Ms", "X-LLM-Ms", "X-Total-Ms"],
    )

    app.include_router(auth.router, prefix="/api/v1/auth", tags=["Auth"])
    app.include_router(consultations.router, prefix="/api/v1/consultations", tags=["Consultations"])
    app.include_router(diagnosis.router, prefix="/api/v1/diagnosis", tags=["Diagnosis"])
    app.include_router(health.router, prefix="/api/v1", tags=["Health"])
    app.include_router(query.router,  prefix="/api/v1", tags=["Diagnosis"])

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        logger.error(
            f"Unhandled exception on {request.method} {request.url.path}: "
            f"{type(exc).__name__}: {exc}"
        )
        err = InternalServerError()
        return JSONResponse(status_code=err.status_code, content=err.detail)

    return app


app: FastAPI = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
        log_level="info",
    )