"""FastAPI application factory for MediAssist backend."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.exceptions import HTTPException, RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.core.config import settings
from src.core.exceptions import InternalServerError, FHIRValidationError
from src.api.middleware.logging_middleware import LoggingMiddleware
from src.core.logger import logger
from src.api.routes import health, query, fhir


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup: load KB and build BM25. Shutdown: log closure."""
    logger.info("=" * 60)
    logger.info("MediAssist API starting up")

    from src.knowledge_base.kb_loader import kb
    kb.load()

    import src.retrieval.bm25_index  # noqa: F401
    logger.info(f"  LLM provider    : {settings.LLM_PROVIDER}")
    logger.info(f"  ChromaDB path   : {settings.CHROMA_DB_PATH}")
    logger.info(f"  Collection name : {settings.CHROMA_COLLECTION_NAME}")
    logger.info(f"  Allowed origins : {settings.allowed_origins_list}")
    logger.info(f"  Host / Port     : {settings.HOST}:{settings.PORT}")
    logger.info("MediAssist API ready ✓")
    logger.info("=" * 60)

    yield

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

    app.add_middleware(LoggingMiddleware)

    app.include_router(health.router, prefix="/api/v1", tags=["Health"])
    app.include_router(query.router,  prefix="/api/v1", tags=["Diagnosis"])
    app.include_router(fhir.router,   prefix="/api/v1", tags=["FHIR Bridge"])

    @app.exception_handler(HTTPException)
    async def http_exception_handler(
        request: Request, exc: HTTPException
    ) -> JSONResponse:
        """Return structured JSON for HTTPExceptions."""
        if isinstance(exc.detail, dict):
            return JSONResponse(status_code=exc.status_code, content=exc.detail)

        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error":     f"HTTP_{exc.status_code}",
                "message":   str(exc.detail),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """Return structured 422 with field-level detail for Pydantic errors."""
        errors = exc.errors()
        field = ".".join(str(e) for e in errors[0]["loc"]) if errors else "unknown"
        message = errors[0]["msg"] if errors else "Validation failed."
        return JSONResponse(
            status_code=422,
            content={
                "error":     "VALIDATION_ERROR",
                "message":   message,
                "field":     field,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "details":   errors,
            },
        )

    @app.exception_handler(FHIRValidationError)
    async def fhir_validation_handler(
        request: Request, exc: FHIRValidationError
    ) -> JSONResponse:
        """Return FHIR OperationOutcome for FHIR validation errors."""
        from src.schemas.errors import FHIROperationOutcome, FHIRIssue
        outcome = FHIROperationOutcome(issue=[
            FHIRIssue(
                severity="error",
                code=exc.detail.get("error", "invalid") if isinstance(exc.detail, dict) else "invalid",
                diagnostics=exc.detail.get("message", str(exc.detail)) if isinstance(exc.detail, dict) else str(exc.detail),
                expression=[exc.detail.get("field")] if isinstance(exc.detail, dict) and exc.detail.get("field") else None,
            )
        ])
        return JSONResponse(
            status_code=400,
            content=outcome.model_dump(),
            media_type="application/fhir+json",
        )

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
