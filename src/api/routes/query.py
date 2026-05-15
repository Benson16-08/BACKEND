"""
src/api/routes/query.py
────────────────────────
POST /api/v1/query        — full pipeline (sync JSON response)
POST /api/v1/query/stream — streaming SSE response

STAGE 3 (current):
  Full pipeline wired:
    retrieve_chunks() → run_pipeline() → check_response() → DiagnosticResponse

Pipeline:
  1. Validate QueryRequest (Pydantic)
  2. retrieve_chunks()    — Stage 2: hybrid BM25 + vector search
  3. run_pipeline()       — Stage 3: calls Peter's LLM pipeline (or direct LLM)
  4. check_response()     — Stage 3: safety rules enforcement
  5. Return DiagnosticResponse with timing fields attached
"""

from __future__ import annotations

import time
from typing import AsyncGenerator

from fastapi import APIRouter, Response
from fastapi.responses import StreamingResponse

from src.core.exceptions import EmptySymptomsError
from src.core.logger import logger
from src.llm_bridge.llm_client import run_pipeline
from src.llm_bridge.safety_checker import check_response
from src.llm_bridge.streaming import stream_pipeline
from src.retrieval.retrieval_service import retrieve_chunks
from src.schemas.diagnosis import DiagnosticResponse
from src.schemas.query import QueryRequest

router = APIRouter()


@router.post(
    "/query",
    response_model=DiagnosticResponse,
    summary="Analyse symptoms — returns full DiagnosticResponse",
    description=(
        "Accepts patient symptoms and optional metadata. "
        "Runs the full RAG + LLM pipeline against Tanzania STG guidelines. "
        "Returns a ranked differential diagnosis grounded in clinical evidence."
    ),
)
async def query_diagnose(request: QueryRequest, response: Response) -> DiagnosticResponse:
    t_start = time.perf_counter()

    if not request.symptoms.strip():
        raise EmptySymptomsError()

    logger.info(
        f"POST /query  symptoms_len={len(request.symptoms)}  "
        f"age={request.patient_age}  gender={request.gender}"
    )

    # ── Stage 2: Retrieval ────────────────────────────────────────────────────
    t_retrieval = time.perf_counter()
    chunks = await retrieve_chunks(request.symptoms)
    retrieval_ms = int((time.perf_counter() - t_retrieval) * 1000)

    # ── Stage 3: LLM pipeline ─────────────────────────────────────────────────
    response_obj = await run_pipeline(request, chunks, retrieval_ms=retrieval_ms)

    # ── Stage 3: Safety check ─────────────────────────────────────────────────
    response_obj = check_response(response_obj)

    # ── Attach final timing ───────────────────────────────────────────────────
    response_obj.total_ms = int((time.perf_counter() - t_start) * 1000)

    response.headers["X-Retrieval-Ms"] = str(retrieval_ms)
    response.headers["X-LLM-Ms"] = str(response_obj.llm_ms or 0)
    response.headers["X-Total-Ms"] = str(response_obj.total_ms or 0)

    logger.info(
        f"POST /query  done  "
        f"retrieval_ms={retrieval_ms}  "
        f"llm_ms={response_obj.llm_ms}  "
        f"total_ms={response_obj.total_ms}  "
        f"diagnoses={len(response_obj.diagnoses)}  "
        f"confidence={response_obj.confidence_overall}"
    )
    return response_obj


@router.post(
    "/query/stream",
    summary="Analyse symptoms — streaming SSE response",
    description=(
        "Returns a Server-Sent Events stream. "
        "First event: thinking status. "
        "Second event: complete DiagnosticResponse JSON. "
        "Final event: [DONE]. "
        "Jesca's (Role 5) frontend uses this for progressive rendering."
    ),
)
async def query_stream(request: QueryRequest) -> StreamingResponse:
    if not request.symptoms.strip():
        raise EmptySymptomsError()

    t_retrieval = time.perf_counter()
    chunks = await retrieve_chunks(request.symptoms)
    retrieval_ms = int((time.perf_counter() - t_retrieval) * 1000)

    return StreamingResponse(
        stream_pipeline(request, chunks, retrieval_ms=retrieval_ms),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
