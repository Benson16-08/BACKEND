
from __future__ import annotations

import time

from fastapi import APIRouter
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
async def query_diagnose(request: QueryRequest) -> DiagnosticResponse:
    t_start = time.perf_counter()

    if not request.symptoms.strip():
        raise EmptySymptomsError()

    logger.info(
        f"POST /query  symptoms_len={len(request.symptoms)}  "
        f"age={request.patient_age}  gender={request.gender}"
    )

    t_retrieval = time.perf_counter()
    chunks = await retrieve_chunks(request.symptoms)
    retrieval_ms = int((time.perf_counter() - t_retrieval) * 1000)

    response = await run_pipeline(request, chunks, retrieval_ms=retrieval_ms)
    response = check_response(response)
    response.total_ms = int((time.perf_counter() - t_start) * 1000)

    logger.info(
        f"POST /query  done  "
        f"retrieval_ms={retrieval_ms}  "
        f"llm_ms={response.llm_ms}  "
        f"total_ms={response.total_ms}  "
        f"diagnoses={len(response.diagnoses)}  "
        f"confidence={response.confidence_overall}"
    )
    return response


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