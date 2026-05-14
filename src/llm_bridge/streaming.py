from __future__ import annotations

import json
import time
from typing import AsyncGenerator, List

from src.core.config import settings
from src.core.logger import logger
from src.llm_bridge.llm_client import run_pipeline
from src.llm_bridge.safety_checker import check_response
from src.schemas.query import QueryRequest


async def stream_pipeline(
    request: QueryRequest,
    chunks: List[dict],
    retrieval_ms: int = 0,
) -> AsyncGenerator[str, None]:
    """
    Async generator yielding SSE events for /api/v1/query/stream.

    Yields:
        SSE events: "data: <payload>\\n\\n"
        Final event: "data: [DONE]\\n\\n"
    """
    t_start = time.perf_counter()

    try:
        # Phase 1: immediate thinking signal
        thinking_event = json.dumps({
            "status": "thinking",
            "message": "Analysing symptoms against Tanzania STG guidelines...",
            "provider": settings.LLM_PROVIDER,
        })
        yield f"data: {thinking_event}\n\n"

        # Phase 2: run full pipeline
        response = await run_pipeline(request, chunks, retrieval_ms=retrieval_ms)
        response = check_response(response)

        total_ms = int((time.perf_counter() - t_start) * 1000)
        response.total_ms = total_ms

        # Phase 3: stream complete response
        payload = response.model_dump_json()
        yield f"data: {payload}\n\n"

        logger.info(f"stream_pipeline: complete in {total_ms}ms")

    except Exception as exc:
        error_event = json.dumps({
            "status": "error",
            "error": type(exc).__name__,
            "message": str(exc),
        })
        yield f"data: {error_event}\n\n"
        logger.error(f"stream_pipeline error: {exc}")

    finally:
        yield "data: [DONE]\n\n"
