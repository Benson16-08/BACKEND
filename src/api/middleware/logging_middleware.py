from __future__ import annotations

import time
from typing import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from src.core.logger import logger


class LoggingMiddleware(BaseHTTPMiddleware):
    """Intercepts requests and logs structured timing data."""

    _SKIP_PATHS = {"/docs", "/redoc", "/openapi.json", "/favicon.ico"}

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if request.url.path in self._SKIP_PATHS:
            return await call_next(request)

        t_start = time.perf_counter()
        client_ip = request.client.host if request.client else "unknown"

        response: Response = await call_next(request)

        total_ms = int((time.perf_counter() - t_start) * 1000)
        retrieval_ms = response.headers.get("X-Retrieval-Ms", "-")
        llm_ms = response.headers.get("X-LLM-Ms", "-")

        response.headers["X-Total-Ms"] = str(total_ms)

        logger.info(
            f"method={request.method} "
            f"path={request.url.path} "
            f"status={response.status_code} "
            f"retrieval_ms={retrieval_ms} "
            f"llm_ms={llm_ms} "
            f"total_ms={total_ms} "
            f"ip={client_ip}"
        )

        return response
