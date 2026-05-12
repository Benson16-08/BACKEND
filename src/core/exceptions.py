
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import HTTPException


def _error_detail(
    code: str,
    message: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "error": code,
        "message": message,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        body.update(extra)
    return body


class LLMUnavailableError(HTTPException):
    def __init__(self, detail: str = "The diagnostic model is temporarily unavailable. Please retry in 30 seconds.") -> None:
        super().__init__(
            status_code=503,
            detail=_error_detail(
                code="LLM_UNAVAILABLE",
                message=detail,
            ),
        )


class LLMTimeoutError(HTTPException):
    def __init__(self, detail: str = "The diagnostic model took too long to respond. Please try again.") -> None:
        super().__init__(
            status_code=504,
            detail=_error_detail(
                code="LLM_TIMEOUT",
                message=detail,
            ),
        )


class LLMResponseInvalidError(HTTPException):
    def __init__(self, detail: str = "The diagnostic model returned an invalid response. Please retry.") -> None:
        super().__init__(
            status_code=503,
            detail=_error_detail(
                code="LLM_RESPONSE_INVALID",
                message=detail,
            ),
        )


class RetrievalFailedError(HTTPException):
    def __init__(self, detail: str = "The clinical knowledge base is temporarily unavailable.") -> None:
        super().__init__(
            status_code=503,
            detail=_error_detail(
                code="RETRIEVAL_FAILED",
                message=detail,
            ),
        )


class EmbeddingFailedError(HTTPException):
    def __init__(self, detail: str = "Failed to encode the symptom query. Please check Ollama is running.") -> None:
        super().__init__(
            status_code=503,
            detail=_error_detail(
                code="EMBEDDING_FAILED",
                message=detail,
            ),
        )


class EmptySymptomsError(HTTPException):
    def __init__(self) -> None:
        super().__init__(
            status_code=422,
            detail=_error_detail(
                code="EMPTY_SYMPTOMS",
                message="Symptoms cannot be empty. Please enter the patient's symptoms.",
            ),
        )


class SymptomsTooShortError(HTTPException):
    def __init__(self, min_chars: int = 5) -> None:
        super().__init__(
            status_code=422,
            detail=_error_detail(
                code="SYMPTOMS_TOO_SHORT",
                message=f"Symptoms must be at least {min_chars} characters. Please provide more detail.",
                extra={"min_chars": min_chars},
            ),
        )


class SafetyCheckFailedError(HTTPException):
    def __init__(self, detail: str = "Response safety validation failed unexpectedly.") -> None:
        super().__init__(
            status_code=503,
            detail=_error_detail(
                code="SAFETY_CHECK_FAILED",
                message=detail,
            ),
        )


class FHIRValidationError(HTTPException):
    def __init__(self, detail: str = "Invalid FHIR resource. Check required fields.", field: Optional[str] = None) -> None:
        super().__init__(
            status_code=400,
            detail=_error_detail(
                code="FHIR_VALIDATION_ERROR",
                message=detail,
                extra={"field": field} if field else None,
            ),
        )


class InternalServerError(HTTPException):
    def __init__(self, detail: str = "An unexpected error occurred. Please contact support.") -> None:
        super().__init__(
            status_code=500,
            detail=_error_detail(
                code="INTERNAL_SERVER_ERROR",
                message=detail,
            ),
        )