"""
src/schemas/errors.py
─────────────────────
Structured error response models.

Every HTTP error returns one of these shapes so Jesca's (Role 5)
frontend can parse all errors consistently — never a raw traceback.

StandardError       → all non-FHIR endpoints
FHIROperationOutcome → /api/v1/fhir/* endpoints (HL7 R4 compliant)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional
from pydantic import BaseModel, Field


class StandardError(BaseModel):
    """
    Consistent error body for all API errors.

    Shape:
    {
        "error":     "LLM_UNAVAILABLE",
        "message":   "The diagnostic model is temporarily unavailable.",
        "timestamp": "2024-06-01T10:00:00+00:00"
    }
    """
    error:     str = Field(..., description="Machine-readable error code.")
    message:   str = Field(..., description="Human-readable error message.")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC timestamp of the error.",
    )
    field: Optional[str] = Field(
        default=None,
        description="Field name that caused the error (validation errors only).",
    )


class FHIRIssue(BaseModel):
    """Single issue within a FHIR OperationOutcome resource."""
    severity:    str            = Field(..., description="'error' | 'warning' | 'information'")
    code:        str            = Field(..., description="FHIR issue type e.g. 'required'")
    diagnostics: Optional[str] = Field(default=None)
    expression:  Optional[List[str]] = Field(default=None)


class FHIROperationOutcome(BaseModel):
    """
    HL7 FHIR R4 OperationOutcome — returned by /api/v1/fhir/* on validation failure.

    Shape:
    {
        "resourceType": "OperationOutcome",
        "issue": [{"severity": "error", "code": "required",
                   "diagnostics": "Missing Patient.birthDate"}]
    }
    """
    resourceType: str            = Field(default="OperationOutcome")
    issue:        List[FHIRIssue] = Field(..., min_length=1)
