
from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional
from pydantic import BaseModel, Field


class StandardError(BaseModel):
    error: str = Field(..., description="Machine-readable error code.")
    message: str = Field(..., description="Human-readable error message.")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC timestamp of when the error occurred.",
    )
    field: Optional[str] = Field(
        default=None,
        description="Field name that caused the error (validation errors only).",
    )



class FHIRIssue(BaseModel):
    severity: str = Field(
        ...,
        description="'error' | 'warning' | 'information' | 'fatal'",
    )
    code: str = Field(
        ...,
        description="FHIR issue type code e.g. 'required', 'invalid'.",
    )
    diagnostics: Optional[str] = Field(
        default=None,
        description="Human-readable description of the issue.",
    )
    expression: Optional[List[str]] = Field(
        default=None,
        description="FHIRPath expression(s) pointing to the offending field.",
    )


class FHIROperationOutcome(BaseModel):
    resourceType: str = Field(
        default="OperationOutcome",
        description="Always 'OperationOutcome' — required by FHIR R4 spec.",
    )
    issue: List[FHIRIssue] = Field(
        ...,
        min_length=1,
        description="List of issues found during validation.",
    )