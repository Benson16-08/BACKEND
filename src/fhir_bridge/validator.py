"""
src/fhir_bridge/validator.py
─────────────────────────────
Validates incoming FHIR R4 payloads before they enter the pipeline.

Two validation layers:
  Layer 1 — Pydantic schema validation (FHIRPatientContext)
             Catches missing required fields, wrong types, malformed JSON.
             Returns FHIROperationOutcome on failure (HTTP 400).

  Layer 2 — Clinical completeness check
             Checks that the payload has enough clinical information to
             produce a meaningful diagnosis (at least one condition or
             observation with usable text).
             Returns FHIROperationOutcome on failure (HTTP 400).

Usage:
    from src.fhir_bridge.validator import validate_fhir_patient
    validate_fhir_patient(raw_dict)   # raises FHIRValidationError on fail
"""

from __future__ import annotations

from typing import Any, Dict, List

from pydantic import ValidationError

from src.core.exceptions import FHIRValidationError
from src.core.logger import logger
from src.schemas.errors import FHIRIssue, FHIROperationOutcome
from src.schemas.fhir import FHIRPatientContext


def _make_outcome(issues: List[FHIRIssue]) -> Dict[str, Any]:
    """Build a serialisable FHIROperationOutcome dict for the error response."""
    return FHIROperationOutcome(issue=issues).model_dump()


def validate_fhir_patient(payload: Dict[str, Any]) -> FHIRPatientContext:
    """
    Validate a raw FHIR payload dict and return a FHIRPatientContext.

    Args:
        payload: Raw dict from the request body (already JSON-decoded by FastAPI).

    Returns:
        A validated FHIRPatientContext ready to pass to parser.py.

    Raises:
        FHIRValidationError (HTTP 400): if any validation layer fails.
            The detail field contains a FHIR OperationOutcome body.
    """
    try:
        ctx = FHIRPatientContext(**payload)
    except ValidationError as exc:
        issues = []
        for err in exc.errors():
            field_path = ".".join(str(loc) for loc in err["loc"])
            issues.append(FHIRIssue(
                severity="error",
                code="required" if "missing" in err["type"] else "invalid",
                diagnostics=f"{field_path}: {err['msg']}",
                expression=[field_path] if field_path else None,
            ))
        logger.warning(f"FHIR schema validation failed: {len(issues)} issue(s)")
        raise FHIRValidationError(
            detail=f"FHIR payload failed schema validation: "
                   f"{issues[0].diagnostics if issues else 'unknown error'}",
            field=issues[0].expression[0] if issues and issues[0].expression else None,
        )
    except Exception as exc:
        raise FHIRValidationError(
            detail=f"Could not parse FHIR payload: {exc}"
        )

    issues = []

    if not ctx.patient.gender or ctx.patient.gender.strip() == "":
        issues.append(FHIRIssue(
            severity="error",
            code="required",
            diagnostics="Patient.gender is required for clinical context.",
            expression=["Patient.gender"],
        ))

    has_condition = any(
        c.get_condition_text() not in ("Unknown condition", "")
        for c in ctx.conditions
    )
    has_observation = any(
        c.get_observation_text() not in ("Observation", "")
        for c in ctx.observations
    )

    if not has_condition and not has_observation:
        issues.append(FHIRIssue(
            severity="error",
            code="required",
            diagnostics=(
                "Payload must include at least one Condition with code.text "
                "or one Observation with valueQuantity or valueString. "
                "Cannot produce a diagnosis without clinical information."
            ),
            expression=["conditions", "observations"],
        ))

    if issues:
        logger.warning(
            f"FHIR clinical completeness check failed: "
            f"{[i.diagnostics for i in issues]}"
        )
        raise FHIRValidationError(
            detail=issues[0].diagnostics,
            field=issues[0].expression[0] if issues[0].expression else None,
        )

    logger.info(
        f"FHIR validation passed: patient={ctx.patient.id}  "
        f"conditions={len(ctx.conditions)}  observations={len(ctx.observations)}"
    )
    return ctx
