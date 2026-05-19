from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Dict

from fastapi import APIRouter, Response
from fastapi.responses import JSONResponse

from src.core.exceptions import FHIRValidationError
from src.core.logger import logger
from src.fhir_bridge.parser import parse_patient_context
from src.fhir_bridge.validator import validate_fhir_patient
from src.llm_bridge.llm_client import run_pipeline
from src.llm_bridge.safety_checker import check_response
from src.retrieval.retrieval_service import retrieve_chunks
from src.schemas.diagnosis import DiagnosticResponse

router = APIRouter()


@router.post(
    "/fhir/patient-context",
    response_model=DiagnosticResponse,
    responses={
        400: {
            "description": "Invalid FHIR payload — returns HL7 OperationOutcome",
            "content": {
                "application/fhir+json": {
                    "example": {
                        "resourceType": "OperationOutcome",
                        "issue": [{
                            "severity": "error",
                            "code": "required",
                            "diagnostics": "Missing required field: Patient.gender"
                        }]
                    }
                }
            }
        }
    },
    summary="FHIR patient context → differential diagnosis",
    description=(
        "Accepts a MediAssist FHIR R4 patient context payload from GoTHoMIS or OpenMRS. "
        "Validates the FHIR resources, extracts symptoms and patient metadata, "
        "runs the full RAG + LLM pipeline, and returns a DiagnosticResponse. "
        "Returns an HL7 OperationOutcome (HTTP 400) if the FHIR payload is invalid."
    ),
)
async def fhir_patient_context(
    payload: Dict[str, Any],
    response: Response,
) -> DiagnosticResponse:
    """Process FHIR patient context and return diagnostic response."""
    t_start = time.perf_counter()

    logger.info(
        f"POST /fhir/patient-context  "
        f"patient_id={payload.get('patient', {}).get('id', 'unknown')}"
    )

    ctx = validate_fhir_patient(payload)
    query_request = parse_patient_context(ctx)

    t_retrieval = time.perf_counter()
    chunks = await retrieve_chunks(query_request.symptoms)
    retrieval_ms = int((time.perf_counter() - t_retrieval) * 1000)

    diagnostic = await run_pipeline(query_request, chunks, retrieval_ms=retrieval_ms)
    diagnostic = check_response(diagnostic)

    total_ms = int((time.perf_counter() - t_start) * 1000)
    diagnostic.total_ms = total_ms

    response.headers["X-Retrieval-Ms"] = str(retrieval_ms)
    response.headers["X-LLM-Ms"]       = str(diagnostic.llm_ms or 0)
    response.headers["X-Total-Ms"]     = str(total_ms)

    logger.info(
        f"POST /fhir/patient-context done  "
        f"retrieval_ms={retrieval_ms}  "
        f"llm_ms={diagnostic.llm_ms}  "
        f"total_ms={total_ms}  "
        f"diagnoses={len(diagnostic.diagnoses)}"
    )
    return diagnostic


@router.get(
    "/fhir/health",
    summary="FHIR bridge liveness — returns CapabilityStatement stub",
    description=(
        "Returns a minimal HL7 FHIR R4 CapabilityStatement confirming "
        "the MediAssist FHIR bridge is reachable. "
        "GoTHoMIS calls this before sending patient data."
    ),
)
async def fhir_health() -> JSONResponse:
    """
    Minimal FHIR CapabilityStatement for liveness checking.
    Not a full conformance statement — just enough for GoTHoMIS to
    confirm the bridge endpoint is reachable and responding.
    """
    capability = {
        "resourceType": "CapabilityStatement",
        "status": "active",
        "date": datetime.now(timezone.utc).isoformat(),
        "kind": "instance",
        "fhirVersion": "4.0.1",
        "format": ["application/fhir+json", "application/json"],
        "rest": [{
            "mode": "server",
            "resource": [{
                "type": "Patient",
                "interaction": [{"code": "read"}],
            }],
        }],
        "implementation": {
            "description": "MediAssist FHIR R4 bridge — GoTHoMIS integration",
            "url": "http://localhost:8000/api/v1/fhir",
        },
    }
    return JSONResponse(
        content=capability,
        media_type="application/fhir+json",
    )
