from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import List, Optional
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel


from src.core.logger import logger
from src.schemas.query import QueryRequest
from src.api.routes.query import _build_stub_response

router = APIRouter()

# In-memory storage for consultations (replace with database later)
_consultations = []


class PatientMeta(BaseModel):
    age: int
    sex: str  # 'male' | 'female' | 'other'
    vitals: Optional[dict] = None


class Diagnosis(BaseModel):
    name: str
    probability: float
    evidenceRefs: List[str]
    accepted: Optional[bool] = None


class DiagnosisResult(BaseModel):
    diagnoses: List[Diagnosis]
    followUps: List[str]
    recommendedTests: List[dict]


class Consultation(BaseModel):
    id: str
    patient: PatientMeta
    symptoms: str
    results: DiagnosisResult
    notes: str = ""
    status: str = "draft"  # 'draft' | 'completed'
    createdAt: str


class ConsultationSummary(BaseModel):
    id: str
    patient: PatientMeta
    summary: str
    createdAt: str
    status: str


class CreateConsultationRequest(BaseModel):
    patient: PatientMeta
    symptoms: str
    meta: Optional[dict] = None


# Mock auth dependency - replace with real JWT validation
def mock_auth():
    # TODO: Implement real JWT token validation
    return {"user_id": "user-demo-001", "role": "doctor"}


@router.post(
    "",
    response_model=Consultation,
    summary="Create new consultation",
    description="Create a new consultation with patient data and symptoms",
)
async def create_consultation(
    request: CreateConsultationRequest,
    auth: dict = Depends(mock_auth),
) -> Consultation:
    # Map frontend patient meta to backend query format
    query_request = QueryRequest(
        symptoms=request.symptoms,
        patient_age=request.patient.age,
        gender=request.patient.sex,
    )

    # Get diagnosis results using existing logic
    diagnosis_response = _build_stub_response(request.symptoms, 0)

    # Convert backend response to frontend format
    diagnoses = []
    for diag in diagnosis_response.diagnoses:
        diagnoses.append(Diagnosis(
            name=diag.condition,
            probability=diag.probability / 100.0,  # Convert to 0-1 scale
            evidenceRefs=[ev.source for ev in diag.evidence],
        ))

    results = DiagnosisResult(
        diagnoses=diagnoses,
        followUps=diagnosis_response.follow_up_questions,
        recommendedTests=[],  # TODO: Add recommended tests logic
    )

    consultation = Consultation(
        id=f"consult-{uuid4().hex[:8]}",
        patient=request.patient,
        symptoms=request.symptoms,
        results=results,
        notes="",
        status="draft",
        createdAt=datetime.now(timezone.utc).isoformat(),
    )

    _consultations.insert(0, consultation.dict())  # Store as dict for JSON compatibility
    logger.info(f"Created consultation {consultation.id} for user {auth['user_id']}")

    return consultation


@router.get(
    "",
    response_model=List[ConsultationSummary],
    summary="List consultations",
    description="Get list of consultations for the authenticated user",
)
async def list_consultations(auth: dict = Depends(mock_auth)) -> List[ConsultationSummary]:
    summaries = []
    for cons in _consultations:
        summaries.append(ConsultationSummary(
            id=cons["id"],
            patient=cons["patient"],
            summary=cons["symptoms"][:80] + ("..." if len(cons["symptoms"]) > 80 else ""),
            createdAt=cons["createdAt"],
            status=cons["status"],
        ))

    logger.info(f"Listed {len(summaries)} consultations for user {auth['user_id']}")
    return summaries


@router.get(
    "/{consultation_id}",
    response_model=Consultation,
    summary="Get consultation",
    description="Get detailed consultation by ID",
)
async def get_consultation(
    consultation_id: str,
    auth: dict = Depends(mock_auth),
) -> Consultation:
    for cons in _consultations:
        if cons["id"] == consultation_id:
            logger.info(f"Retrieved consultation {consultation_id} for user {auth['user_id']}")
            return Consultation(**cons)

    raise HTTPException(status_code=404, detail="Consultation not found")