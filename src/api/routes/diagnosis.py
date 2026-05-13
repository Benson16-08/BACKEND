from __future__ import annotations

import time
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.api.routes.query import _build_stub_response
from src.core.logger import logger
from src.schemas.query import QueryRequest

router = APIRouter()


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


class RetrieveRequest(BaseModel):
    symptoms: str
    patientMeta: PatientMeta


class RetrieveResponse(BaseModel):
    retrievedDocs: List[dict]
    embeddingsMeta: dict


class GenerateRequest(BaseModel):
    symptoms: str
    patientMeta: PatientMeta


@router.post(
    "/retrieve",
    response_model=RetrieveResponse,
    summary="Retrieve relevant documents",
    description="Retrieve relevant STG documents for symptoms and patient metadata",
)
async def retrieve_documents(request: RetrieveRequest) -> RetrieveResponse:
    # TODO: Implement real retrieval logic
    # For now, return mock data similar to frontend mocks

    logger.info(f"Retrieving documents for symptoms: {request.symptoms[:50]}...")

    # Mock retrieved documents
    retrieved_docs = [
        {
            "id": "stg-malaria-sec3",
            "source": "Tanzania STG 2017 — Chapter 3: Malaria",
            "section": "Chapter 3",
            "title": "Malaria Diagnosis and Treatment",
            "excerpt": "Uncomplicated malaria presents with fever, chills, headache, and malaise. Diagnosis must be confirmed parasitologically before treatment."
        },
        {
            "id": "stg-typhoid-sec2",
            "source": "Tanzania STG 2017 — Chapter 5: Bacterial Infections",
            "section": "Chapter 5",
            "title": "Typhoid Fever",
            "excerpt": "Typhoid fever presents with high-grade fever, headache, abdominal discomfort, and relative bradycardia."
        },
        {
            "id": "stg-urti-sec1",
            "source": "Tanzania STG 2017 — Chapter 8: Respiratory Infections",
            "section": "Chapter 8",
            "title": "Upper Respiratory Tract Infections",
            "excerpt": "Viral URTI presents with low-grade fever, headache, and general malaise. Treatment is supportive."
        }
    ]

    return RetrieveResponse(
        retrievedDocs=retrieved_docs,
        embeddingsMeta={
            "model": "gte-small",
            "latency": 120
        }
    )


@router.post(
    "/generate",
    response_model=DiagnosisResult,
    summary="Generate diagnosis",
    description="Generate differential diagnosis based on symptoms and patient metadata",
)
async def generate_diagnosis(request: GenerateRequest) -> DiagnosisResult:
    # Convert frontend request to backend format
    query_request = QueryRequest(
        symptoms=request.symptoms,
        patient_age=request.patientMeta.age,
        gender=request.patientMeta.sex,
    )

    # Use existing diagnosis logic
    from src.api.routes.query import _build_stub_response
    diagnosis_response = _build_stub_response(request.symptoms, 0)

    # Convert backend response to frontend format
    diagnoses = []
    for diag in diagnosis_response.diagnoses:
        diagnoses.append(Diagnosis(
            name=diag.condition,
            probability=diag.probability / 100.0,  # Convert to 0-1 scale
            evidenceRefs=[ev.source for ev in diag.evidence],
        ))

    logger.info(f"Generated diagnosis for symptoms: {request.symptoms[:50]}...")

    return DiagnosisResult(
        diagnoses=diagnoses,
        followUps=diagnosis_response.follow_up_questions,
        recommendedTests=[
            {"test": "mRDT", "rationale": "STG recommends mRDT for suspected malaria"},
            {"test": "Blood culture", "rationale": "To rule out bacterial infections like typhoid"}
        ]  # Add some basic recommended tests
    )