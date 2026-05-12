
from __future__ import annotations

import time
from typing import AsyncGenerator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from src.core.exceptions import EmptySymptomsError
from src.core.logger import logger
from src.schemas.diagnosis import DiagnosisItem, DiagnosticResponse, EvidenceItem
from src.schemas.query import QueryRequest

router = APIRouter()


def _build_stub_response(symptoms: str, total_ms: int) -> DiagnosticResponse:
    return DiagnosticResponse(
        diagnoses=[
            DiagnosisItem(
                rank=1,
                condition="Malaria (Plasmodium falciparum)",
                probability=75,
                reasoning=(
                    f"Symptoms '{symptoms}' are consistent with uncomplicated "
                    "malaria. Fever, chills, and headache are classic presentations "
                    "in a malaria-endemic region. Confirm with RDT or microscopy."
                ),
                evidence=[
                    EvidenceItem(
                        source="Tanzania STG 2017 — Chapter 3: Malaria",
                        excerpt=(
                            "Uncomplicated malaria presents with fever, chills, "
                            "headache, and malaise. Diagnosis must be confirmed "
                            "parasitologically before treatment."
                        ),
                        chapter="Chapter 3",
                        relevance_score=0.92,
                    )
                ],
                icd10_code="B50.9",
            ),
            DiagnosisItem(
                rank=2,
                condition="Typhoid Fever",
                probability=15,
                reasoning=(
                    "Sustained fever with headache may indicate enteric fever. "
                    "Consider Widal test or blood culture if malaria RDT is negative."
                ),
                evidence=[
                    EvidenceItem(
                        source="Tanzania STG 2017 — Chapter 5: Bacterial Infections",
                        excerpt=(
                            "Typhoid fever presents with high-grade fever, headache, "
                            "abdominal discomfort, and relative bradycardia."
                        ),
                        chapter="Chapter 5",
                        relevance_score=0.71,
                    )
                ],
                icd10_code="A01.0",
            ),
            DiagnosisItem(
                rank=3,
                condition="Viral Upper Respiratory Tract Infection",
                probability=10,
                reasoning=(
                    "Fever with headache may also represent a non-specific viral "
                    "illness. Exclude malaria and typhoid before settling on this."
                ),
                evidence=[
                    EvidenceItem(
                        source="Tanzania STG 2017 — Chapter 8: Respiratory Infections",
                        excerpt=(
                            "Viral URTI presents with low-grade fever, headache, "
                            "and general malaise. Treatment is supportive."
                        ),
                        chapter="Chapter 8",
                        relevance_score=0.55,
                    )
                ],
                icd10_code="J06.9",
            ),
        ],
        follow_up_questions=[
            "How many days has the fever been present?",
            "Has the patient travelled recently or is in a malaria-endemic area?",
            "Is there associated nausea, vomiting, or diarrhoea?",
            "Has the patient taken any antimalarials or antibiotics recently?",
        ],
        total_ms=total_ms,
        warning="⚠️ STUB RESPONSE — Real retrieval and LLM pipeline not yet connected (Stage 3).",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/query",
    response_model=DiagnosticResponse,
    summary="Analyse symptoms and return differential diagnoses",
    description=(
        "Accepts a symptom description and optional patient metadata. "
        "Returns a ranked list of differential diagnoses grounded in "
        "Tanzania Standard Treatment Guidelines (STG). "
        "**Stage 1**: returns a stub response. "
        "**Stage 3**: fully wired RAG + LLM pipeline."
    ),
)
async def query_diagnose(request: QueryRequest) -> DiagnosticResponse:
    t_start = time.perf_counter()

    # Guard: catch empty/whitespace symptoms before hitting the pipeline
    if not request.symptoms.strip():
        raise EmptySymptomsError()

    logger.info(
        f"POST /query  symptoms_len={len(request.symptoms)}  "
        f"age={request.patient_age}  gender={request.gender}"
    )

    # ── STAGE 1: stub response ─────────────────────────────────────────────
    # TODO Stage 3: replace stub with real pipeline
    #   chunks   = await retrieve_chunks(request.symptoms)
    #   response = await run_pipeline(request, chunks)
    #   response = check_response(response)
    #   return response

    total_ms = int((time.perf_counter() - t_start) * 1000)
    response = _build_stub_response(request.symptoms, total_ms)

    logger.info(f"POST /query  completed  total_ms={total_ms}")
    return response


@router.post(
    "/query/stream",
    summary="Streaming SSE diagnosis (token-by-token)",
    description=(
        "Returns a Server-Sent Events (SSE) stream of diagnosis tokens. "
        "Jesca's (Role 5) frontend uses this for progressive rendering. "
        "**Stage 1**: streams a single stub event. "
        "**Stage 3**: streams real LLM tokens."
    ),
)
async def query_stream(request: QueryRequest) -> StreamingResponse:
    if not request.symptoms.strip():
        raise EmptySymptomsError()

    async def _stub_stream() -> AsyncGenerator[str, None]:
        """
        Stub SSE stream — yields one data event then [DONE].
        Stage 3 replaces this with the real Ollama/OpenAI token stream.
        """
        import json
        stub = _build_stub_response(request.symptoms, total_ms=0)
        payload = stub.model_dump_json()
        yield f"data: {payload}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        _stub_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )