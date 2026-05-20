from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.schemas.diagnosis import DiagnosisItem, DiagnosticResponse

SAMPLE_SYMPTOMS = "fever with chills and severe headache for three days"

SAMPLE_CHUNKS = [
    {
        "text": "Malaria presents with fever, chills, and headache in endemic areas. "
                "Diagnosis confirmed by mRDT or microscopy per Tanzania STG.",
        "chapter": "Chapter Five: MALARIA",
        "section": "Uncomplicated Malaria",
        "chunk_id": "chunk-001",
        "source": "Chapter Five: MALARIA — Uncomplicated Malaria",
        "rrf_score": 0.032,
        "retrieval_method": "hybrid",
    },
    {
        "text": "Severe malaria in pregnancy requires urgent treatment with IV artesunate. "
                "Refer immediately per Tanzania STG guidelines.",
        "chapter": "Chapter Five: MALARIA",
        "section": "Severe Malaria In Pregnancy",
        "chunk_id": "chunk-002",
        "source": "Chapter Five: MALARIA — Severe Malaria In Pregnancy",
        "rrf_score": 0.028,
        "retrieval_method": "bm25",
    },
]

SAMPLE_DIAGNOSTIC_RESPONSE = DiagnosticResponse(
    diagnoses=[
        DiagnosisItem(
            rank=1,
            condition="Malaria (Plasmodium falciparum)",
            probability=75,
            reasoning=(
                "Fever with chills and headache in a malaria-endemic region "
                "is highly consistent with uncomplicated malaria per Tanzania STG."
            ),
            evidence=(
                "Malaria presents with fever, chills, and headache in endemic areas. "
                "Diagnosis confirmed by mRDT or microscopy per Tanzania STG."
            ),
            source_section="Chapter Five: MALARIA — Uncomplicated Malaria",
        ),
        DiagnosisItem(
            rank=2,
            condition="Typhoid Fever",
            probability=15,
            reasoning=(
                "Prolonged fever with headache may indicate enteric fever. "
                "Consider Widal test if malaria RDT is negative per STG."
            ),
            evidence=(
                "Typhoid presents with prolonged fever, headache, and abdominal "
                "discomfort. Blood culture is the gold standard per Tanzania STG."
            ),
            source_section="Chapter Five: BACTERIAL INFECTIONS — Typhoid",
        ),
        DiagnosisItem(
            rank=3,
            condition="Viral Upper Respiratory Infection",
            probability=10,
            reasoning=(
                "Non-specific viral illness may also present with fever and headache. "
                "Exclude malaria and typhoid first per STG differential."
            ),
            evidence=(
                "Viral URTI presents with low-grade fever, headache, and malaise. "
                "Treatment is supportive per Tanzania STG guidelines."
            ),
            source_section="Chapter Eight: RESPIRATORY — Viral URTI",
        ),
    ],
    follow_up_questions=[
        "How many days has the fever been present?",
        "Any recent travel or mosquito exposure history?",
        "Has the patient taken any antimalarials recently?",
    ],
    recommended_tests=[
        "Malaria RDT (mRDT)",
        "Full blood count",
        "Widal test if mRDT negative",
    ],
    confidence_overall="high",
    pipeline_confidence="high",
    retrieval_ms=45,
    llm_ms=1200,
    total_ms=1250,
)

SAMPLE_FHIR_PAYLOAD = {
    "patient": {
        "resourceType": "Patient",
        "id": "test-patient-001",
        "gender": "male",
        "birthDate": "1990-01-15",
        "name": [{"family": "Mwangi", "given": ["James"]}],
    },
    "conditions": [
        {
            "resourceType": "Condition",
            "code": {"text": "Fever with chills for 3 days"},
            "clinicalStatus": {"coding": [{"code": "active"}]},
        }
    ],
    "observations": [
        {
            "resourceType": "Observation",
            "status": "final",
            "code": {"text": "Body temperature"},
            "valueQuantity": {"value": 39.2, "unit": "C"},
        }
    ],
}


@pytest.fixture(scope="session")
def kb_loaded():
    from src.knowledge_base.kb_loader import kb

    if not kb._loaded:
        kb.load()
    return kb


@pytest.fixture(scope="session")
def client(kb_loaded):
    from src.api.main import app

    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


@pytest.fixture
def mock_llm(mocker):
    return mocker.patch(
        "src.api.routes.query.run_pipeline",
        return_value=SAMPLE_DIAGNOSTIC_RESPONSE,
    )


@pytest.fixture
def mock_fhir_llm(mocker):
    return mocker.patch(
        "src.api.routes.fhir.run_pipeline",
        return_value=SAMPLE_DIAGNOSTIC_RESPONSE,
    )


@pytest.fixture
def mock_retrieval(mocker):
    async def _fake_retrieve(*args, **kwargs):
        return SAMPLE_CHUNKS

    mocker.patch("src.api.routes.query.retrieve_chunks", side_effect=_fake_retrieve)
    mocker.patch("src.api.routes.fhir.retrieve_chunks", side_effect=_fake_retrieve)
    return SAMPLE_CHUNKS


@pytest.fixture
def sample_response():
    return SAMPLE_DIAGNOSTIC_RESPONSE


@pytest.fixture
def sample_fhir_payload():
    return SAMPLE_FHIR_PAYLOAD
