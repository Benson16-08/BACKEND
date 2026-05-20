import pytest
from pydantic import ValidationError

from src.core.exceptions import FHIRValidationError
from src.fhir_bridge.validator import validate_fhir_patient
from src.fhir_bridge.parser import parse_patient_context
from src.schemas.fhir import FHIRPatient, FHIRPatientContext


class TestValidator:

    def test_valid_payload_returns_context(self, sample_fhir_payload):
        ctx = validate_fhir_patient(sample_fhir_payload)
        assert ctx.patient.id == "test-patient-001"
        assert len(ctx.conditions) == 1
        assert len(ctx.observations) == 1

    def test_missing_patient_id_raises(self, sample_fhir_payload):
        payload = {**sample_fhir_payload,
                   "patient": {"gender": "male", "birthDate": "1990-01-01"}}
        with pytest.raises(FHIRValidationError) as exc_info:
            validate_fhir_patient(payload)
        assert exc_info.value.status_code == 400

    def test_missing_patient_gender_raises(self, sample_fhir_payload):
        payload = {**sample_fhir_payload,
                   "patient": {"id": "p1", "birthDate": "1990-01-01"}}
        with pytest.raises(FHIRValidationError) as exc_info:
            validate_fhir_patient(payload)
        assert exc_info.value.status_code == 400

    def test_empty_conditions_and_observations_raises(self):
        with pytest.raises(FHIRValidationError) as exc_info:
            validate_fhir_patient({
                "patient": {"id": "p1", "gender": "female"},
                "conditions": [],
                "observations": [],
            })
        assert exc_info.value.status_code == 400
        assert "clinical" in exc_info.value.detail["message"].lower()

    def test_wrong_structure_raises(self):
        with pytest.raises(FHIRValidationError):
            validate_fhir_patient({"not_patient": "bad_data"})

    def test_extra_fhir_fields_allowed(self, sample_fhir_payload):
        payload = {
            **sample_fhir_payload,
            "patient": {
                **sample_fhir_payload["patient"],
                "meta": {"versionId": "1"},
                "communication": [{"language": {"text": "Swahili"}}],
            }
        }
        ctx = validate_fhir_patient(payload)
        assert ctx.patient.id == "test-patient-001"


class TestParser:

    def test_symptoms_extracted_from_conditions(self, sample_fhir_payload):
        ctx = validate_fhir_patient(sample_fhir_payload)
        qr = parse_patient_context(ctx)
        assert "Fever with chills" in qr.symptoms

    def test_vitals_appended_to_symptoms(self, sample_fhir_payload):
        ctx = validate_fhir_patient(sample_fhir_payload)
        qr = parse_patient_context(ctx)
        assert "39.2" in qr.symptoms or "39.2" in (qr.temperature or "")

    def test_patient_age_calculated(self, sample_fhir_payload):
        ctx = validate_fhir_patient(sample_fhir_payload)
        qr = parse_patient_context(ctx)
        assert qr.patient_age is not None
        assert 20 < qr.patient_age < 50

    def test_gender_extracted(self, sample_fhir_payload):
        ctx = validate_fhir_patient(sample_fhir_payload)
        qr = parse_patient_context(ctx)
        assert qr.gender == "male"

    def test_temperature_extracted(self, sample_fhir_payload):
        ctx = validate_fhir_patient(sample_fhir_payload)
        qr = parse_patient_context(ctx)
        assert qr.temperature is not None
        assert "39.2" in qr.temperature

    def test_patient_name_extracted(self, sample_fhir_payload):
        ctx = validate_fhir_patient(sample_fhir_payload)
        qr = parse_patient_context(ctx)
        assert qr.patient_name == "James Mwangi"

    def test_symptoms_not_empty(self, sample_fhir_payload):
        ctx = validate_fhir_patient(sample_fhir_payload)
        qr = parse_patient_context(ctx)
        assert len(qr.symptoms.strip()) >= 5

    def test_multiple_conditions_joined(self):
        payload = {
            "patient": {"id": "p1", "gender": "female", "birthDate": "1995-01-01"},
            "conditions": [
                {"resourceType": "Condition", "code": {"text": "High fever for 4 days"}},
                {"resourceType": "Condition", "code": {"text": "Severe abdominal pain"}},
            ],
            "observations": [],
        }
        ctx = validate_fhir_patient(payload)
        qr = parse_patient_context(ctx)
        assert "High fever" in qr.symptoms
        assert "abdominal pain" in qr.symptoms


class TestFHIREndpoints:

    def test_fhir_health_returns_200(self, client):
        r = client.get("/api/v1/fhir/health")
        assert r.status_code == 200

    def test_fhir_health_is_capability_statement(self, client):
        r = client.get("/api/v1/fhir/health")
        body = r.json()
        assert body["resourceType"] == "CapabilityStatement"
        assert body["status"] == "active"
        assert body["fhirVersion"] == "4.0.1"

    def test_fhir_patient_context_valid_payload(
        self, client, sample_fhir_payload, mock_retrieval, mock_fhir_llm
    ):
        r = client.post("/api/v1/fhir/patient-context", json=sample_fhir_payload)
        assert r.status_code == 200

    def test_fhir_patient_context_returns_diagnostic_response(
        self, client, sample_fhir_payload, mock_retrieval, mock_fhir_llm
    ):
        r = client.post("/api/v1/fhir/patient-context", json=sample_fhir_payload)
        body = r.json()
        assert "diagnoses" in body
        assert "follow_up_questions" in body
        assert "disclaimer" in body

    def test_fhir_patient_context_malformed_400(self, client):
        r = client.post("/api/v1/fhir/patient-context", json={"bad": "payload"})
        assert r.status_code == 400

    def test_fhir_patient_context_missing_gender_400(self, client):
        r = client.post("/api/v1/fhir/patient-context", json={
            "patient": {"id": "p1"},
            "conditions": [{"code": {"text": "Fever"}}],
            "observations": [],
        })
        assert r.status_code == 400

    def test_fhir_400_returns_operation_outcome(self, client):
        r = client.post("/api/v1/fhir/patient-context", json={"bad": "payload"})
        assert r.status_code == 400
        body = r.json()
        assert body.get("resourceType") == "OperationOutcome" or "error" in body

    def test_fhir_empty_clinical_data_400(self, client):
        r = client.post("/api/v1/fhir/patient-context", json={
            "patient": {"id": "p1", "gender": "male"},
            "conditions": [],
            "observations": [],
        })
        assert r.status_code == 400
