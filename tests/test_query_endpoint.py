import pytest

from tests.conftest import SAMPLE_SYMPTOMS, SAMPLE_CHUNKS


def test_query_happy_path_200(client, mock_retrieval, mock_llm):
    r = client.post("/api/v1/query", json={"symptoms": SAMPLE_SYMPTOMS})
    assert r.status_code == 200


def test_query_returns_diagnostic_response_schema(client, mock_retrieval, mock_llm):
    r = client.post("/api/v1/query", json={"symptoms": SAMPLE_SYMPTOMS})
    body = r.json()
    for field in ["diagnoses", "follow_up_questions", "disclaimer"]:
        assert field in body, f"Missing field: {field}"


def test_query_diagnoses_are_ranked(client, mock_retrieval, mock_llm):
    r = client.post("/api/v1/query", json={"symptoms": SAMPLE_SYMPTOMS})
    diagnoses = r.json()["diagnoses"]
    assert len(diagnoses) >= 1
    ranks = [d["rank"] for d in diagnoses]
    assert ranks == list(range(1, len(ranks) + 1)), f"Ranks not sequential: {ranks}"


def test_query_probabilities_capped_at_99(client, mock_retrieval, mock_llm):
    r = client.post("/api/v1/query", json={"symptoms": SAMPLE_SYMPTOMS})
    diagnoses = r.json()["diagnoses"]
    for d in diagnoses:
        assert d["probability"] <= 99, (
            f"{d['condition']} has probability {d['probability']} — must be ≤ 99"
        )


def test_query_every_diagnosis_has_evidence(client, mock_retrieval, mock_llm):
    r = client.post("/api/v1/query", json={"symptoms": SAMPLE_SYMPTOMS})
    diagnoses = r.json()["diagnoses"]
    for d in diagnoses:
        assert d.get("evidence"), f"{d['condition']} has no evidence"
        assert len(d["evidence"]) >= 10


def test_query_has_follow_up_questions(client, mock_retrieval, mock_llm):
    r = client.post("/api/v1/query", json={"symptoms": SAMPLE_SYMPTOMS})
    fuq = r.json()["follow_up_questions"]
    assert len(fuq) >= 1


def test_query_has_disclaimer(client, mock_retrieval, mock_llm):
    r = client.post("/api/v1/query", json={"symptoms": SAMPLE_SYMPTOMS})
    disclaimer = r.json()["disclaimer"]
    assert len(disclaimer) > 20


def test_query_with_full_patient_metadata(client, mock_retrieval, mock_llm):
    r = client.post("/api/v1/query", json={
        "symptoms": SAMPLE_SYMPTOMS,
        "patient_age": 28,
        "gender": "female",
        "temperature": "38.5C",
        "blood_pressure": "120/80",
        "weight": "65kg",
        "allergy": "Penicillin",
    })
    assert r.status_code == 200


def test_query_has_timing_headers(client, mock_retrieval, mock_llm):
    r = client.post("/api/v1/query", json={"symptoms": SAMPLE_SYMPTOMS})
    assert r.status_code == 200
    headers_lower = {k.lower(): v for k, v in r.headers.items()}
    assert "x-total-ms" in headers_lower


def test_query_empty_symptoms_422(client):
    r = client.post("/api/v1/query", json={"symptoms": ""})
    assert r.status_code == 422


def test_query_whitespace_symptoms_422(client):
    r = client.post("/api/v1/query", json={"symptoms": "    "})
    assert r.status_code == 422


def test_query_too_short_symptoms_422(client):
    r = client.post("/api/v1/query", json={"symptoms": "hi"})
    assert r.status_code == 422


def test_query_missing_symptoms_field_422(client):
    r = client.post("/api/v1/query", json={})
    assert r.status_code == 422


def test_query_422_has_structured_error_body(client):
    r = client.post("/api/v1/query", json={"symptoms": ""})
    assert r.status_code == 422
    body = r.json()
    assert "error" in body or "detail" in body


def test_query_symptoms_too_long_422(client):
    r = client.post("/api/v1/query", json={"symptoms": "x" * 2001})
    assert r.status_code == 422


def test_query_llm_unavailable_503(client, mock_retrieval, mocker):
    from src.core.exceptions import LLMUnavailableError

    mocker.patch(
        "src.api.routes.query.run_pipeline",
        side_effect=LLMUnavailableError(),
    )
    r = client.post("/api/v1/query", json={"symptoms": SAMPLE_SYMPTOMS})
    assert r.status_code == 503
    body = r.json()
    assert body.get("error") == "LLM_UNAVAILABLE"
    assert "message" in body
    assert "timestamp" in body


def test_query_llm_timeout_504(client, mock_retrieval, mocker):
    from src.core.exceptions import LLMTimeoutError

    mocker.patch(
        "src.api.routes.query.run_pipeline",
        side_effect=LLMTimeoutError(),
    )
    r = client.post("/api/v1/query", json={"symptoms": SAMPLE_SYMPTOMS})
    assert r.status_code == 504
    assert r.json().get("error") == "LLM_TIMEOUT"


def test_query_retrieval_failed_503(client, mocker):
    from src.core.exceptions import RetrievalFailedError

    async def _fail(*args, **kwargs):
        raise RetrievalFailedError()

    mocker.patch("src.api.routes.query.retrieve_chunks", side_effect=_fail)
    r = client.post("/api/v1/query", json={"symptoms": SAMPLE_SYMPTOMS})
    assert r.status_code == 503
    assert r.json().get("error") == "RETRIEVAL_FAILED"


def test_query_stream_returns_200(client, mock_retrieval, mock_llm):
    with client.stream(
        "POST", "/api/v1/query/stream",
        json={"symptoms": SAMPLE_SYMPTOMS}
    ) as r:
        assert r.status_code == 200


def test_query_stream_content_type_sse(client, mock_retrieval, mock_llm):
    with client.stream(
        "POST", "/api/v1/query/stream",
        json={"symptoms": SAMPLE_SYMPTOMS}
    ) as r:
        ct = r.headers.get("content-type", "")
        assert "text/event-stream" in ct


def test_query_stream_ends_with_done(client, mock_retrieval, mock_llm):
    with client.stream(
        "POST", "/api/v1/query/stream",
        json={"symptoms": SAMPLE_SYMPTOMS}
    ) as r:
        content = b"".join(r.iter_bytes()).decode()
    assert "[DONE]" in content


def test_query_stream_empty_symptoms_422(client):
    r = client.post("/api/v1/query/stream", json={"symptoms": ""})
    assert r.status_code == 422
