import time
from datetime import datetime


def test_health_returns_200(client):
    r = client.get("/api/v1/health")
    assert r.status_code == 200


def test_health_responds_under_100ms(client):
    t0 = time.perf_counter()
    r = client.get("/api/v1/health")
    elapsed_ms = (time.perf_counter() - t0) * 1000
    assert r.status_code == 200
    assert elapsed_ms < 100, f"Health check took {elapsed_ms:.1f}ms — must be under 100ms"


def test_health_status_is_ok(client):
    r = client.get("/api/v1/health")
    body = r.json()
    assert body["status"] == "ok"


def test_health_has_required_fields(client):
    r = client.get("/api/v1/health")
    body = r.json()
    required = ["status", "version", "timestamp", "llm_provider", "collection"]
    for field in required:
        assert field in body, f"Missing field: {field}"


def test_health_timestamp_is_iso8601(client):
    r = client.get("/api/v1/health")
    timestamp = r.json()["timestamp"]
    datetime.fromisoformat(timestamp)


def test_health_content_type_json(client):
    r = client.get("/api/v1/health")
    assert "application/json" in r.headers.get("content-type", "")


def test_health_has_x_total_ms_header(client):
    r = client.get("/api/v1/health")
    assert "x-total-ms" in r.headers or "X-Total-Ms" in r.headers
