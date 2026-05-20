"""
GoTHoMIS mock client — simulates FHIR R4 patient context submission to MediAssist bridge.

Run the API server first:
    uvicorn src.api.main:app --reload

Then run this script:
    python scripts/gothomis_mock_client.py
"""

import json
import sys
from datetime import date

import requests


API_BASE = "http://localhost:8000"
FHIR_ENDPOINT = f"{API_BASE}/api/v1/fhir/patient-context"
FHIR_HEALTH   = f"{API_BASE}/api/v1/fhir/health"
TIMEOUT_SECS  = 90

MOCK_FHIR_PAYLOAD = {
    "patient": {
        "resourceType": "Patient",
        "id": "gothomis-patient-001",
        "meta": {"versionId": "1"},
        "name": [{"use": "official", "family": "Mwangi", "given": ["James"]}],
        "gender": "male",
        "birthDate": "1990-06-15",
        "address": [{"city": "Dar es Salaam", "country": "Tanzania"}],
    },
    "conditions": [
        {
            "resourceType": "Condition",
            "id": "condition-001",
            "code": {
                "coding": [
                    {
                        "system": "http://snomed.info/sct",
                        "code": "386661006",
                        "display": "Fever"
                    }
                ],
                "text": "High fever with chills and rigors for 3 days"
            },
            "clinicalStatus": {
                "coding": [{"code": "active"}]
            },
            "onsetDateTime": "2024-06-10",
            "subject": {"reference": "Patient/gothomis-patient-001"}
        },
        {
            "resourceType": "Condition",
            "id": "condition-002",
            "code": {
                "text": "Severe headache and generalised body aches"
            },
            "clinicalStatus": {
                "coding": [{"code": "active"}]
            },
            "subject": {"reference": "Patient/gothomis-patient-001"}
        },
        {
            "resourceType": "Condition",
            "id": "condition-003",
            "code": {
                "text": "Loss of appetite and general weakness"
            },
            "clinicalStatus": {
                "coding": [{"code": "active"}]
            },
            "subject": {"reference": "Patient/gothomis-patient-001"}
        },
    ],
    "observations": [
        {
            "resourceType": "Observation",
            "id": "obs-temp",
            "status": "final",
            "code": {
                "coding": [
                    {
                        "system": "http://loinc.org",
                        "code": "8310-5",
                        "display": "Body temperature"
                    }
                ],
                "text": "Body temperature"
            },
            "valueQuantity": {
                "value": 39.2,
                "unit": "°C",
                "system": "http://unitsofmeasure.org",
                "code": "Cel"
            },
            "effectiveDateTime": "2024-06-13T08:30:00+03:00",
            "subject": {"reference": "Patient/gothomis-patient-001"}
        },
        {
            "resourceType": "Observation",
            "id": "obs-bp",
            "status": "final",
            "code": {
                "text": "Blood pressure"
            },
            "valueString": "110/70 mmHg",
            "effectiveDateTime": "2024-06-13T08:30:00+03:00",
        },
        {
            "resourceType": "Observation",
            "id": "obs-weight",
            "status": "final",
            "code": {
                "text": "Body weight"
            },
            "valueQuantity": {
                "value": 68,
                "unit": "kg"
            },
        },
    ]
}


def _divider(char: str = "─", width: int = 65) -> str:
    return char * width


def _print_header() -> None:
    print(_divider("═"))
    print("  MediAssist — GoTHoMIS FHIR Mock Client")
    print("  Simulating GoTHoMIS sending patient data to FHIR bridge")
    print(_divider("═"))
    print()


def _check_fhir_health() -> bool:
    """Check /api/v1/fhir/health before sending patient data."""
    print("Step 1: Checking FHIR bridge liveness...")
    try:
        r = requests.get(FHIR_HEALTH, timeout=5)
        if r.status_code == 200:
            cs = r.json()
            print(f"  Bridge status    : ✓ active")
            print(f"  FHIR version     : {cs.get('fhirVersion', '?')}")
            print(f"  resourceType     : {cs.get('resourceType', '?')}")
            return True
        else:
            print(f"  FAIL: /fhir/health returned HTTP {r.status_code}")
            return False
    except requests.ConnectionError:
        print("  FAIL: Could not connect to API server.")
        print("  Make sure the server is running:")
        print("    uvicorn src.api.main:app --reload")
        return False


def _print_patient_summary() -> None:
    """Print the mock patient details being sent."""
    pt = MOCK_FHIR_PAYLOAD["patient"]
    conds = MOCK_FHIR_PAYLOAD["conditions"]
    obs   = MOCK_FHIR_PAYLOAD["observations"]

    print(_divider())
    print("Step 2: Patient payload summary")
    print(_divider())
    print(f"  GoTHoMIS Patient ID : {pt['id']}")
    print(f"  Name                : {pt['name'][0]['given'][0]} {pt['name'][0]['family']}")
    print(f"  Gender              : {pt['gender']}")
    print(f"  Date of Birth       : {pt['birthDate']}")
    born_year = int(pt['birthDate'].split('-')[0])
    age = date.today().year - born_year
    print(f"  Age (approx)        : {age} years")
    print()
    print(f"  Conditions ({len(conds)}):")
    for c in conds:
        print(f"    - {c['code']['text']}")
    print()
    print(f"  Observations ({len(obs)}):")
    for o in obs:
        code_text = o['code'].get('text') or o['code'].get('coding', [{}])[0].get('display', '?')
        value = (o.get('valueString') or
                 f"{o['valueQuantity']['value']} {o['valueQuantity']['unit']}"
                 if 'valueQuantity' in o else '?')
        print(f"    - {code_text}: {value}")
    print()


def _send_fhir_request() -> dict:
    """POST the FHIR payload and return the JSON response."""
    print(_divider())
    print(f"Step 3: POSTing to {FHIR_ENDPOINT}")
    print(f"        (timeout: {TIMEOUT_SECS}s — LLM call may take up to 60s)")
    print(_divider())

    r = requests.post(
        FHIR_ENDPOINT,
        json=MOCK_FHIR_PAYLOAD,
        headers={"Content-Type": "application/json"},
        timeout=TIMEOUT_SECS,
    )

    print(f"  HTTP status         : {r.status_code}")
    print(f"  X-Retrieval-Ms      : {r.headers.get('X-Retrieval-Ms', '-')}ms")
    print(f"  X-LLM-Ms            : {r.headers.get('X-LLM-Ms', '-')}ms")
    print(f"  X-Total-Ms          : {r.headers.get('X-Total-Ms', '-')}ms")

    if r.status_code != 200:
        print(f"\n  ERROR response body:")
        print(f"  {json.dumps(r.json(), indent=2)}")
        sys.exit(1)

    return r.json()


def _print_diagnostic_response(data: dict) -> None:
    """Pretty-print the DiagnosticResponse."""
    print()
    print(_divider("═"))
    print("  DIAGNOSTIC RESPONSE")
    print(_divider("═"))

    diagnoses = data.get("diagnoses", [])
    print(f"\n  Differential diagnoses ({len(diagnoses)}):")
    print(_divider())
    for d in diagnoses:
        print(f"\n  [{d['rank']}] {d['condition']}")
        print(f"      Probability   : {d['probability']}%")
        print(f"      Reasoning     : {d['reasoning'][:120]}...")
        print(f"      Evidence      : {d['evidence'][:100]}...")
        print(f"      Source        : {d['source_section']}")

    print()
    print(_divider())
    fq = data.get("follow_up_questions", [])
    print(f"  Follow-up questions ({len(fq)}):")
    for q in fq:
        print(f"    • {q}")

    rt = data.get("recommended_tests", [])
    if rt:
        print(f"\n  Recommended tests ({len(rt)}):")
        for t in rt:
            print(f"    • {t}")

    print(f"\n  Confidence (LLM)      : {data.get('confidence_overall', '?')}")
    print(f"  Confidence (pipeline) : {data.get('pipeline_confidence', 'N/A')}")

    warning = data.get("warning")
    if warning:
        print(f"\n  ⚠️  WARNING: {warning[:200]}")

    print(f"\n  Disclaimer: {data.get('disclaimer', '')[:120]}...")
    print()
    print(_divider("═"))
    print("  Mock client completed successfully — exit code 0")
    print(_divider("═"))


def main() -> None:
    _print_header()

    if not _check_fhir_health():
        sys.exit(1)
    print()

    _print_patient_summary()
    data = _send_fhir_request()

    diagnoses = data.get("diagnoses", [])
    if not diagnoses:
        print("  FAIL: DiagnosticResponse has no diagnoses")
        sys.exit(1)

    _print_diagnostic_response(data)
    sys.exit(0)


if __name__ == "__main__":
    main()
