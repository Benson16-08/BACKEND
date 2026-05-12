import requests

BASE_URL = "http://127.0.0.1:8000/api/v1"

def test_health():
    url = f"{BASE_URL}/health"
    response = requests.get(url)
    print(f"GET {url} -> {response.status_code}")
    print(response.json())

def test_query():
    url = f"{BASE_URL}/query"
    # Adjust payload as needed for your API
    payload = {"example": "test"}
    response = requests.post(url, json=payload)
    print(f"POST {url} -> {response.status_code}")
    try:
        print(response.json())
    except Exception:
        print(response.text)

if __name__ == "__main__":
    print("Testing /health endpoint:")
    test_health()
    print("\nTesting /query endpoint:")
    test_query()
