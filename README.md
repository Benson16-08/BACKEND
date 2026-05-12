MediAssist Backend
MediAssist is an AI-powered clinical decision support system designed to assist healthcare professionals in generating differential diagnoses based on patient symptoms.
This backend is built with Python and FastAPI, integrating retrieval-augmented generation (RAG) pipelines and large language models (LLMs) for clinical reasoning.

🚀 Features
RESTful API built with FastAPI

Symptom Analysis Endpoint (/api/v1/query)

Stage 1: Stub responses for frontend development

Stage 3: Full RAG + LLM pipeline integration

Streaming Diagnosis (/api/v1/query/stream) with Server-Sent Events (SSE)

Health Check Endpoint (/api/v1/health) for liveness/readiness probes

Structured Logging with retrieval, LLM, and total latency metrics

Typed Configuration via .env and pydantic-settings

Robust Error Handling with consistent JSON error shapes

Security: CORS, input validation, safety checks, and probability caps

📂 Project Structure
Code
src/
│── api/
│   ├── main.py          # FastAPI app factory
│   └── routes/
│       ├── health.py    # Health check endpoint
│       └── query.py     # Symptom analysis routes
│
│── core/
│   ├── config.py        # Typed configuration
│   ├── exceptions.py    # Custom HTTP exceptions
│   └── logger.py        # Structured logging
│
│── schemas/
│   ├── query.py         # Query request/response models
│   ├── diagnosis.py     # Diagnostic response models
│   └── errors.py        # Error response models
│
│── services/            # AI engine, DB, cache, notifications (future)
│── tests/               # Unit tests
│── migrations/          # Database migrations
⚙️ Configuration
All environment variables are defined in .env and loaded via pydantic-settings.

Example .env:

Code
LLM_PROVIDER=openai
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-3.5-turbo
CHROMA_DB_PATH=./knowledge_base/chroma_db
CHROMA_COLLECTION_NAME=mediassist_stg
PORT=8000
HOST=0.0.0.0
ALLOWED_ORIGINS=http://localhost:5173,http://localhost:3000
RETRIEVAL_TOP_K=5
MAX_DIAGNOSIS_PROBABILITY=99
🩺 API Endpoints
Health Check  
GET /api/v1/health → Returns liveness status and metadata

Symptom Query  
POST /api/v1/query → Returns ranked differential diagnoses

Streaming Query  
POST /api/v1/query/stream → Streams diagnosis tokens via SSE

📝 Logging
Every request logs:

retrieval_ms → Retrieval pipeline latency

llm_ms → LLM generation latency

total_ms → End-to-end latency

Logs are written to both stdout and mediassist.log in structured JSON format.

🛡️ Error Handling
422 → Input validation errors (empty/short symptoms)

503 → Service unavailable (LLM/ChromaDB issues)

504 → Gateway timeout (LLM exceeded budget)

500 → Internal server error

400 → FHIR validation errors (HL7 R4 compliant)

🧪 Development
Run locally with Uvicorn:

bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
📊 Roadmap
Stage 1 → Stub responses for frontend integration

Stage 2 → Retrieval pipeline (ChromaDB + BM25)

Stage 3 → Full RAG + LLM pipeline

Stage 5 → FHIR compliance endpoints

Stage 6 → Benchmarking and performance reporting
