# ─────────────────────────────────────────────────────────────────────────────
# MediAssist Backend — Dockerfile
# Stage 6: reproducible container for demo and deployment
#
# Build:  docker build -t mediassist-backend .
# Run:    docker run --env-file .env -p 8000:8000 mediassist-backend
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.12-slim

LABEL maintainer="Benson (Role 4) — MediAssist Backend"
LABEL description="MediAssist clinical decision support API"
LABEL version="1.0.0"

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY scripts/ ./scripts/

COPY knowledge_base/ ./knowledge_base/

RUN useradd --no-create-home --shell /bin/false mediassist \
    && chown -R mediassist:mediassist /app
USER mediassist

ENV PORT=8000 \
    HOST=0.0.0.0 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TOKENIZERS_PARALLELISM=false

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${PORT}/api/v1/health || exit 1

EXPOSE ${PORT}

CMD ["sh", "-c", \
     "uvicorn src.api.main:app \
      --host ${HOST} \
      --port ${PORT} \
      --workers 2 \
      --timeout-keep-alive 65 \
      --log-level info"]
