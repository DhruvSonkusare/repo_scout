# The deployed app: small enough for a 512 MB free instance. Embeddings come from Gemini, so this
# installs requirements-base.txt, which leaves out FastEmbed and ONNX Runtime and their memory use.
FROM python:3.12-slim

# git clones repos, and needs ca-certificates to reach GitHub over HTTPS.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --uid 1000 --create-home appuser
WORKDIR /app

COPY requirements-base.txt .
RUN pip install --no-cache-dir -r requirements-base.txt

COPY alembic.ini .
COPY alembic ./alembic
COPY app ./app
COPY static ./static
RUN mkdir data && chown -R appuser:appuser /app

# Settings that aren't secrets. Render's environment settings override any of these.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MALLOC_ARENA_MAX=2 \
    DATA_DIR=/app/data \
    LLM_PROVIDER=gemini \
    CHAT_MODELS=gemini-3.5-flash-lite,gemini-3.1-flash-lite,gemini-3-flash-preview \
    GUIDE_MODELS=gemini-3-flash-preview,gemini-3.5-flash-lite,gemini-3.1-flash-lite \
    EMBEDDING_PROVIDER=gemini \
    EMBEDDING_MODEL=gemini-embedding-001 \
    DEMO_MODE=true \
    CLIENT_IP_HEADER=true-client-ip \
    DB_POOL_SIZE=2 \
    DB_MAX_OVERFLOW=3 \
    DB_CONNECT_TIMEOUT_SECONDS=15

USER appuser
# Hosts tell the app which port to use through PORT. 10000 is Render's default.
EXPOSE 10000

# One worker keeps memory low, and background indexing assumes a single server process.
CMD ["sh", "-c", "alembic upgrade head && exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000} --workers 1 --limit-concurrency 20"]
