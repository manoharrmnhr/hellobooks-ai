# Stage 1: dependency builder
FROM python:3.12-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: runtime image
FROM python:3.12-slim AS runtime

LABEL org.opencontainers.image.title="Hellobooks AI"
LABEL org.opencontainers.image.description="RAG-powered accounting assistant"
LABEL org.opencontainers.image.version="1.0.0"

RUN useradd --create-home --shell /bin/bash hellobooks

WORKDIR /app

COPY --from=builder /install /usr/local
COPY --chown=hellobooks:hellobooks . .

RUN mkdir -p /app/vector_store /app/model_cache \
    && chown -R hellobooks:hellobooks /app/vector_store /app/model_cache

USER hellobooks

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ANTHROPIC_API_KEY="" \
    EMBEDDING_MODEL="all-MiniLM-L6-v2" \
    LLM_MODEL="claude-sonnet-4-20250514" \
    TOP_K=3 \
    API_HOST="0.0.0.0" \
    API_PORT=8000 \
    LOG_LEVEL="INFO" \
    TRANSFORMERS_CACHE="/app/model_cache" \
    SENTENCE_TRANSFORMERS_HOME="/app/model_cache"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" \
    || exit 1

CMD ["python", "main.py", "serve"]
