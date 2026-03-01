# Build from ai-learn repo root: docker build -f mw2/docker/mw2.Dockerfile .
# Only mw2/docker/requirements.txt and mw2/src are used; no pyproject.toml in the image.
# Source lives at /app/src; PYTHONPATH=/app/src. Config resolves mw2 root to /app (so .env at /app/.env).
# dumb-init as PID 1 forwards SIGINT/SIGTERM to the real process (gunicorn or rqworker).
# Default CMD runs gunicorn; run rqworker: docker run mw2:latest python -m rqworker

FROM python:3.11-slim

LABEL org.opencontainers.image.title="mw2"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src

WORKDIR /app

# dumb-init for proper signal handling (Ctrl+C, SIGTERM) in containers
RUN apt-get update && apt-get install -y --no-install-recommends dumb-init \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install runtime deps first so this layer is cached when only source changes
COPY mw2/docker/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy app source into /app/src
COPY mw2/src ./src

EXPOSE 8000

# Gunicorn defaults; override at runtime with same env vars or with GUNICORN_CMD_ARGS
ENV GUNICORN_WORKERS=4 \
    GUNICORN_BIND=0.0.0.0:8000 \
    GUNICORN_TIMEOUT=30 \
    GUNICORN_LOG_LEVEL=info
ENV GUNICORN_CMD_ARGS="--workers $GUNICORN_WORKERS --bind $GUNICORN_BIND --timeout $GUNICORN_TIMEOUT --log-level $GUNICORN_LOG_LEVEL"

# dumb-init forwards signals to child; CMD is overridden for rqworker (no --entrypoint needed)
ENTRYPOINT ["dumb-init", "--"]
CMD ["gunicorn", "app:create_app"]
