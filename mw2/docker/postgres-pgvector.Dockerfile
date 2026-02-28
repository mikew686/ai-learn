# Postgres 16 with pgvector from PGDG apt (no source build).
# Build from repo root: docker build -f t7e/docker/postgres-pgvector.Dockerfile -t t7e/postgres-pgvector:16 .
FROM public.ecr.aws/docker/library/postgres:16-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends postgresql-16-pgvector \
  && rm -rf /var/lib/apt/lists/*
