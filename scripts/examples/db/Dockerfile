# File-backed Postgres + pgvector
# Uses official pgvector image; data is persisted via a volume (see docker-compose.yml).
FROM pgvector/pgvector:pg17

# Optional: set env defaults (override in docker-compose or at run time)
ENV POSTGRES_USER=postgres
ENV POSTGRES_PASSWORD=postgres
ENV PGDATA=/var/lib/postgresql/data

# Init scripts in docker-entrypoint-initdb.d run on first start only.
# We use a mounted init script from docker-compose instead of copying here,
# so the same image works with or without custom init.
# See docker-compose.yml and init-pgvector.sql.

USER postgres
EXPOSE 5432
