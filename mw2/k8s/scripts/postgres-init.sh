#!/usr/bin/env bash
# Create mw2_user (superuser) and mw2_database. Connect with default Postgres creds
# (postgres/postgres, password from postgres-auth secret). Run from host with port-forward
# or inside cluster (e.g. kubectl exec into postgres pod).
set -euo pipefail

export PGHOST="${PGHOST:-localhost}"
export PGPORT="${PGPORT:-5432}"
export PGUSER="${PGUSER:-postgres}"
export PGPASSWORD="${PGPASSWORD:-localdev}"
export PGDATABASE="${PGDATABASE:-postgres}"

psql -v ON_ERROR_STOP=1 -d "$PGDATABASE" <<'SQL'
-- Superuser for mw2 app; idempotent (drop if exists then create for clean reruns)
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'mw2_user') THEN
    ALTER ROLE mw2_user WITH PASSWORD 'local123';
  ELSE
    CREATE ROLE mw2_user WITH LOGIN SUPERUSER PASSWORD 'local123';
  END IF;
END
$$;

SELECT 'Role mw2_user ok' AS step;

-- Database owned by mw2_user; idempotent
SELECT 'CREATE DATABASE mw2_database OWNER mw2_user'
WHERE NOT EXISTS (SELECT 1 FROM pg_database WHERE datname = 'mw2_database')\gexec

SELECT 'Database mw2_database ok' AS step;
SQL

echo "Done: mw2_user (superuser, password local123), mw2_database (owner mw2_user)."
