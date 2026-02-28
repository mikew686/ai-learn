#!/usr/bin/env bash
# Build mw2/postgres-pgvector:16 for the local overlay. Run from anywhere; uses repo root as build context.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
docker build -f "$REPO_ROOT/mw2/docker/postgres-pgvector.Dockerfile" -t mw2/postgres-pgvector:16 "$REPO_ROOT"
