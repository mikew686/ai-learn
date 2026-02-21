#!/usr/bin/env bash
# Build t7e/postgres-pgvector:16 for the local overlay. Run from anywhere; uses repo root as build context.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
docker build -f "$REPO_ROOT/t7e/docker/postgres-pgvector.Dockerfile" -t t7e/postgres-pgvector:16 "$REPO_ROOT"
