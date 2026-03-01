#!/usr/bin/env bash
# Build mw2 app image (gunicorn/rqworker). Run from anywhere; uses ai-learn repo root as build context.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
docker build -f "$REPO_ROOT/mw2/docker/mw2.Dockerfile" -t mw2:latest "$REPO_ROOT"
