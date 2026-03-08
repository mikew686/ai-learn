#!/usr/bin/env bash
# Apply local overlay and restart app/rqworker. Use when config changed but image did not.
# Run from anywhere. Requires kubectl.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OVERLAY="$REPO_ROOT/mw2/k8s/overlays/local"
NAMESPACE="${NAMESPACE:-mw2}"

echo "=== Applying local overlay ==="
kubectl apply -k "$OVERLAY"

echo "=== Restarting app and rqworker ==="
"$SCRIPT_DIR/restart.sh"
