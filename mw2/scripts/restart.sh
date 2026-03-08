#!/usr/bin/env bash
# Restart app and rqworker deployments. Use after patching secrets or to pick up config changes.
# Run from anywhere. Requires kubectl.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAMESPACE="${NAMESPACE:-mw2}"

echo "=== Restarting app and rqworker ==="
kubectl rollout restart deployment/app deployment/rqworker -n "$NAMESPACE"

echo "Done. Watch pods: kubectl get pods -n $NAMESPACE -w"
