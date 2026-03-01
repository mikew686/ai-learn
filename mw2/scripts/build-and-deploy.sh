#!/usr/bin/env bash
# Build mw2 image, apply local overlay, and restart app/rqworker so they use the new image.
# Run from anywhere. Requires kubectl and Docker.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OVERLAY="$REPO_ROOT/mw2/k8s/overlays/local"
NAMESPACE="${NAMESPACE:-mw2}"

echo "=== Building mw2 image ==="
"$SCRIPT_DIR/build-mw2.sh"

echo "=== Applying local overlay ==="
kubectl apply -k "$OVERLAY"

echo "=== Restarting app and rqworker to pick up new image ==="
kubectl rollout restart deployment/app deployment/rqworker -n "$NAMESPACE"

echo "Done. Watch pods: kubectl get pods -n $NAMESPACE -w"
