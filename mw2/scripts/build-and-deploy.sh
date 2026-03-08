#!/usr/bin/env bash
# Build mw2 image, then deploy (apply overlay + restart). Run from anywhere. Requires kubectl and Docker.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Building mw2 image ==="
"$SCRIPT_DIR/build-mw2.sh"

"$SCRIPT_DIR/deploy.sh"
