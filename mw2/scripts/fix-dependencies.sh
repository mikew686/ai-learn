#!/usr/bin/env bash
# Regenerate mw2/docker/requirements.txt from current pip env (pip freeze, excluding editable installs).
# Run with the venv activated that has mw2 deps installed (e.g. from ai-learn root: pip install -e .).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQUIREMENTS_TXT="${SCRIPT_DIR}/../docker/requirements.txt"
pip freeze | grep -v "^-e" > "${REQUIREMENTS_TXT}"
echo "Wrote $(wc -l < "${REQUIREMENTS_TXT}") lines to ${REQUIREMENTS_TXT}"
