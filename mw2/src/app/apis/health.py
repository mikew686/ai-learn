"""REST API: IETF health check and status for dashboard polling."""

import json
from datetime import datetime, timezone

from flask import Blueprint, Response, jsonify

from app.services.health import get_health_context
from utils.health.run_checks import run_health_checks

HEALTH_JSON = "application/health+json"

health_api_bp = Blueprint("health_api", __name__, url_prefix="/mw2/v1")


def _iso_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _ietf_payload() -> tuple[dict, str, int]:
    """
    Build IETF health payload. Returns (payload, status, http_code).
    Overall status: fail if any component fails, else warn if any warns, else pass.
    HTTP: 503 if fail, else 200.
    """
    results = run_health_checks()
    has_fail = any(r.status == "fail" for r in results)
    has_warn = any(r.status == "warn" for r in results)

    if has_fail:
        overall = "fail"
        http_code = 503
    elif has_warn:
        overall = "warn"
        http_code = 200
    else:
        overall = "pass"
        http_code = 200

    ts = _iso_timestamp()
    checks = {}
    for r in results:
        entry = {
            "status": r.status,
            "observedValue": r.description,
            "time": ts,
        }
        if r.status != "pass":
            entry["output"] = r.description
        checks[r.component_name] = [entry]

    payload = {
        "status": overall,
        "timestamp": ts,
        "checks": checks,
    }
    return payload, overall, http_code


@health_api_bp.route("/health", methods=["GET"])
def health():
    """IETF health check. Returns 503 if any component fails."""
    payload, _status, http_code = _ietf_payload()
    return Response(
        json.dumps(payload),
        status=http_code,
        mimetype=HEALTH_JSON,
        headers={"Cache-Control": "no-cache"},
    )


@health_api_bp.route("/status", methods=["GET"])
def status():
    """Health check results for dashboard polling (redis, postgres, pgvector, rq, ai)."""
    return jsonify(get_health_context())
