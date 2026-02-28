"""REST API endpoints."""
from flask import Blueprint, jsonify

from app.routes.index import (
    _postgres_status,
    _redis_available,
    _running_on_kubernetes,
    _rq_workers_count,
)

api_bp = Blueprint("api", __name__, url_prefix="/api")


@api_bp.route("/status")
def status():
    """Return Redis, Postgres, pgvector, and RQ worker status as JSON."""
    redis_enabled = _redis_available()
    postgres_enabled, postgres_vector_available = _postgres_status()
    rq_worker_count = _rq_workers_count()
    return jsonify({
        "redis_enabled": redis_enabled,
        "postgres_enabled": postgres_enabled,
        "postgres_vector_available": postgres_vector_available,
        "rq_workers_running": rq_worker_count > 0,
        "rq_worker_count": rq_worker_count,
        "running_on_kubernetes": _running_on_kubernetes(),
    })
