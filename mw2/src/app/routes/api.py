"""REST API endpoints."""

from flask import Blueprint, jsonify

from app.services.health import get_health_context

api_bp = Blueprint("api", __name__, url_prefix="/api")


@api_bp.route("/status")
def status():
    """Return Redis, Postgres, pgvector, and RQ worker status as JSON."""
    return jsonify(get_health_context())
