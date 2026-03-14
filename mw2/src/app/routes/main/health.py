"""Health dashboard blueprint: HTML status page."""

from flask import Blueprint, render_template

from app.services.health import get_health_context

health_bp = Blueprint("health", __name__, url_prefix="/health")


@health_bp.route("/dashboard")
def dashboard():
    """System health page with Redis, Postgres, pgvector, and RQ worker status."""
    return render_template("main/health/index.html", **get_health_context())


@health_bp.route("/fragments/services")
def services_fragment():
    """Return services HTML fragment for HTMX polling."""
    return render_template("main/health/_services.html", **get_health_context())
