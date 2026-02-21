"""Index route (serves index.html with caller IP)."""
from flask import Blueprint, render_template, request

index_bp = Blueprint("index", __name__)


def _caller_ip():
    """Client IP from request, respecting X-Forwarded-For when behind a proxy."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.remote_addr or "unknown"


@index_bp.route("/")
def index():
    return render_template("index.html", caller_ip=_caller_ip())
