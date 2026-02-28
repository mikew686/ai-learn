"""Index route (serves index.html with Redis, Postgres, and RQ worker status)."""
import os

import redis
from flask import Blueprint, current_app, render_template
from rq import Worker
from sqlalchemy import create_engine, text

index_bp = Blueprint("index", __name__)


def _running_on_kubernetes():
    """True if process is running inside a Kubernetes pod (KUBERNETES_SERVICE_HOST is set)."""
    return os.environ.get("KUBERNETES_SERVICE_HOST") is not None


def _redis_available():
    """Ping Redis; return True if reachable."""
    try:
        r = redis.from_url(current_app.config["REDIS_URL"])
        r.ping()
        return True
    except Exception:
        return False


def _rq_workers_count():
    """Return number of RQ workers currently registered (0 if Redis unreachable or no workers)."""
    try:
        conn = redis.from_url(current_app.config["REDIS_URL"])
        workers = Worker.all(connection=conn)
        return len(workers)
    except Exception:
        return 0


def _postgres_status():
    """Connect to Postgres, ensure vector extension exists, return (connected, vector_available)."""
    try:
        engine = create_engine(current_app.config["DATABASE_URL"])
        with engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()
            result = conn.execute(text("SELECT 1 FROM pg_extension WHERE extname = 'vector'"))
            vector_available = result.scalar() is not None
        return True, vector_available
    except Exception:
        return False, False


@index_bp.route("/")
def index():
    redis_enabled = _redis_available()
    postgres_enabled, postgres_vector_available = _postgres_status()
    rq_worker_count = _rq_workers_count()
    return render_template(
        "index.html",
        redis_enabled=redis_enabled,
        postgres_enabled=postgres_enabled,
        postgres_vector_available=postgres_vector_available,
        rq_workers_running=rq_worker_count > 0,
        rq_worker_count=rq_worker_count,
        running_on_kubernetes=_running_on_kubernetes(),
    )
