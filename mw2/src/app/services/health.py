"""Shared health-check logic for health page and API."""
import os

import redis
from flask import current_app
from rq import Worker
from sqlalchemy import create_engine, text


def running_on_kubernetes():
    """True if process is running inside a Kubernetes pod."""
    return os.environ.get("KUBERNETES_SERVICE_HOST") is not None


def redis_available():
    """Ping Redis; return True if reachable."""
    try:
        r = redis.from_url(current_app.config["REDIS_URL"])
        r.ping()
        return True
    except Exception:
        return False


def rq_workers_count():
    """Return number of RQ workers currently registered (0 if Redis unreachable or no workers)."""
    try:
        conn = redis.from_url(current_app.config["REDIS_URL"])
        workers = Worker.all(connection=conn)
        return len(workers)
    except Exception:
        return 0


def postgres_status():
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


def get_health_context():
    """Return dict of health status for templates and API."""
    redis_enabled = redis_available()
    postgres_enabled, postgres_vector_available = postgres_status()
    rq_worker_count = rq_workers_count()
    return {
        "redis_enabled": redis_enabled,
        "postgres_enabled": postgres_enabled,
        "postgres_vector_available": postgres_vector_available,
        "rq_workers_running": rq_worker_count > 0,
        "rq_worker_count": rq_worker_count,
        "running_on_kubernetes": running_on_kubernetes(),
    }
