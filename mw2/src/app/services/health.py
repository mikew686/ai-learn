"""Shared health-check logic for health page and API."""

import os

from rq import Worker
from sqlalchemy import text
from utils.ai_llm import AIClient, OpenRouterKeyMissingError
from utils.pgd import get_database_connection
from utils.redis import get_redis_connection


def running_on_kubernetes():
    """True if process is running inside a Kubernetes pod."""
    return os.environ.get("KUBERNETES_SERVICE_HOST") is not None


def redis_status():
    """Return (connected, key_count). key_count=0 if unreachable."""
    try:
        conn = get_redis_connection()
        conn.ping()
        return True, conn.dbsize()
    except Exception:
        return False, 0


def rq_workers_count():
    """Return number of RQ workers currently registered (0 if Redis unreachable or no workers)."""
    try:
        conn = get_redis_connection()
        workers = Worker.all(connection=conn)
        return len(workers)
    except Exception:
        return 0


def ai_client_status():
    """Check AI client: create client, get models. Returns (success, model_count).
    success=False if token missing (OpenRouterKeyMissingError) or other error; model_count=0 then.
    success=True and model_count from client.models() when available.
    """
    try:
        client = AIClient()
        return True, len(client.models())
    except (OpenRouterKeyMissingError, Exception):
        return False, 0


def postgres_status():
    """Connect to Postgres, ensure vector extension exists. Return (connected, vector_available, table_count).
    table_count is from information_schema (public schema, base tables). (False, False, 0) on failure.
    """
    try:
        with get_database_connection() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()
            result = conn.execute(
                text("SELECT 1 FROM pg_catalog.pg_extension WHERE extname = 'vector'")
            )
            vector_available = result.scalar() is not None
            table_result = conn.execute(
                text(
                    "SELECT count(*) FROM information_schema.tables "
                    "WHERE table_schema = 'public' AND table_type = 'BASE TABLE'"
                )
            )
            table_count = table_result.scalar() or 0
        return True, vector_available, table_count
    except Exception:
        return False, False, 0


def get_health_context():
    """Return dict of health status for templates and API."""
    redis_connected, redis_key_count = redis_status()
    postgres_enabled, postgres_vector_available, postgres_table_count = (
        postgres_status()
    )
    rq_worker_count = rq_workers_count()
    ai_success, ai_model_count = ai_client_status()
    return {
        "redis_enabled": redis_connected,
        "redis_key_count": redis_key_count,
        "postgres_enabled": postgres_enabled,
        "postgres_vector_available": postgres_vector_available,
        "postgres_table_count": postgres_table_count,
        "rq_workers_running": rq_worker_count > 0,
        "rq_worker_count": rq_worker_count,
        "ai_success": ai_success,
        "ai_model_count": ai_model_count,
        "ai_available": ai_success,
        "running_on_kubernetes": running_on_kubernetes(),
    }
