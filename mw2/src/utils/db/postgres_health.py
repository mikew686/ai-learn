"""Postgres health checks returning HealthResult."""

from sqlalchemy import text

from utils.db import get_database_connection
from utils.health import HealthResult, health_check


@health_check
def pg_db_health_check() -> HealthResult:
    """Connect to Postgres, return table count. Pass/fail."""
    try:
        with get_database_connection() as conn:
            table_result = conn.execute(
                text(
                    "SELECT count(*) FROM information_schema.tables "
                    "WHERE table_schema = 'public' AND table_type = 'BASE TABLE'"
                )
            )
            table_count = table_result.scalar() or 0
        return HealthResult(
            component_name="postgres",
            status="pass",
            description=f"{table_count} tables",
        )
    except Exception:
        return HealthResult(
            component_name="postgres",
            status="fail",
            description="unreachable",
        )


@health_check
def pg_vector_health_check() -> HealthResult:
    """Check if pgvector extension is loaded. Pass when available, warn when not."""
    try:
        with get_database_connection() as conn:
            result = conn.execute(
                text("SELECT 1 FROM pg_catalog.pg_extension WHERE extname = 'vector'")
            )
            vector_available = result.scalar() is not None
        return HealthResult(
            component_name="pgvector",
            status="pass" if vector_available else "warn",
            description="available" if vector_available else "not loaded",
        )
    except Exception:
        return HealthResult(
            component_name="pgvector",
            status="fail",
            description="unreachable",
        )
