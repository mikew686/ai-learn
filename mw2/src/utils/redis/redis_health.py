"""Redis health check returning HealthResult."""

from utils.health import HealthResult, health_check
from utils.redis.redis import get_redis_connection


@health_check
def redis_health_check() -> HealthResult:
    """Return HealthResult with pass/fail and key count description."""
    try:
        conn = get_redis_connection()
        conn.ping()
        key_count = conn.dbsize()
        return HealthResult(
            component_name="redis",
            status="pass",
            description=f"{key_count} keys",
        )
    except Exception:
        return HealthResult(
            component_name="redis",
            status="fail",
            description="unreachable",
        )
