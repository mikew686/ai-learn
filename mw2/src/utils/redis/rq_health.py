"""RQ workers health check returning HealthResult."""

from rq import Worker

from utils.health import HealthResult, health_check
from utils.redis.redis import get_redis_connection


@health_check
def rq_health_check() -> HealthResult:
    """Return HealthResult for RQ workers. Pass when >0 running, fail when 0 or Redis unreachable."""
    try:
        conn = get_redis_connection()
        workers = Worker.all(connection=conn)
        count = len(workers)
        return HealthResult(
            component_name="rq",
            status="pass" if count > 0 else "fail",
            description=f"{count} running",
        )
    except Exception:
        return HealthResult(
            component_name="rq",
            status="fail",
            description="unreachable",
        )
