"""RQ workers health check returning HealthResult."""

from rq import Worker

from utils.health import HealthResult, health_check
from utils.redis.redis import get_redis_connection


def _healthy_worker_count(workers: list) -> int:
    """Count workers with state busy or idle (suspended are not healthy)."""
    healthy = 0
    for worker in workers:
        try:
            state = worker.get_state()
            if state in ("busy", "idle"):
                healthy += 1
        except Exception:
            pass
    return healthy


@health_check
def rq_health_check() -> HealthResult:
    """Return HealthResult for RQ workers. Pass when all healthy, warn when some, fail when none."""
    try:
        conn = get_redis_connection()
        workers = list(Worker.all(connection=conn))
        total = len(workers)
        healthy = _healthy_worker_count(workers)

        if healthy == 0:
            status = "fail"
        elif healthy < total:
            status = "warn"
        else:
            status = "pass"

        return HealthResult(
            component_name="rq",
            status=status,
            description=f"{healthy} / {total} workers",
        )
    except Exception:
        return HealthResult(
            component_name="rq",
            status="fail",
            description="unreachable",
        )
