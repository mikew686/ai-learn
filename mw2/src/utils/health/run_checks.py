"""Run registered health checks and return formatted context."""

# Import health modules so @health_check decorators register the functions
import utils.ai_llm.ai_llm_health  # noqa: F401
import utils.db.postgres_health  # noqa: F401
import utils.redis.redis_health  # noqa: F401
import utils.redis.rq_health  # noqa: F401

from utils.health import HealthResult, get_health_checks

DISPLAY_ORDER = ("redis", "postgres", "pgvector", "rq", "ai")


def run_health_checks() -> list[HealthResult]:
    """Run all registered health checks and return list of HealthResult."""
    return [fn() for fn in get_health_checks()]


def get_health_context() -> dict:
    """Return dict of health status for templates and API."""
    results = run_health_checks()
    by_component = {r.component_name: r for r in results}
    ordered = [by_component[c] for c in DISPLAY_ORDER if c in by_component]
    ordered.extend(r for r in results if r.component_name not in DISPLAY_ORDER)
    return {
        "health_results": [
            {
                "component_name": r.component_name,
                "status": r.status,
                "description": r.description,
            }
            for r in ordered
        ]
    }
