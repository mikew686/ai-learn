"""AI LLM health check returning HealthResult."""

from utils.ai_llm import AIClient, OpenRouterKeyMissingError
from utils.health import HealthResult, health_check


@health_check
def ai_health_check() -> HealthResult:
    """Check AI client: create client, get models. Pass with model count, fail on token missing or error."""
    try:
        client = AIClient(timeout=10.0)
        model_count = len(client.models())
        return HealthResult(
            component_name="ai",
            status="pass",
            description=f"{model_count} models",
        )
    except OpenRouterKeyMissingError:
        return HealthResult(
            component_name="ai",
            status="fail",
            description="token missing",
        )
    except Exception:
        return HealthResult(
            component_name="ai",
            status="fail",
            description="unreachable",
        )
