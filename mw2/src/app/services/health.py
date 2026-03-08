"""Shared health-check logic for health page and API."""

from utils.health.run_checks import get_health_context

__all__ = ["get_health_context"]
