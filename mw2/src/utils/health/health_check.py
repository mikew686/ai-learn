"""Decorator and registry for discoverable health check functions."""

from collections.abc import Callable
from typing import ParamSpec, TypeVar

from utils.health.health_result import HealthResult

P = ParamSpec("P")
R = TypeVar("R", bound=HealthResult)

_health_checks: list[Callable[..., HealthResult]] = []


def health_check(fn: Callable[P, R]) -> Callable[P, R]:
    """Decorator that registers a function as a discoverable health check."""

    _health_checks.append(fn)
    return fn


def get_health_checks() -> list[Callable[..., HealthResult]]:
    """Return all registered health check functions."""

    return list(_health_checks)
