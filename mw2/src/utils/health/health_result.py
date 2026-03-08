"""Health check result type."""

from dataclasses import dataclass
from typing import Literal

HealthStatus = Literal["warn", "fail", "pass"]


@dataclass
class HealthResult:
    """IETF health check result to be returned by health functions."""

    component_name: str
    status: HealthStatus
    description: str
