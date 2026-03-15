"""Route blueprints; re-export from main and mobile subpackages."""

from .main import about_bp, examples_bp, health_bp, root_bp
from .mobile import mobile_example_bp

__all__ = ["about_bp", "examples_bp", "health_bp", "mobile_example_bp", "root_bp"]
