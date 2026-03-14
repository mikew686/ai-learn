"""Route blueprints; re-export from main subpackage."""

from .main import about_bp, examples_bp, health_bp, root_bp

__all__ = ["about_bp", "examples_bp", "health_bp", "root_bp"]
