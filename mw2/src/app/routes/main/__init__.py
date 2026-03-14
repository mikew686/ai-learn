"""Route blueprints; register in app factory."""

from .about import about_bp
from .health import health_bp
from .root import root_bp
from .examples import examples_bp

__all__ = ["about_bp", "examples_bp", "health_bp", "root_bp"]
