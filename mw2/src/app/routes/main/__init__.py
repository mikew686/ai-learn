"""Route blueprints; register in app factory."""

from .about import about_bp
from .health import health_bp
from .root import root_bp
from .t7e import t7e_bp

__all__ = ["about_bp", "health_bp", "root_bp", "t7e_bp"]
