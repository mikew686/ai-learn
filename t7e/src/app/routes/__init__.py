"""Route blueprints; register in app factory."""
from .api import api_bp
from .index import index_bp

__all__ = ["api_bp", "index_bp"]
