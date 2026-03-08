"""Flask app factory; registers blueprints only."""

from flask import Flask

from .config import load_config
from .routes import api_bp, health_bp, root_bp, t7e_bp


def create_app(config_overrides=None):
    """Create and configure the Flask application."""
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(load_config())
    if config_overrides:
        app.config.update(config_overrides)

    app.register_blueprint(root_bp)
    app.register_blueprint(health_bp)
    app.register_blueprint(t7e_bp)
    app.register_blueprint(api_bp)

    return app
