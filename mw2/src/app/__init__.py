"""Flask app factory; registers blueprints only."""

import os

from flask import Flask

from config import load_config
from .apis import health_api_bp
from .routes import about_bp, examples_bp, health_bp, mobile_example_bp, root_bp


def _parse_bool(val: str | None) -> bool | None:
    if val is None:
        return None
    return val.lower() in ("1", "true", "yes")


def _parse_int(val: str | None) -> int | None:
    if val is None:
        return None
    try:
        return int(val)
    except ValueError:
        return None


def create_app(config_overrides=None):
    """Create and configure the Flask application."""
    load_config()
    config = {
        "SECRET_KEY": os.getenv("SECRET_KEY", "dev-secret-change-in-production"),
        "DEBUG": bool(_parse_bool(os.getenv("FLASK_DEBUG", "1"))),
        "TESTING": bool(_parse_bool(os.getenv("FLASK_TESTING", "0"))),
        "SESSION_COOKIE_SECURE": bool(
            _parse_bool(os.getenv("FLASK_SESSION_COOKIE_SECURE", "0"))
        ),
        "PREFERRED_URL_SCHEME": os.getenv("FLASK_PREFERRED_URL_SCHEME", "http"),
    }
    if (v := _parse_bool(os.getenv("FLASK_PROPAGATE_EXCEPTIONS"))) is not None:
        config["PROPAGATE_EXCEPTIONS"] = v
    if (v := _parse_int(os.getenv("FLASK_MAX_CONTENT_LENGTH"))) is not None:
        config["MAX_CONTENT_LENGTH"] = v
    if v := os.getenv("FLASK_SESSION_COOKIE_SAMESITE"):
        config["SESSION_COOKIE_SAMESITE"] = v
    if v := os.getenv("FLASK_TRUSTED_HOSTS"):
        config["TRUSTED_HOSTS"] = [h.strip() for h in v.split(",")]
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(config)
    if config_overrides:
        app.config.update(config_overrides)

    app.register_blueprint(root_bp)
    app.register_blueprint(health_api_bp)  # /mw2/v1/* (IETF health, status)
    app.register_blueprint(health_bp)
    app.register_blueprint(examples_bp)
    app.register_blueprint(about_bp)
    app.register_blueprint(mobile_example_bp)

    return app
