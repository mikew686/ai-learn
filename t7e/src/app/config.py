"""App configuration from environment."""
import os


def load_config():
    """Load config from environment (and optional .env)."""
    return {
        "SECRET_KEY": os.environ.get("SECRET_KEY", "dev-secret-change-in-production"),
        "ENV": os.environ.get("FLASK_ENV", "development"),
        "DEBUG": os.environ.get("FLASK_DEBUG", "1").lower() in ("1", "true", "yes"),
    }
