"""Load config from environment. Used by Flask app and RQ worker."""

from pathlib import Path

# Environment override in mw2 project root; app, rqworker, and alembic (run from mw2/) can use it.
_mw2_root = Path(__file__).resolve().parent.parent.parent
_dotenv_path = _mw2_root / ".env"
if _dotenv_path.is_file():
    from dotenv import load_dotenv

    load_dotenv(_dotenv_path, override=False)


def load_config():
    """Config dict from env. DATABASE_URL or PG*; REDIS_URL; Flask keys."""
    pass
