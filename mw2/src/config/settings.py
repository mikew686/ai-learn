"""Load config from environment. Used by Flask app and RQ worker."""
import os
from pathlib import Path

# Environment override in mw2 project root; app, rqworker, and alembic (run from mw2/) can use it.
_mw2_root = Path(__file__).resolve().parent.parent.parent
_dotenv_path = _mw2_root / ".env"
if _dotenv_path.is_file():
    from dotenv import load_dotenv
    load_dotenv(_dotenv_path, override=False)


def _int_env(name: str, default: int) -> int:
    """Parse an integer from the environment; return default if unset or invalid."""
    val = os.environ.get(name, "").strip()
    if not val:
        return default
    try:
        return int(val)
    except ValueError:
        return default


def load_config():
    """Config dict from env. DATABASE_URL or PG*; REDIS_URL; Flask keys."""
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        pg_host = os.environ.get("PGHOST", "localhost")
        pg_port = os.environ.get("PGPORT", "5432")
        pg_user = os.environ.get("PGUSER", "postgres")
        pg_password = os.environ.get("PGPASSWORD", "localdev")
        pg_database = os.environ.get("PGDATABASE", "postgres")
        database_url = f"postgresql://{pg_user}:{pg_password}@{pg_host}:{pg_port}/{pg_database}"

    database_connect_timeout = _int_env("DATABASE_CONNECT_TIMEOUT", 5)
    database_pool_recycle = _int_env("DATABASE_POOL_RECYCLE", 300)

    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    redis_socket_timeout = _int_env("REDIS_SOCKET_TIMEOUT", 5)
    redis_socket_connect_timeout = _int_env("REDIS_SOCKET_CONNECT_TIMEOUT", 5)
    redis_health_check_interval = _int_env("REDIS_HEALTH_CHECK_INTERVAL", 30)

    rq_debug = os.environ.get("RQ_DEBUG", "").lower() in ("1", "true", "yes")
    openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")

    return {
        "SECRET_KEY": os.environ.get("SECRET_KEY", "dev-secret-change-in-production"),
        "ENV": os.environ.get("FLASK_ENV", "development"),
        "DEBUG": os.environ.get("FLASK_DEBUG", "1").lower() in ("1", "true", "yes"),
        "RQ_DEBUG": rq_debug,
        "DATABASE_URL": database_url,
        "DATABASE_CONNECT_TIMEOUT": database_connect_timeout,
        "DATABASE_POOL_RECYCLE": database_pool_recycle,
        "REDIS_URL": redis_url,
        "REDIS_SOCKET_TIMEOUT": redis_socket_timeout,
        "REDIS_SOCKET_CONNECT_TIMEOUT": redis_socket_connect_timeout,
        "REDIS_HEALTH_CHECK_INTERVAL": redis_health_check_interval,
        "OPENROUTER_API_KEY": openrouter_api_key,
    }
