"""Load config from environment. Used by Flask app and RQ worker."""
import os


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

    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

    return {
        "SECRET_KEY": os.environ.get("SECRET_KEY", "dev-secret-change-in-production"),
        "ENV": os.environ.get("FLASK_ENV", "development"),
        "DEBUG": os.environ.get("FLASK_DEBUG", "1").lower() in ("1", "true", "yes"),
        "DATABASE_URL": database_url,
        "REDIS_URL": redis_url,
    }
