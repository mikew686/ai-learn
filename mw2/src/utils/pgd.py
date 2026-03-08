"""Postgres database connection utility. Uses config for URL and connection defaults."""

from contextlib import contextmanager
from typing import Generator

from sqlalchemy import Engine, create_engine
from sqlalchemy.engine import Connection
from sqlalchemy.orm import Session, sessionmaker

from config import load_config

# Defaults (also in config/settings.py); used if config not yet loaded
DEFAULT_CONNECT_TIMEOUT = 5
DEFAULT_POOL_RECYCLE = 300

_engine: Engine | None = None
_SessionLocal: sessionmaker[Session] | None = None


def get_database_engine() -> Engine:
    """
    Return a SQLAlchemy engine using DATABASE_URL and timeout/pool settings from config.
    Cached so connection and session accessors share one pool.

    Config (env): DATABASE_URL (or PG*); DATABASE_CONNECT_TIMEOUT, DATABASE_POOL_RECYCLE.
    """
    global _engine
    if _engine is None:
        cfg = load_config()
        _engine = create_engine(
            cfg["DATABASE_URL"],
            connect_args={"connect_timeout": cfg["DATABASE_CONNECT_TIMEOUT"]},
            pool_pre_ping=True,
            pool_recycle=cfg["DATABASE_POOL_RECYCLE"],
        )
    return _engine


@contextmanager
def get_database_connection() -> Generator[Connection, None, None]:
    """Yield a database connection; closes on exit. Uses shared engine.
    Must be used as a context manager: with get_database_connection() as conn:
    """
    with get_database_engine().connect() as conn:
        yield conn


@contextmanager
def get_database_session() -> Generator[Session, None, None]:
    """
    Yield a SQLAlchemy ORM session; commits on exit, rolls back on exception.
    Use for ORM models (session.add, session.query, etc.). Uses shared engine.
    Must be used as a context manager: with get_database_session() as session:
    """
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(
            bind=get_database_engine(),
            autocommit=False,
            autoflush=False,
        )
    session = _SessionLocal()
    with session:
        yield session
