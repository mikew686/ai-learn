"""Postgres database connection utility. Uses environment variables for URL and optional settings."""

import os
from contextlib import contextmanager
from typing import Any, Dict, Generator

from sqlalchemy import Engine, create_engine
from sqlalchemy.engine import Connection
from sqlalchemy.orm import Session, sessionmaker

from config import load_config

_engine: Engine | None = None
_SessionLocal: sessionmaker[Session] | None = None


def _add_if_set(target: Dict[str, Any], key: str, value: Any):
    if value is not None:
        target[key] = value


def _parse_int(val: str | None) -> int | None:
    if val is None:
        return None
    try:
        return int(val)
    except ValueError:
        return None


def _parse_bool(val: str | None) -> bool | None:
    if val is None:
        return None
    return val.lower() in ("1", "true", "yes")


def _get_database_url() -> str:
    url = os.getenv("DATABASE_URL")
    if url:
        return url
    host = os.getenv("PGHOST", "localhost")
    port = os.getenv("PGPORT", "5432")
    user = os.getenv("PGUSER", "postgres")
    password = os.getenv("PGPASSWORD", "localdev")
    database = os.getenv("PGDATABASE", "postgres")
    return f"postgresql://{user}:{password}@{host}:{port}/{database}"


def get_database_engine() -> Engine:
    """
    Return a cached SQLAlchemy engine using DATABASE_URL and optional settings.

    If DATABASE_URL is unset, builds from PGHOST, PGPORT, PGUSER, PGPASSWORD, PGDATABASE
    (defaults: localhost, 5432, postgres, localdev, postgres). Only parameters explicitly
    set in environment are passed through.

    Supported environment configuration variables:

    DATABASE_URL (or PGHOST, PGPORT, PGUSER, PGPASSWORD, PGDATABASE)
    DATABASE_CONNECT_TIMEOUT
    DATABASE_POOL_PRE_PING
    DATABASE_POOL_RECYCLE
    DATABASE_POOL_SIZE
    DATABASE_MAX_OVERFLOW
    DATABASE_POOL_TIMEOUT
    DATABASE_POOL_USE_LIFO
    DATABASE_ECHO
    DATABASE_ECHO_POOL
    DATABASE_POOL_RESET_ON_RETURN
    DATABASE_ISOLATION_LEVEL
    DATABASE_CLIENT_ENCODING
    DATABASE_APPLICATION_NAME
    """

    global _engine

    if _engine is None:
        load_config()

        engine_kwargs: Dict[str, Any] = {}
        connect_args: Dict[str, Any] = {}

        _add_if_set(
            connect_args,
            "connect_timeout",
            _parse_int(os.getenv("DATABASE_CONNECT_TIMEOUT")),
        )
        _add_if_set(
            connect_args, "client_encoding", os.getenv("DATABASE_CLIENT_ENCODING")
        )
        _add_if_set(
            connect_args, "application_name", os.getenv("DATABASE_APPLICATION_NAME")
        )

        if connect_args:
            engine_kwargs["connect_args"] = connect_args

        _add_if_set(
            engine_kwargs,
            "pool_pre_ping",
            _parse_bool(os.getenv("DATABASE_POOL_PRE_PING")),
        )
        _add_if_set(
            engine_kwargs,
            "pool_recycle",
            _parse_int(os.getenv("DATABASE_POOL_RECYCLE")),
        )
        _add_if_set(
            engine_kwargs, "pool_size", _parse_int(os.getenv("DATABASE_POOL_SIZE"))
        )
        _add_if_set(
            engine_kwargs,
            "max_overflow",
            _parse_int(os.getenv("DATABASE_MAX_OVERFLOW")),
        )
        _add_if_set(
            engine_kwargs,
            "pool_timeout",
            _parse_int(os.getenv("DATABASE_POOL_TIMEOUT")),
        )
        _add_if_set(
            engine_kwargs,
            "pool_use_lifo",
            _parse_bool(os.getenv("DATABASE_POOL_USE_LIFO")),
        )
        _add_if_set(engine_kwargs, "echo", _parse_bool(os.getenv("DATABASE_ECHO")))
        _add_if_set(
            engine_kwargs, "echo_pool", _parse_bool(os.getenv("DATABASE_ECHO_POOL"))
        )
        _add_if_set(
            engine_kwargs,
            "pool_reset_on_return",
            os.getenv("DATABASE_POOL_RESET_ON_RETURN"),
        )
        _add_if_set(
            engine_kwargs, "isolation_level", os.getenv("DATABASE_ISOLATION_LEVEL")
        )

        _engine = create_engine(_get_database_url(), **engine_kwargs)

    return _engine


def get_sessionmaker() -> sessionmaker[Session]:
    """Return a cached SQLAlchemy sessionmaker bound to the shared engine."""
    global _SessionLocal

    if _SessionLocal is None:
        _SessionLocal = sessionmaker(
            bind=get_database_engine(),
            autoflush=False,
            autocommit=False,
            expire_on_commit=False,
        )

    return _SessionLocal


@contextmanager
def get_database_connection() -> Generator[Connection, None, None]:
    """Yield a database connection from the shared engine."""
    connection = get_database_engine().connect()
    try:
        yield connection
    finally:
        connection.close()


@contextmanager
def get_database_session() -> Generator[Session, None, None]:
    """Yield a database session from the shared sessionmaker."""
    session = get_sessionmaker()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
