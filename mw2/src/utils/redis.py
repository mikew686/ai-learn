"""Redis connection utility using environment configuration."""

import os
from typing import Any, Dict

from redis import Redis

from config import load_config


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


def _get_redis_url() -> str:
    url = os.getenv("REDIS_URL")
    if url:
        return url
    host = os.getenv("REDIS_HOST", "localhost")
    port = os.getenv("REDIS_PORT", "6379")
    db = os.getenv("REDIS_DB", "0")
    return f"redis://{host}:{port}/{db}"


def get_redis_connection() -> Redis:
    """
    Return a Redis connection using REDIS_URL and optional connection parameters.

    If REDIS_URL is unset, builds from REDIS_HOST, REDIS_PORT, REDIS_DB
    (defaults: localhost, 6379, 0). Only parameters explicitly set in environment
    are passed to redis-py.

    Supported environment configuration variables:

    REDIS_URL (or REDIS_HOST, REDIS_PORT, REDIS_DB)
    REDIS_SOCKET_TIMEOUT
    REDIS_SOCKET_CONNECT_TIMEOUT
    REDIS_SOCKET_KEEPALIVE
    REDIS_HEALTH_CHECK_INTERVAL
    REDIS_RETRY_ON_TIMEOUT
    REDIS_DECODE_RESPONSES
    REDIS_PROTOCOL
    REDIS_CLIENT_NAME
    REDIS_SSL
    REDIS_SSL_CERTFILE
    REDIS_SSL_KEYFILE
    REDIS_SSL_CA_CERTS
    REDIS_SSL_CERT_REQS
    """

    load_config()

    kwargs: Dict[str, Any] = {}

    _add_if_set(kwargs, "socket_timeout", _parse_int(os.getenv("REDIS_SOCKET_TIMEOUT")))
    _add_if_set(
        kwargs,
        "socket_connect_timeout",
        _parse_int(os.getenv("REDIS_SOCKET_CONNECT_TIMEOUT")),
    )
    _add_if_set(
        kwargs, "socket_keepalive", _parse_bool(os.getenv("REDIS_SOCKET_KEEPALIVE"))
    )
    _add_if_set(
        kwargs,
        "health_check_interval",
        _parse_int(os.getenv("REDIS_HEALTH_CHECK_INTERVAL")),
    )
    _add_if_set(
        kwargs, "retry_on_timeout", _parse_bool(os.getenv("REDIS_RETRY_ON_TIMEOUT"))
    )
    _add_if_set(
        kwargs, "decode_responses", _parse_bool(os.getenv("REDIS_DECODE_RESPONSES"))
    )
    _add_if_set(kwargs, "protocol", _parse_int(os.getenv("REDIS_PROTOCOL")))
    _add_if_set(kwargs, "client_name", os.getenv("REDIS_CLIENT_NAME"))

    # TLS / SSL options
    _add_if_set(kwargs, "ssl", _parse_bool(os.getenv("REDIS_SSL")))
    _add_if_set(kwargs, "ssl_certfile", os.getenv("REDIS_SSL_CERTFILE"))
    _add_if_set(kwargs, "ssl_keyfile", os.getenv("REDIS_SSL_KEYFILE"))
    _add_if_set(kwargs, "ssl_ca_certs", os.getenv("REDIS_SSL_CA_CERTS"))
    _add_if_set(kwargs, "ssl_cert_reqs", os.getenv("REDIS_SSL_CERT_REQS"))

    return Redis.from_url(_get_redis_url(), **kwargs)
