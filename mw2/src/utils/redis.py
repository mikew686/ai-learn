"""Redis connection utility. Uses config (REDIS_URL and timeout env)."""
from redis import Redis

from config import load_config


def get_redis_connection() -> Redis:
    """
    Return a Redis connection using REDIS_URL and timeout settings from config.

    Config (env): REDIS_URL; REDIS_SOCKET_TIMEOUT, REDIS_SOCKET_CONNECT_TIMEOUT,
    REDIS_HEALTH_CHECK_INTERVAL (see config/settings.py defaults).
    """
    cfg = load_config()
    return Redis.from_url(
        cfg["REDIS_URL"],
        socket_timeout=cfg["REDIS_SOCKET_TIMEOUT"],
        socket_connect_timeout=cfg["REDIS_SOCKET_CONNECT_TIMEOUT"],
        health_check_interval=cfg["REDIS_HEALTH_CHECK_INTERVAL"],
    )
