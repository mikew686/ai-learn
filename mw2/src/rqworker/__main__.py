"""Start the RQ worker.

Use RQ_LOGGING_LEVEL (default INFO), RQ_LOG_FORMAT, RQ_DATE_FORMAT for logging.
Handles SIGTERM/SIGINT/SIGHUP for graceful shutdown.

Supports modern RQ built-in scheduling via worker.work(with_scheduler=True).
Only applies worker/worker-loop options when explicitly overridden by env.
"""

import logging
import os
import signal
import sys
from typing import Any

from rq import Queue, Worker
from rq.command import send_shutdown_command

from config import load_config
from utils.redis import get_redis_connection

logger = logging.getLogger(__name__)

worker = None
redis_conn = None


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


def _signal_name(signum: int) -> str:
    try:
        return signal.Signals(signum).name
    except (ValueError, AttributeError):
        return str(signum)


def _signal_handler(signum, frame):
    """Handle shutdown signals: set worker stop flag and send RQ shutdown command."""
    global worker, redis_conn
    name = _signal_name(signum)
    logger.warning(
        "Received signal %s (%s). Initiating graceful shutdown...", name, signum
    )
    if worker:
        try:
            worker._stop_requested = True
            if redis_conn:
                send_shutdown_command(redis_conn, worker.name)
        except Exception as e:
            logger.error("Error during shutdown: %s", e)
    sys.exit(0)


def _setup_signal_handlers():
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGHUP, _signal_handler)
    logger.info("Signal handlers configured (SIGTERM, SIGINT, SIGHUP)")


def _is_set(value: Any) -> bool:
    return value is not None


def _add_if_set(target: dict[str, Any], key: str, value: Any):
    if _is_set(value):
        target[key] = value


def _build_queue(connection) -> Queue:
    queue_kwargs: dict[str, Any] = {"connection": connection}

    _add_if_set(
        queue_kwargs, "default_timeout", _parse_int(os.getenv("RQ_DEFAULT_TIMEOUT"))
    )
    _add_if_set(queue_kwargs, "is_async", _parse_bool(os.getenv("RQ_IS_ASYNC")))
    _add_if_set(queue_kwargs, "serializer", os.getenv("RQ_SERIALIZER"))

    queue_name = os.getenv("RQ_QUEUE_NAME")
    if queue_name:
        return Queue(name=queue_name, **queue_kwargs)

    return Queue(**queue_kwargs)


def _build_worker(queue: Queue, connection) -> Worker:
    worker_kwargs: dict[str, Any] = {"connection": connection}

    _add_if_set(worker_kwargs, "name", os.getenv("RQ_WORKER_NAME"))
    _add_if_set(
        worker_kwargs, "default_result_ttl", _parse_int(os.getenv("RQ_RESULTS_TTL"))
    )
    _add_if_set(worker_kwargs, "job_class", os.getenv("RQ_JOB_CLASS"))
    _add_if_set(worker_kwargs, "queue_class", os.getenv("RQ_QUEUE_CLASS"))
    _add_if_set(worker_kwargs, "serializer", os.getenv("RQ_SERIALIZER"))
    _add_if_set(
        worker_kwargs,
        "maintenance_interval",
        _parse_int(os.getenv("RQ_MAINTENANCE_INTERVAL")),
    )

    return Worker(queues=[queue], **worker_kwargs)


def _build_work_kwargs() -> dict[str, Any]:
    work_kwargs: dict[str, Any] = {}

    _add_if_set(work_kwargs, "burst", _parse_bool(os.getenv("RQ_BURST")))
    rq_disable_scheduler = _parse_bool(os.getenv("RQ_DISABLE_SCHEDULER"))
    _add_if_set(work_kwargs, "with_scheduler", False if rq_disable_scheduler else True)
    _add_if_set(work_kwargs, "logging_level", os.getenv("RQ_LOGGING_LEVEL", "INFO"))
    _add_if_set(work_kwargs, "date_format", os.getenv("RQ_DATE_FORMAT", "%H:%M:%S"))
    _add_if_set(
        work_kwargs,
        "log_format",
        os.getenv("RQ_LOG_FORMAT", "%(asctime)s %(levelname)s %(message)s"),
    )
    _add_if_set(work_kwargs, "max_jobs", _parse_int(os.getenv("RQ_MAX_JOBS")))
    _add_if_set(work_kwargs, "max_idle_time", _parse_int(os.getenv("RQ_MAX_IDLE_TIME")))

    # Supported dequeue strategies in RQ docs: default, random, round_robin
    _add_if_set(work_kwargs, "dequeue_strategy", os.getenv("RQ_DEQUEUE_STRATEGY"))

    return work_kwargs


def main():
    global worker, redis_conn

    load_config()

    level_name = os.getenv("RQ_LOGGING_LEVEL", "INFO")
    level = getattr(logging, level_name.upper(), logging.INFO)
    log_format = os.getenv("RQ_LOG_FORMAT", "%(asctime)s %(levelname)s %(message)s")
    date_format = os.getenv("RQ_DATE_FORMAT", "%H:%M:%S")
    logging.basicConfig(level=level, format=log_format, datefmt=date_format)

    _setup_signal_handlers()

    redis_conn = get_redis_connection()
    queue = _build_queue(redis_conn)
    worker = _build_worker(queue, redis_conn)
    work_kwargs = _build_work_kwargs()

    try:
        logger.info(
            "Starting RQ worker name=%s queue=%s with options=%s",
            worker.name,
            queue.name,
            work_kwargs,
        )
        worker.work(**work_kwargs)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt; shutting down")
    except Exception as e:
        logger.exception("Worker error: %s", e)
        sys.exit(1)
    finally:
        logger.info("Worker shutdown completed")


if __name__ == "__main__":
    main()
