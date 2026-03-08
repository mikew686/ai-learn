"""Start the RQ worker. Set RQ_DEBUG=1 for debug logging. Handles SIGTERM/SIGINT for graceful shutdown."""

import logging
import signal
import sys

from rq import Queue, Worker
from rq.command import send_shutdown_command

from config import load_config
from utils.redis import get_redis_connection

logger = logging.getLogger(__name__)

worker = None
redis_conn = None


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


def main():
    global worker, redis_conn
    config = load_config()
    if config["RQ_DEBUG"]:
        logging.basicConfig(
            level=logging.DEBUG, format="%(levelname)s %(name)s %(message)s"
        )
    else:
        logging.basicConfig(
            level=logging.INFO, format="%(levelname)s %(name)s %(message)s"
        )

    _setup_signal_handlers()

    redis_conn = get_redis_connection()
    queue = Queue(connection=redis_conn)
    worker = Worker(queues=[queue], connection=redis_conn)

    try:
        logger.info("Starting RQ worker...")
        worker.work()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt; shutting down")
    except Exception as e:
        logger.exception("Worker error: %s", e)
        sys.exit(1)
    finally:
        logger.info("Worker shutdown completed")


if __name__ == "__main__":
    main()
