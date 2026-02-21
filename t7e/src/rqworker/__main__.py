"""Start the RQ worker. Use RQ_DEBUG=1 or FLASK_DEBUG=1 for debug logging."""
import logging
import os
import sys

from redis import Redis

from config import load_config
from rq import Queue, Worker


def _debug_logging():
    """Enable debug logging if RQ_DEBUG or FLASK_DEBUG is set."""
    rq_debug = os.environ.get("RQ_DEBUG", "").lower() in ("1", "true", "yes")
    flask_debug = os.environ.get("FLASK_DEBUG", "").lower() in ("1", "true", "yes")
    return rq_debug or flask_debug


def main():
    config = load_config()
    redis_url = config["REDIS_URL"]

    if _debug_logging():
        logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(name)s %(message)s")
    else:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

    conn = Redis.from_url(redis_url)
    queue = Queue(connection=conn)
    worker = Worker(queues=[queue], connection=conn)
    worker.work()


if __name__ == "__main__":
    main()
    sys.exit(0)
