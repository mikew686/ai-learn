"""Start the RQ worker. Set RQ_DEBUG=1 for debug logging."""
import logging
import sys

from config import load_config
from rq import Queue, Worker
from utils.redis import get_redis_connection


def main():
    config = load_config()
    if config["RQ_DEBUG"]:
        logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(name)s %(message)s")
    else:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

    conn = get_redis_connection()
    queue = Queue(connection=conn)
    worker = Worker(queues=[queue], connection=conn)
    worker.work()


if __name__ == "__main__":
    main()
    sys.exit(0)
