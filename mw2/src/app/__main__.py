"""Run the Flask dev server when executing: python -m app"""

import os

from config import load_config

from . import create_app

if __name__ == "__main__":
    load_config()
    host = os.getenv("FLASK_HOST", "0.0.0.0")
    port = int(os.getenv("FLASK_PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "1").lower() in ("1", "true", "yes")
    create_app().run(host=host, port=port, debug=debug)
