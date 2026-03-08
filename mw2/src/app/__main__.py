"""Run the Flask dev server when executing: python -m app"""

from config import load_config

from . import create_app

if __name__ == "__main__":
    config = load_config()
    create_app().run(host="0.0.0.0", port=5000, debug=config["DEBUG"])
