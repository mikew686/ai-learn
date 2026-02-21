"""Run the Flask dev server when executing: python -m app"""
from . import create_app

if __name__ == "__main__":
    create_app().run(host="0.0.0.0", port=5000, debug=True)
