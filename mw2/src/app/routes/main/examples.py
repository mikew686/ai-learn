"""Examples blueprint."""

from flask import Blueprint, render_template

examples_bp = Blueprint("examples", __name__, url_prefix="/examples")


@examples_bp.route("/")
def index():
    """Examples placeholder: coming soon."""
    return render_template("main/examples/index.html")
