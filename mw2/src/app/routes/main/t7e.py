"""T7e (translation example) blueprint."""

from flask import Blueprint, render_template

t7e_bp = Blueprint("t7e", __name__, url_prefix="/t7e")


@t7e_bp.route("/")
def index():
    """Translation example placeholder: coming soon."""
    return render_template("main/t7e/index.html")
