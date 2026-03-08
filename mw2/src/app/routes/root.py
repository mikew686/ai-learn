"""Root (main) page blueprint."""

from flask import Blueprint, render_template

root_bp = Blueprint("root", __name__)


@root_bp.route("/")
def index():
    """Main landing page: mwsquared AI engineering learning site."""
    return render_template("root/index.html")
