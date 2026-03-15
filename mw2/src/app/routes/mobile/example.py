"""Mobile example blueprint: /mobile/example and /mobile/example/subpage."""

from flask import Blueprint, render_template

mobile_example_bp = Blueprint("mobile_example", __name__, url_prefix="/mobile/example")


@mobile_example_bp.route("/")
def index():
    """Mobile example home."""
    return render_template("mobile/example/index.html")


@mobile_example_bp.route("/subpage")
def subpage():
    """Mobile example subpage (back to home)."""
    return render_template("mobile/example/subpage.html")


@mobile_example_bp.route("/subpage/subpage")
def subpage_subpage():
    """Mobile example nested subpage (back to subpage)."""
    return render_template("mobile/example/subpage_subpage.html")
