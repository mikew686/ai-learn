"""About page blueprint."""

from flask import Blueprint, render_template

about_bp = Blueprint("about", __name__, url_prefix="/about")

REPO_BASE_URL = "https://github.com/mikew686/ai-learn/blob/main"


@about_bp.route("/")
def index():
    """About page: project information."""
    return render_template(
        "about/index.html",
        repo_readme_url=f"{REPO_BASE_URL}/README.md",
        repo_license_url=f"{REPO_BASE_URL}/LICENSE",
        repo_docs_license_url=f"{REPO_BASE_URL}/docs/license.md",
        repo_external_assets_url=f"{REPO_BASE_URL}/docs/external-assets.md",
    )
