"""Sphinx configuration for the project documentation."""

project = "nuxnet-training"
author = "nuxnet-training contributors"

extensions = ["myst_parser"]
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
root_doc = "index"
myst_heading_anchors = 6

html_theme = "sphinx_rtd_theme"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
