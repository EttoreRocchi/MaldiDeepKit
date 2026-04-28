"""Sphinx configuration for MaldiDeepKit documentation."""

from __future__ import annotations

import importlib.util
import os
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath("../.."))

# Copy notebooks from repo root into the Sphinx source tree so that nbsphinx
# processes real files rather than following a symlink. Symlinked notebooks
# break image extraction on ReadTheDocs.
_here = Path(__file__).parent
_notebooks_src = _here.parent.parent / "notebooks"
_notebooks_dst = _here / "tutorials" / "notebooks"

if _notebooks_src.exists():
    if _notebooks_dst.is_symlink():
        _notebooks_dst.unlink()
    if _notebooks_dst.exists():
        shutil.rmtree(_notebooks_dst)
    shutil.copytree(_notebooks_src, _notebooks_dst)


def _read_version() -> str:
    init = Path(__file__).resolve().parents[2] / "maldideepkit" / "__init__.py"
    match = re.search(r'^__version__\s*=\s*"([^"]+)"', init.read_text(), re.MULTILINE)
    if not match:
        raise RuntimeError(f"Could not find __version__ in {init}.")
    return match.group(1)


def _available(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


project = "MaldiDeepKit"
copyright = "2026, Ettore Rocchi"
author = "Ettore Rocchi"

release = _read_version()
version = ".".join(release.split(".")[:2])


_required_extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
]
# Optional extensions: loaded when installed, silently skipped otherwise so
# docs still build in minimal environments without the full
# ``requirements-docs.txt`` installed.
_optional_extensions = [
    "sphinx_autodoc_typehints",
    "sphinx_design",
    "nbsphinx",
]

extensions = _required_extensions + [m for m in _optional_extensions if _available(m)]


napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = False
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True


autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

# Mock heavy runtime deps so autodoc can introspect modules even in the
# minimal docs environment (RTD or a local build without the full stack).
autodoc_mock_imports = [
    "torch",
    "einops",
    "sklearn",
    "numpy",
    "pandas",
    "scipy",
    "matplotlib",
    "maldiamrkit",
]

autosummary_generate = True

typehints_fully_qualified = False
always_document_param_types = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}


# Suppress warnings for shorthand types in NumPy-style docstrings that
# Sphinx cannot resolve.
nitpick_ignore_regex = [
    (r"py:class", r"optional"),
    (r"py:class", r"default=.*"),
    (r"py:class", r"array-like"),
    (r"py:class", r"np\..*"),
    (r"py:class", r"pd\..*"),
    (r"py:class", r"torch\..*"),
    (r"py:class", r"ndarray"),
    (r"py:class", r"Path"),
    (r"py:class", r"callable"),
    (r"py:class", r"self"),
    (r"py:class", r"typing\..*"),
    (r"py:class", r"MaldiSet"),
    (r"py:class", r"maldideepkit\..*"),
    (r"py:class", r"\d+"),
    (r"py:class", r"\{.*"),
    (r"py:class", r"\".*"),
]


exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


if _available("pydata_sphinx_theme"):
    html_theme = "pydata_sphinx_theme"
    html_logo = "_static/maldideepkit_logo.png"
    html_theme_options = {
        # Logo configuration
        "logo": {
            "text": "MaldiDeepKit",
            "image_light": "_static/maldideepkit_logo.png",
            "image_dark": "_static/maldideepkit_logo.png",
        },
        # Top navigation bar layout
        "navbar_start": ["navbar-logo"],
        "navbar_center": ["navbar-nav"],
        "navbar_end": ["theme-switcher", "navbar-icon-links"],
        "header_links_before_dropdown": 4,
        # Icon links (GitHub, PyPI)
        "icon_links": [
            {
                "name": "GitHub",
                "url": "https://github.com/EttoreRocchi/MaldiDeepKit",
                "icon": "fa-brands fa-github",
                "type": "fontawesome",
            },
            {
                "name": "PyPI",
                "url": "https://pypi.org/project/MaldiDeepKit/",
                "icon": "fa-brands fa-python",
                "type": "fontawesome",
            },
        ],
        # Sidebar behaviour
        "show_toc_level": 2,
        "navigation_depth": 3,
        "show_nav_level": 1,
        "collapse_navigation": True,
        # Footer
        "footer_start": ["copyright"],
        "footer_end": ["last-updated"],
        # Syntax highlighting
        "pygments_light_style": "default",
        "pygments_dark_style": "monokai",
    }
    html_css_files = ["css/custom.css"]
    # Sidebar configuration: pydata-specific templates, no sidebar on
    # the landing page.
    html_sidebars = {
        "**": ["sidebar-nav-bs"],
        "index": [],
    }
else:
    # Fallback for minimal local builds without the pydata theme installed.
    # Install the full docs stack with ``pip install -e ".[docs]"``.
    html_theme = "alabaster"
    html_css_files = []

html_static_path = ["_static"]

nbsphinx_execute = "never"
nbsphinx_allow_errors = True
