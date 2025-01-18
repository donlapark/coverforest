# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
from importlib.metadata import version as get_version

project = "coverforest"
copyright = "2025, D. Ponnoprat"
author = "D. Ponnoprat"
release = get_version("coverforest")
version = ".".join(release.split(".")[:3])

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinxcontrib.bibtex",
    "sphinx_copybutton",
    "myst_nb",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "_templates", "Thumbs.db", ".DS_Store"]

# The reST default role (used for this markup: `text`) to use for all
# documents.
default_role = "literal"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
# html_style = "css/coverforest.css"
html_logo = "_static/img/coverforest_96.png"
html_favicon = "_static/img/coverforest.ico"

html_theme_options = {
    "repository_url": "https://github.com/donlapark/coverforest",
    "repository_branch": "main",
    "path_to_docs": "docs/",
    "use_repository_button": True,
    "use_fullscreen_button": False,
    "pygments_light_style": "tango",
    "pygments_dark_style": "monokai",
    "launch_buttons": {
        "colab_url": "https://colab.research.google.com",
        "binderhub_url": "https://mybinder.org",
        "notebook_interface": "jupyterlab",
    },
}

master_doc = "index"

todo_include_todos = False
autosummary_generate = True
bibtex_bibfiles = ["refs.bib"]

# -- Options for numpydoc -----------------------------------------------------

# this is needed for some reason...
# see https://github.com/numpy/numpydoc/issues/69
numpydoc_show_class_members = False

# -- Options for intersphinx --------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/{.major}".format(sys.version_info), None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference", None),
    "scikit-learn": ("https://scikit-learn.org/stable", None),
    "matplotlib": ("https://matplotlib.org/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "joblib": ("https://joblib.readthedocs.io/en/latest/", None),
}

# -- Options for myst-nb --------------------------------------------------
myst_heading_anchors = 2
nb_execution_mode = "off"
nb_mime_priority_overrides = [("spelling", "text/plain", 0)]
myst_enable_extensions = [
    "colon_fence",
    "amsmath",
    "dollarmath",
]

# -- Options for mathjax --------------------------------------------------
mathjax3_config = {
    "tex": {
        "macros": {
            "min": r"\mathrm{min}",
            "argmin": r"\mathrm{argmin}",
        }
    }
}
