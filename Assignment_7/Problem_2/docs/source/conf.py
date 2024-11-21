# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys

sys.path.insert(0, os.path.abspath("../../src"))

project = "Radial Schrödinger Equation Solver"
copyright = "2024, Gilbert Young"
author = "Gilbert Young"
release = "1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "numpydoc",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
]

templates_path = ["_templates"]
exclude_patterns = []

language = "zh_CN"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# Autodoc settings
autosummary_generate = True
autosummary_generate_dir = "generated"
if not os.path.exists("_templates"):
    os.makedirs("_templates")
if not os.path.exists("_templates/autosummary"):
    os.makedirs("_templates/autosummary")

# Theme settings
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
