# Configuration file for the Sphinx documentation builder.

import os
import sys
from datetime import datetime

# Add project src to sys.path so autodoc can import the package
sys.path.insert(0, os.path.abspath("../src"))

project = "option-pricing"
author = "Yang Ou"
current_year = datetime.now().year
copyright = f"{current_year}, {author}"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
]

autosummary_generate = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_static_path = ["_static"]

# Prevent Sphinx from failing on NumPy style param types in annotations
napoleon_google_docstring = False
napoleon_numpy_docstring = True

# Primary document
master_doc = "index"
