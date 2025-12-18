import os
import sys
from datetime import datetime

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(".."))

project = "XVA / CCR / PFE Engine"
author = "Amine Badii"
year = datetime.now().year
copyright = f"{year}, {author}"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",           # Google / NumPy style docstrings
    "sphinx.ext.autodoc.typehints",  # type hints in docs
    "sphinx.ext.viewcode",
    "sphinxcontrib.plantuml",        # UML diagrams
]

# Autodoc settings
autodoc_member_order = "bysource"
autoclass_content = "class"
napoleon_google_docstring = False
napoleon_numpy_docstring = True

templates_path = ["_templates"]
exclude_patterns = ["_build"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# --- PlantUML configuration ---
# If plantuml is installed as a command-line jar:
# plantuml = "java -jar /path/to/plantuml.jar"

# If plantuml is in PATH as `plantuml`:
plantuml = "plantuml"

plantuml_output_format = "svg"

# Optional: if you keep UML snippets in separate files
# you can refer to them from .rst using .. uml:: /_uml/file.puml

