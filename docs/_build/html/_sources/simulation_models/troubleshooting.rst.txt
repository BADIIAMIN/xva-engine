Troubleshooting
===============

Sphinx build fails due to extensions
------------------------------------
- ``sphinx.ext.autodoc.typehints`` is not a valid Sphinx extension.
  Use ``sphinx_autodoc_typehints`` after installing ``sphinx-autodoc-typehints``.

PlantUML extension import error
-------------------------------
If you enable ``sphinxcontrib.plantuml`` you must install:

- ``pip install sphinxcontrib-plantuml``

Additionally, the PlantUML executable must be available if you set:

- ``plantuml = "plantuml"``

Module not found when running scripts
-------------------------------------
If you see ``ModuleNotFoundError`` for newly created packages:

- Ensure directories contain ``__init__.py``
- Ensure the file name matches the import path
- Ensure PyCharm marks the repo root as *Sources Root*
- Ensure your run configuration uses the correct working directory
