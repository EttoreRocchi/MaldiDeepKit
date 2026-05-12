Installation
============

.. code-block:: bash

   pip install maldideepkit

``maldiamrkit`` is a core dependency and is installed automatically -
MaldiDeepKit duck-types on the :class:`maldiamrkit.MaldiSet` data model
and reuses :class:`maldiamrkit.alignment.Warping` for leak-safe
spectral warping.

MaldiDeepKit requires Python 3.10 - 3.13 and pulls in PyTorch (≥ 2.0),
scikit-learn, einops, numpy, pandas, scipy, and matplotlib as its core
runtime dependencies.

Optional extras
---------------

The uncertainty-quantification estimators in
:mod:`maldideepkit.uncertainty` work out of the box for Monte Carlo
Dropout and split conformal prediction. The Laplace approximation
estimator additionally requires the optional ``laplace-torch``
dependency, available via the ``uncertainty`` extra:

.. code-block:: bash

   pip install "maldideepkit[uncertainty]"

GPU vs. CPU
-----------

Every classifier runs on CPU, which is what the project's continuous
integration tests against. All four architectures benefit
significantly from CUDA when training on ~6000-bin spectra; point a
classifier at a GPU by passing ``device="cuda"`` at construction, or
leave the default ``device="auto"`` to pick the best available.

Development Installation
------------------------

.. code-block:: bash

   git clone https://github.com/EttoreRocchi/MaldiDeepKit.git
   cd MaldiDeepKit
   pip install -e ".[dev]"
   pre-commit install

See :doc:`contributing` for coding conventions, testing, and PR
guidelines.

Documentation Build
-------------------

To build the documentation locally:

.. code-block:: bash

   pip install -e ".[docs]"
   make docs           # builds HTML into docs/build/html
   make docs-serve     # same, then serves on http://localhost:8080

The rendered site lands in ``docs/build/html/index.html``.
