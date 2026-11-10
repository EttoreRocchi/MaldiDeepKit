Installation
============

.. code-block:: bash

   pip install maldideepkit

``maldiamrkit`` is a core dependency and is installed automatically
(MaldiDeepKit duck-types on the :class:`maldiamrkit.MaldiSet` data
model and reuses :class:`maldiamrkit.alignment.Warping` for leak-safe
spectral warping).

MaldiDeepKit requires Python 3.10 or newer, is tested on 3.10 to 3.13,
and pulls in PyTorch (≥ 2.3),
scikit-learn, einops, NumPy, pandas, SciPy, and Matplotlib as its core
runtime dependencies.

Optional extras
---------------

Of the uncertainty-quantification estimators in
:doc:`maldideepkit.uncertainty <api/uncertainty>`, Monte Carlo Dropout
and split conformal prediction need no extra. The Laplace approximation
estimator also requires the optional ``laplace-torch`` dependency,
available through the ``uncertainty`` extra:

.. code-block:: bash

   pip install "maldideepkit[uncertainty]"

The :func:`~maldideepkit.smiles_to_fingerprint` helper used by
:class:`~maldideepkit.MaldiDualBranchClassifier` workflows requires
RDKit, available through the ``rdkit`` extra:

.. code-block:: bash

   pip install "maldideepkit[rdkit]"

GPU vs CPU
----------

With the default ``device="auto"``, every classifier trains on a CUDA
GPU when one is available and on CPU otherwise. Pass ``device="cuda"``
or ``device="cpu"`` at construction to force one of the two. The
project's continuous integration runs the test suite on CPU.

Development installation
------------------------

.. code-block:: bash

   git clone https://github.com/EttoreRocchi/MaldiDeepKit.git
   cd MaldiDeepKit
   pip install -e ".[dev]"
   pre-commit install

See :doc:`contributing` for coding conventions, testing, and PR
guidelines.

Documentation build
-------------------

To build the documentation locally:

.. code-block:: bash

   pip install -e ".[docs]"
   make docs           # builds HTML into docs/build/html
   make docs-serve     # serves the built HTML on http://localhost:8080

The rendered site lands in ``docs/build/html/index.html``.
