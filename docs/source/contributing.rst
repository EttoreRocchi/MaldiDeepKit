Contributing
============

Thanks for your interest in contributing! Here is how to get started.

Development setup
-----------------

.. code-block:: bash

   git clone https://github.com/EttoreRocchi/MaldiDeepKit.git
   cd MaldiDeepKit
   pip install -e ".[dev]"
   pre-commit install

Running tests
-------------

.. code-block:: bash

   make test            # full test suite
   make test-cov        # full test suite with a terminal coverage report

Please add tests for any new public API.

Linting
-------

.. code-block:: bash

   make lint            # ruff check --fix
   make format          # ruff format

Pre-commit hooks run ``ruff-check --fix``, ``ruff-format``,
``end-of-file-fixer``, and ``trailing-whitespace``. Make sure they are
installed (``pre-commit install``).

Style
-----

- NumPy-style docstrings for all public API.
- :class:`~maldideepkit.BaseSpectralClassifier` subclasses should
  override only ``_build_model()`` (the base class handles device
  placement, validation splits, early stopping, calibration, and
  persistence). Store fitted attributes with trailing underscores
  (``threshold_``, ``temperature_``, ``warper_``, ...).
- ``predict`` / ``predict_proba`` must be idempotent (no side effects
  outside ``fit``).
- Seed Python, NumPy, and PyTorch via ``random_state`` so identical
  configs produce identical weights.

Adding a new classifier
-----------------------

Every classifier lives in its own subpackage
(``maldideepkit/<family>/``) and consists of:

1. A :class:`torch.nn.Module` subclass implementing the architecture.
2. A classifier subclass of
   :class:`~maldideepkit.BaseSpectralClassifier` whose only required
   override is ``_build_model()`` returning the module.

Follow the existing modules as templates. Add the classifier to the
package ``__init__.py``, wire it into ``conftest.py``'s
``ALL_FACTORIES`` dict, and add per-model tests in
``tests/test_<family>.py``.

Submitting changes
------------------

1. Fork the repository and create a feature branch from ``main``.
2. Add tests for any new functionality.
3. Run ``make test`` and ``make lint`` (both must be clean).
4. Update ``CHANGELOG.md`` under the next version heading.
5. Open a pull request with a clear title and a body that motivates
   the design decision (not just what changed).

Reporting issues
----------------

Open an issue on `GitHub <https://github.com/EttoreRocchi/MaldiDeepKit/issues>`_
with a minimal reproducer:

- MaldiDeepKit version (``python -c "import maldideepkit; print(maldideepkit.__version__)"``)
- PyTorch, scikit-learn, and ``maldiamrkit`` versions
- Feature matrix shape, classifier and its parameters, and the
  exception or wrong result
