Uncertainty Module
==================

Drop-in uncertainty-quantification estimators for fitted MaldiDeepKit
classifiers. Three methods share a single
:meth:`~maldideepkit.uncertainty.BaseUncertaintyEstimator.predict_with_uncertainty`
interface and return a common
:class:`~maldideepkit.uncertainty.UncertaintyResult` dataclass, so
downstream code can swap methods without touching call sites.

- :class:`~maldideepkit.uncertainty.MCDropoutEstimator` - Monte Carlo
  Dropout (Gal and Ghahramani, 2016) with epistemic / aleatoric
  decomposition.
- :class:`~maldideepkit.uncertainty.LaplaceEstimator` - last-layer or
  full-network Laplace approximation via the optional ``laplace-torch``
  dependency.
- :class:`~maldideepkit.uncertainty.ConformalPredictor` - split
  conformal prediction with the LAC non-conformity score.

The Laplace estimator requires the ``uncertainty`` extra; install it
with ``pip install "maldideepkit[uncertainty]"`` (see
:doc:`../installation`). Monte Carlo Dropout and split conformal
prediction need no extras.

UncertaintyResult
-----------------

.. autoclass:: maldideepkit.uncertainty.UncertaintyResult
   :members:
   :undoc-members:
   :show-inheritance:

BaseUncertaintyEstimator
------------------------

.. autoclass:: maldideepkit.uncertainty.BaseUncertaintyEstimator
   :members:
   :undoc-members:
   :show-inheritance:

MCDropoutEstimator
------------------

.. autoclass:: maldideepkit.uncertainty.MCDropoutEstimator
   :members:
   :undoc-members:
   :show-inheritance:

LaplaceEstimator
----------------

.. autoclass:: maldideepkit.uncertainty.LaplaceEstimator
   :members:
   :undoc-members:
   :show-inheritance:

ConformalPredictor
------------------

.. autoclass:: maldideepkit.uncertainty.ConformalPredictor
   :members:
   :undoc-members:
   :show-inheritance:

Examples
--------

Monte Carlo Dropout on a fitted attention-MLP:

.. code-block:: python

    import numpy as np
    from maldideepkit import MaldiMLPClassifier
    from maldideepkit.uncertainty import MCDropoutEstimator

    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 6000)).astype("float32")
    y = rng.integers(0, 2, size=200)

    clf = MaldiMLPClassifier(random_state=0).fit(X, y)
    est = MCDropoutEstimator(clf, n_samples=30)
    result = est.predict_with_uncertainty(X[:10])
    # result.predictions, result.proba_mean, result.uncertainty,
    # result.epistemic, result.aleatoric

Split conformal prediction with calibration:

.. code-block:: python

    from sklearn.model_selection import train_test_split
    from maldideepkit.uncertainty import ConformalPredictor

    X_tr, X_cal, y_tr, y_cal = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=0,
    )
    clf = MaldiMLPClassifier(random_state=0).fit(X_tr, y_tr)

    cp = ConformalPredictor(clf, alpha=0.1).calibrate(X_cal, y_cal)
    result = cp.predict_with_uncertainty(X[:10])
    sets = result.metadata["prediction_sets"]   # (10, n_classes) bool

Laplace approximation (requires the optional ``laplace-torch``
dependency):

.. code-block:: python

    from maldideepkit.uncertainty import LaplaceEstimator

    la = LaplaceEstimator(clf, subset_of_weights="last_layer",
                          hessian_structure="diag")
    la.calibrate(X_cal, y_cal)
    result = la.predict_with_uncertainty(X[:10])
