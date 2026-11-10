Utils module
============

Reproducibility helpers, loss functions, training primitives, and
diagnostic utilities shared across the classifier catalogue.
:class:`~maldideepkit.BaseSpectralClassifier` uses most of them
internally, so you rarely call them directly; they are exposed for
custom training loops and for investigating training dynamics.

Reproducibility
---------------

.. autofunction:: maldideepkit.utils.seed_everything

.. autofunction:: maldideepkit.utils.resolve_device

Training primitives
-------------------

.. autoclass:: maldideepkit.utils.Callback
   :members:
   :inherited-members:
   :show-inheritance:

.. autoclass:: maldideepkit.utils.EarlyStopping
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: maldideepkit.utils.train_loop

Loss functions
--------------

.. autoclass:: maldideepkit.utils.FocalLoss
   :members:
   :undoc-members:
   :show-inheritance:

Optimisers
----------

.. autoclass:: maldideepkit.utils.SAMOptimizer
   :members:
   :undoc-members:
   :show-inheritance:

Post-hoc calibration
--------------------

Post-hoc calibrators fitted on the validation split, used by
:class:`~maldideepkit.BaseSpectralClassifier` through
``tune_threshold`` / ``calibrate_temperature``, and also callable
directly.

.. autofunction:: maldideepkit.utils.tune_threshold

.. autofunction:: maldideepkit.utils.fit_temperature

Diagnostics
-----------

.. autofunction:: maldideepkit.utils.find_lr

Ensembling
----------

Mean-of-``predict_proba`` ensemble and scikit-learn estimator. ``fit``
trains unfitted members (instances or registered names) independently;
an ensemble of fitted members predicts without ``fit``, and raises
:class:`ValueError` when their ``classes_`` differ. ``save`` writes
every member plus an index file, and writes nothing when a member
fails.

.. autoclass:: maldideepkit.utils.SpectralEnsemble
   :members:
   :undoc-members:
   :show-inheritance:

Examples
--------

.. code-block:: python

   from maldideepkit.utils import seed_everything, resolve_device

   seed_everything(42)
   device = resolve_device("auto")   # picks the GPU if available, else the CPU

LR finder:

.. code-block:: python

   import numpy as np
   from maldideepkit import MaldiCNNClassifier
   from maldideepkit.utils import find_lr

   X = np.random.default_rng(0).standard_normal((256, 1000)).astype("float32")
   y = np.random.default_rng(0).integers(0, 2, size=256)
   out = find_lr(MaldiCNNClassifier(input_dim=1000, random_state=0), X, y, num_iter=200)
   print(out["suggested_lr"])
