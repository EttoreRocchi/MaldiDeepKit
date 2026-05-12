Utils Module
============

Reproducibility helpers, loss functions, training primitives, and
diagnostic utilities shared across the classifier catalog. You
typically don't need to call most of these directly --
:class:`~maldideepkit.BaseSpectralClassifier` uses them internally --
but they are exposed for users building custom training loops or
investigating training dynamics.

Reproducibility
---------------

.. autofunction:: maldideepkit.utils.seed_everything

.. autofunction:: maldideepkit.utils.resolve_device

Training Primitives
-------------------

.. autoclass:: maldideepkit.utils.EarlyStopping
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: maldideepkit.utils.train_loop

Loss Functions
--------------

.. autoclass:: maldideepkit.utils.FocalLoss
   :members:
   :undoc-members:
   :show-inheritance:

Optimizers
----------

.. autoclass:: maldideepkit.utils.SAMOptimizer
   :members:
   :undoc-members:
   :show-inheritance:

Post-hoc Calibration
--------------------

Post-hoc, validation-set calibrators wired into
:class:`~maldideepkit.BaseSpectralClassifier` via
``tune_threshold`` / ``calibrate_temperature``. Also callable
directly.

.. autofunction:: maldideepkit.utils.tune_threshold

.. autofunction:: maldideepkit.utils.fit_temperature

Diagnostics
-----------

.. autofunction:: maldideepkit.utils.find_lr

Ensembling
----------

Mean-of-``predict_proba`` ensemble that fits each member independently
and averages their probability outputs at inference time.

.. autoclass:: maldideepkit.utils.SpectralEnsemble
   :members:
   :undoc-members:
   :show-inheritance:

Example
-------

.. code-block:: python

    from maldideepkit.utils import seed_everything, resolve_device

    seed_everything(42)
    device = resolve_device("auto")   # picks CUDA if available, else CPU

LR finder:

.. code-block:: python

    import numpy as np
    from maldideepkit import MaldiCNNClassifier
    from maldideepkit.utils import find_lr

    X = np.random.default_rng(0).standard_normal((256, 1000)).astype("float32")
    y = np.random.default_rng(0).integers(0, 2, size=256)
    out = find_lr(MaldiCNNClassifier(input_dim=1000, random_state=0), X, y, num_iter=200)
    print(out["suggested_lr"])
