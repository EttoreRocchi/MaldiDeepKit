Quickstart Guide
================

This guide walks through the core workflows for fitting, inspecting, and
persisting MaldiDeepKit classifiers on binned MALDI-TOF spectra.

Fitting a Classifier
--------------------

Every MaldiDeepKit classifier exposes the standard scikit-learn estimator
API - ``fit``, ``predict``, ``predict_proba``, and ``score``:

.. code-block:: python

   import numpy as np
   from maldideepkit import MaldiMLPClassifier

   rng = np.random.default_rng(0)
   X = rng.standard_normal((200, 6000)).astype("float32")
   y = rng.integers(0, 2, size=200)

   clf = MaldiMLPClassifier(random_state=0)
   clf.fit(X, y)

   proba = clf.predict_proba(X)
   preds = clf.predict(X)
   acc = clf.score(X, y)

Switching Architectures
-----------------------

The four classifiers share the same base API, so swapping one out is a
one-line change:

.. code-block:: python

   from maldideepkit import (
       MaldiMLPClassifier,
       MaldiCNNClassifier,
       MaldiResNetClassifier,
       MaldiTransformerClassifier,
   )

   classifiers = {
       "mlp":         MaldiMLPClassifier(random_state=0),
       "cnn":         MaldiCNNClassifier(random_state=0),
       "resnet":      MaldiResNetClassifier(random_state=0),
       "transformer": MaldiTransformerClassifier(random_state=0),
   }

   for name, clf in classifiers.items():
       clf.fit(X, y)
       print(f"{name}: {clf.score(X, y):.3f}")

Inspecting Attention Weights
----------------------------

:class:`~maldideepkit.MaldiMLPClassifier` has a sigmoid-gated attention
layer enabled by default. After fitting, the last forward pass is cached
on ``attention_weights_`` and :meth:`get_attention_weights` recomputes
them for arbitrary inputs:

.. code-block:: python

   clf = MaldiMLPClassifier(random_state=0).fit(X, y)

   cached = clf.attention_weights_           # (len(X_last_forward), hidden_dim)
   weights = clf.get_attention_weights(X[:10])  # (10, hidden_dim)

Set ``use_attention=False`` to fall back to a plain MLP of the same
depth.

Integration with MaldiAMRKit
----------------------------

Any object with a DataFrame-like ``.X`` attribute - notably
:class:`maldiamrkit.MaldiSet` - is accepted directly:

.. code-block:: python

   from maldiamrkit import MaldiSet
   from maldideepkit import MaldiCNNClassifier

   ds = MaldiSet.from_directory(
       "spectra/", "metadata.csv",
       aggregate_by={"antibiotics": "Ciprofloxacin"},
   )
   clf = MaldiCNNClassifier(random_state=0).fit(ds, ds.y.squeeze())
   preds = clf.predict(ds)

Using sklearn Pipelines
-----------------------

MaldiDeepKit classifiers behave like any other scikit-learn estimator
and compose inside pipelines and cross-validators:

.. code-block:: python

   from sklearn.model_selection import StratifiedKFold, cross_val_score
   from sklearn.pipeline import Pipeline
   from sklearn.preprocessing import StandardScaler
   from maldideepkit import MaldiMLPClassifier

   pipe = Pipeline([
       ("scaler", StandardScaler()),
       ("clf", MaldiMLPClassifier(random_state=0)),
   ])

   cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
   scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
   print(f"CV accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

Class Imbalance
---------------

Three orthogonal knobs, combinable:

.. code-block:: python

   # 1. Loss weighting (balanced or explicit)
   clf = MaldiCNNClassifier(class_weight="balanced", random_state=0).fit(X, y)
   clf = MaldiCNNClassifier(class_weight=[1.0, 3.0], random_state=0).fit(X, y)

   # 2. Focal loss (down-weights easy examples; pairs with class_weight)
   clf = MaldiCNNClassifier(
       loss="focal", focal_gamma=2.0, class_weight="balanced", random_state=0,
   ).fit(X, y)

   # 3. Post-hoc threshold tuning on the validation split (binary only).
   # Sweeps thresholds and stores the one that maximises balanced accuracy,
   # F1, or Youden's J. `predict()` uses it instead of argmax @ 0.5.
   clf = MaldiCNNClassifier(
       tune_threshold=True,
       threshold_metric="balanced_accuracy",   # or "f1", "youden"
       random_state=0,
   ).fit(X, y)
   print(clf.threshold_)

Training Recipe and LR Schedule
-------------------------------

Every classifier uses linear warmup + cosine-annealing LR decay. The
deep models (:class:`~maldideepkit.MaldiResNetClassifier`,
:class:`~maldideepkit.MaldiTransformerClassifier`) ship with the
training recipe that keeps them stable out of the box:
``weight_decay`` > 0 (engages ``AdamW``), ``grad_clip_norm=1.0``, a
short ``warmup_epochs``, and for the Transformer additionally
``drop_path_rate=0.1``, ``attention_dropout=0.0``, ``layerscale_init=1e-4``,
and ``learning_rate=3e-4``. Override any of these at construction time:

.. code-block:: python

   clf = MaldiTransformerClassifier(
       learning_rate=2e-4,
       weight_decay=0.1,
       warmup_epochs=10,
       drop_path_rate=0.2,
       random_state=0,
   ).fit(X, y)

The MLP and CNN baselines keep the lean defaults (Adam, no clipping,
no warmup).

Mixed Precision
---------------

Enable ``use_amp=True`` to train with :func:`torch.autocast` +
:class:`torch.amp.GradScaler` on CUDA. ~2× speedup on recent NVIDIA
GPUs; a no-op on CPU.

.. code-block:: python

   clf = MaldiTransformerClassifier(use_amp=True, random_state=0).fit(X, y)

Stochastic Weight Averaging
---------------------------

Set ``swa_start_epoch`` to maintain a
:class:`torch.optim.swa_utils.AveragedModel` from that epoch onward
and use the SWA-averaged weights at prediction time.

.. code-block:: python

   clf = MaldiCNNClassifier(epochs=80, swa_start_epoch=60, random_state=0).fit(X, y)

Probability Calibration
-----------------------

``calibrate_temperature=True`` fits a scalar temperature on the
validation-split logits by LBFGS (Guo et al. 2017). Applied in
:meth:`predict_proba` without changing argmax order.

.. code-block:: python

   clf = MaldiCNNClassifier(calibrate_temperature=True, random_state=0).fit(X, y)
   print(clf.temperature_)

Spectral Warping (pre-scaling)
------------------------------

Pass any sklearn-style transformer with ``fit(X) / transform(X)`` --
typically :class:`maldiamrkit.alignment.Warping` - via ``warping=``.
It is fitted on the **training split only** (so no leakage from the
validation fold) and applied to both splits, *before* per-feature
standardization. At :meth:`predict` time, incoming spectra are
transformed by the fitted warper first.

.. code-block:: python

   from maldiamrkit.alignment import Warping
   from maldideepkit import MaldiCNNClassifier

   clf = MaldiCNNClassifier(
       warping=Warping(method="shift", n_jobs=-1),
       standardize=True,
       random_state=0,
   ).fit(X, y)

The fitted transformer is stored as ``clf.warper_`` and persisted as a
sibling joblib pickle (``<path>.warper.pkl``) by :meth:`save`.

Finding a Learning Rate
-----------------------

:func:`~maldideepkit.utils.find_lr` sweeps the LR geometrically over a
short training run and returns the curve plus a steepest-descent
suggestion:

.. code-block:: python

   from maldideepkit.utils import find_lr
   out = find_lr(MaldiCNNClassifier(random_state=0), X, y, num_iter=200)
   print(out["suggested_lr"])

Sharpness-Aware Minimization (SAM)
----------------------------------

``use_sam=True`` wraps the base optimizer in
:class:`~maldideepkit.utils.SAMOptimizer`. Two forward/backward passes
per step (~2× compute); typically helps generalization on small
datasets.

.. code-block:: python

   clf = MaldiCNNClassifier(use_sam=True, sam_rho=0.05, random_state=0).fit(X, y)

Device Placement and Reproducibility
------------------------------------

``device="auto"`` (the default) picks CUDA when available and CPU
otherwise. Set ``random_state`` to seed Python, NumPy, and PyTorch in
one call; the same seed produces identical weights and predictions:

.. code-block:: python

   clf = MaldiTransformerClassifier(device="cuda", random_state=42).fit(X, y)

Persistence
-----------

:meth:`save` writes a state-dict ``.pt`` plus a hyperparameter ``.json``
pair; :meth:`load` restores both. Attempting :meth:`predict` with a
different number of bins from the training matrix raises a clear
``ValueError``:

.. code-block:: python

   clf.save("my_model")
   # -> my_model.pt, my_model.json

   from maldideepkit import MaldiMLPClassifier, BaseSpectralClassifier

   restored = MaldiMLPClassifier.load("my_model")
   # or infer the class from the JSON:
   restored = BaseSpectralClassifier.load("my_model")

Early Stopping and Validation
-----------------------------

A stratified internal validation split (controlled by ``val_fraction``)
is carved out of every :meth:`fit` call and used for early stopping
(``early_stopping_patience`` epochs without improvement) and learning-
rate scheduling. Set ``verbose=True`` to watch the validation loss:

.. code-block:: python

   clf = MaldiCNNClassifier(
       epochs=100,
       val_fraction=0.1,
       early_stopping_patience=10,
       verbose=True,
       random_state=0,
   ).fit(X, y)
