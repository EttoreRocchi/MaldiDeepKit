Semi-Supervised Module
======================

Iterative pseudo-labeling meta-classifier that wraps any sklearn
classifier (defaults to :class:`~maldideepkit.MaldiMLPClassifier`).
Rows of ``y`` equal to ``unlabeled_marker`` (default ``-1``) are
treated as unlabeled; the loop takes pseudo-labels from the base
estimator's ``predict`` (so a tuned decision threshold is respected),
accepts those whose predicted-class probability is at least
``confidence_threshold``, and refits the base estimator on the
expanded training set on each iteration.

Follows the :class:`sklearn.semi_supervised.SelfTrainingClassifier`
API and plugs into ``Pipeline``, ``GridSearchCV``, and
``cross_val_score`` like any other sklearn classifier.

MaldiSelfTrainingClassifier
---------------------------

.. autoclass:: maldideepkit.MaldiSelfTrainingClassifier
   :members:
   :undoc-members:
   :show-inheritance:

Example
-------

.. code-block:: python

   import numpy as np
   from maldideepkit import (
       MaldiMLPClassifier, MaldiSelfTrainingClassifier,
   )

   rng = np.random.default_rng(0)
   X = rng.standard_normal((100, 32)).astype("float32")
   y = rng.integers(0, 2, size=100)
   y[20:] = -1  # mark the trailing rows as unlabeled

   clf = MaldiSelfTrainingClassifier(
       base_estimator=MaldiMLPClassifier(
           epochs=10, batch_size=8, random_state=0,
       ),
       n_iter=3,
       confidence_threshold=0.9,
       random_state=0,
   ).fit(X, y)

   preds = clf.predict(X)
   # Iteration-level diagnostics:
   print(clf.history_["n_pseudolabels"])
   print(clf.history_["confidence_mean"])
   print(clf.history_["n_remaining_unlabeled"])
