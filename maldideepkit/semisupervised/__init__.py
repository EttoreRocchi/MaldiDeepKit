"""Semi-supervised learning wrappers around a MaldiDeepKit backbone.

Current contents:

- :class:`MaldiSelfTrainingClassifier`: iterative pseudo-labeling
  on a labeled seed plus an unlabeled pool (rows marked with a
  configurable ``unlabeled_marker``, default ``-1``). Wraps any
  scikit-learn classifier (typically a
  :class:`~maldideepkit.BaseSpectralClassifier`); follows the
  :class:`sklearn.semi_supervised.SelfTrainingClassifier` API.
"""

from .self_training import MaldiSelfTrainingClassifier

__all__ = ["MaldiSelfTrainingClassifier"]
