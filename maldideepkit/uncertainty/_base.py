"""Abstract base class shared by every uncertainty estimator."""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Any

import numpy as np
from sklearn.utils.validation import check_is_fitted

from ..base.classifier import BaseSpectralClassifier
from ..base.data import _to_numpy
from ._result import UncertaintyResult


def _apply_classifier_preprocessing(
    classifier: BaseSpectralClassifier, X: Any
) -> np.ndarray:
    """Apply the classifier's full inference-time preprocessing to ``X``.

    Mirrors the preprocessing performed inside
    :meth:`BaseSpectralClassifier._forward_logits` exactly: optional
    spectral warping, the fitted ``input_transform`` state, and the
    legacy ``standardize`` path used when no modern input transform is
    set. Replicating it here lets uncertainty estimators feed data
    through ``classifier.model_`` directly without going through the
    eval-only ``_forward_logits`` shortcut.

    Parameters
    ----------
    classifier : BaseSpectralClassifier
        Fitted classifier whose preprocessing artefacts will be applied.
    X : array-like or MaldiSet of shape (n_samples, n_bins)
        Raw input spectra.

    Returns
    -------
    ndarray of shape (n_samples, n_bins)
        Preprocessed feature matrix as ``float32``.
    """
    X_np = _to_numpy(X)
    classifier._check_input_dim(X_np)
    if getattr(classifier, "warper_", None) is not None:
        from ..base.data import _warp_numpy

        X_np = _warp_numpy(classifier.warper_, X_np)
    state = getattr(classifier, "input_transform_state_", None)
    if state is not None and state.get("mode", "none") != "none":
        from ..base.data import apply_input_transform

        X_np = apply_input_transform(X_np, state)
    elif (
        classifier.standardize
        and getattr(classifier, "feature_mean_", None) is not None
    ):
        from ..base.data import _STD_FLOOR

        safe_std = np.maximum(classifier.feature_std_, _STD_FLOOR).astype(np.float32)
        X_np = (X_np - classifier.feature_mean_) / safe_std
    return X_np.astype(np.float32, copy=False)


class BaseUncertaintyEstimator(metaclass=ABCMeta):
    """Abstract base for every estimator in :mod:`maldideepkit.uncertainty`.

    Parameters
    ----------
    classifier : BaseSpectralClassifier
        A *fitted* MaldiDeepKit classifier. Its preprocessing pipeline
        (warping, input transform, legacy standardisation) is reused so
        the uncertainty estimate is produced on inputs identical to
        those seen at training time.

    Raises
    ------
    sklearn.exceptions.NotFittedError
        If ``classifier`` has not been fitted.
    """

    def __init__(self, classifier: BaseSpectralClassifier) -> None:
        check_is_fitted(classifier, "model_")
        self.classifier = classifier

    @abstractmethod
    def predict_with_uncertainty(self, X: Any) -> UncertaintyResult:
        """Return predictions and per-sample uncertainty for ``X``.

        Parameters
        ----------
        X : array-like or MaldiSet of shape (n_samples, n_bins)
            Spectra to score. Must match ``classifier.input_dim_``.

        Returns
        -------
        UncertaintyResult
            Predictions, probability mean, scalar uncertainty, optional
            epistemic / aleatoric decomposition, and method-specific
            metadata.
        """

    def __repr__(self) -> str:
        return f"{type(self).__name__}(classifier={type(self.classifier).__name__})"


__all__ = ["BaseUncertaintyEstimator"]
