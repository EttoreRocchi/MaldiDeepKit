"""Abstract base class shared by every uncertainty estimator."""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Any

import numpy as np
from sklearn.utils.validation import check_is_fitted

from ..base.classifier import BaseSpectralClassifier
from ._result import UncertaintyResult


def _apply_classifier_preprocessing(
    classifier: BaseSpectralClassifier, X: Any
) -> np.ndarray:
    """Apply the classifier's full inference-time preprocessing to ``X``.

    Delegates to :meth:`~maldideepkit.BaseSpectralClassifier._preprocess_input`, the
    single source of truth for inference preprocessing (optional
    spectral warping, the fitted ``input_transform`` state, and the
    legacy ``standardize`` path), so uncertainty estimators feed data
    through ``classifier.model_`` on inputs identical to what
    ``predict_proba`` sees (including dual-input models, whose
    preprocessing is confined to the spectrum columns).

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
    return classifier._preprocess_input(X)


class BaseUncertaintyEstimator(metaclass=ABCMeta):
    """Abstract base for every estimator in :mod:`maldideepkit.uncertainty`.

    Parameters
    ----------
    classifier : BaseSpectralClassifier
        A *fitted* MaldiDeepKit classifier. Its preprocessing pipeline
        (warping, input transform, legacy standardization) is reused so
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
