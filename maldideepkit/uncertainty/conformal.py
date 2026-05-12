"""Split conformal prediction estimator.

Implements the LAC (Least Ambiguous set-valued Classifier) variant of
split conformal classification.
The non-conformity score for a calibration sample ``(x_i, y_i)`` is
``1 - p_hat[y_i | x_i]``; a single empirical quantile of those scores
gives a coverage guarantee on every test sample with no further model
training. Pure NumPy: no dependency outside the package.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import torch

from ..base.classifier import BaseSpectralClassifier
from ._base import BaseUncertaintyEstimator, _apply_classifier_preprocessing
from ._result import UncertaintyResult, _softmax


class ConformalPredictor(BaseUncertaintyEstimator):
    """Split conformal predictor with the LAC non-conformity score.

    Parameters
    ----------
    classifier : BaseSpectralClassifier
        A fitted classifier whose :meth:`predict_proba`-style softmax
        scores act as the underlying probability estimate.
    alpha : float, default=0.1
        Miscoverage level in ``(0, 1)``. The target marginal coverage
        is ``1 - alpha``.
    score : {"lac"}, default="lac"
        Non-conformity score. Currently only ``"lac"`` is supported:
        ``s(x, y) = 1 - p_hat(y | x)``.

    Notes
    -----
    The :attr:`UncertaintyResult.uncertainty` field is the prediction
    set size normalised by ``n_classes``: ``1 / n_classes`` for a
    singleton set, ``1.0`` when every class is included. Empty sets
    (which can occur with very small calibration sets) yield ``0`` and
    are flagged in metadata via the empty-set count.
    """

    def __init__(
        self,
        classifier: BaseSpectralClassifier,
        alpha: float = 0.1,
        score: str = "lac",
    ) -> None:
        super().__init__(classifier)
        if not 0.0 < float(alpha) < 1.0:
            raise ValueError(f"alpha must be in (0, 1); got {alpha!r}.")
        if score != "lac":
            raise ValueError(
                f"score={score!r} is not supported; only 'lac' is implemented."
            )
        self.alpha = float(alpha)
        self.score = score
        self.quantile_: float | None = None
        self.calibration_coverage_: float | None = None
        self.n_calibration_: int | None = None

    def _classifier_proba(self, X_proc: np.ndarray) -> np.ndarray:
        """Run the classifier's ``model_`` in eval mode and return softmax."""
        device = self.classifier._device_
        X_t_full = torch.from_numpy(X_proc.astype(np.float32)).to(device)
        chunk = max(1, int(getattr(self.classifier, "batch_size", 32)))
        model = self.classifier.model_
        model.eval()
        logits_chunks: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, X_t_full.shape[0], chunk):
                x_chunk = X_t_full[start : start + chunk]
                logits_chunks.append(model(x_chunk).detach().cpu().numpy())
        if logits_chunks:
            logits = np.concatenate(logits_chunks, axis=0)
        else:
            logits = np.empty((0, int(self.classifier.n_classes_)), dtype=np.float32)
        temperature = getattr(self.classifier, "temperature_", None)
        if temperature is not None:
            logits = logits / float(temperature)
        return _softmax(logits)

    def calibrate(self, X_cal: Any, y_cal: Any) -> "ConformalPredictor":
        """Compute the conformal quantile from calibration data.

        Parameters
        ----------
        X_cal : array-like or MaldiSet of shape (n_samples, n_bins)
            Calibration spectra.
        y_cal : array-like of shape (n_samples,)
            Calibration labels using the original label space stored in
            ``classifier.classes_``.

        Returns
        -------
        ConformalPredictor
            ``self``, with :attr:`quantile_`,
            :attr:`calibration_coverage_`, and :attr:`n_calibration_`
            populated.
        """
        if hasattr(y_cal, "to_numpy"):
            y_cal = y_cal.to_numpy()
        y_np = np.asarray(y_cal).ravel()
        classes = np.asarray(self.classifier.classes_)
        if not np.all(np.isin(y_np, classes)):
            unknown = np.setdiff1d(np.unique(y_np), classes).tolist()
            raise ValueError(f"y_cal contains labels not seen at fit time: {unknown}.")
        y_encoded = np.searchsorted(classes, y_np).astype(np.int64)

        X_proc = _apply_classifier_preprocessing(self.classifier, X_cal)
        if X_proc.shape[0] != y_encoded.shape[0]:
            raise ValueError(
                f"X_cal has {X_proc.shape[0]} rows but y_cal has {y_encoded.shape[0]}."
            )
        n_cal = int(X_proc.shape[0])
        if n_cal < 1:
            raise ValueError("Calibration set must contain at least one sample.")

        proba = self._classifier_proba(X_proc)
        true_proba = proba[np.arange(n_cal), y_encoded]
        scores = 1.0 - true_proba
        sorted_scores = np.sort(scores)
        k = int(np.ceil((n_cal + 1) * (1.0 - self.alpha)))
        k = min(max(k, 1), n_cal)
        quantile = float(sorted_scores[k - 1])

        threshold = 1.0 - quantile
        prediction_sets = proba >= threshold
        covered = prediction_sets[np.arange(n_cal), y_encoded]
        coverage = float(np.mean(covered))

        target = 1.0 - self.alpha
        if coverage < target:
            warnings.warn(
                f"Empirical calibration coverage {coverage:.3f} is below the "
                f"target {target:.3f}. Consider a larger calibration set or a "
                "less aggressive alpha.",
                stacklevel=2,
            )

        self.quantile_ = quantile
        self.calibration_coverage_ = coverage
        self.n_calibration_ = n_cal
        return self

    def predict_with_uncertainty(self, X: Any) -> UncertaintyResult:
        """Return conformal predictions and prediction sets for ``X``.

        Returns
        -------
        UncertaintyResult
            ``method="conformal"`` with :attr:`epistemic` and
            :attr:`aleatoric` set to ``None``. Boolean prediction sets
            are stored in ``metadata["prediction_sets"]`` with shape
            ``(n_samples, n_classes)``; the empirical calibration
            coverage is stored in ``metadata["calibration_coverage"]``.

        Raises
        ------
        RuntimeError
            If :meth:`calibrate` has not been called.
        """
        if self.quantile_ is None:
            raise RuntimeError(
                "ConformalPredictor has not been calibrated. "
                "Call calibrate(X_cal, y_cal) before predict_with_uncertainty."
            )
        X_proc = _apply_classifier_preprocessing(self.classifier, X)
        proba = self._classifier_proba(X_proc)

        threshold = 1.0 - float(self.quantile_)
        prediction_sets = proba >= threshold

        n_classes = int(self.classifier.n_classes_)
        set_sizes = prediction_sets.sum(axis=1).astype(np.float64)
        uncertainty = np.clip(set_sizes / float(n_classes), 0.0, 1.0)

        idx = np.argmax(proba, axis=1)
        predictions = np.asarray(self.classifier.classes_)[idx]

        metadata: dict[str, Any] = {
            "prediction_sets": prediction_sets,
            "calibration_coverage": self.calibration_coverage_,
            "quantile": float(self.quantile_),
            "alpha": float(self.alpha),
            "n_calibration": self.n_calibration_,
            "set_sizes": set_sizes.astype(np.int64, copy=False),
            "n_empty_sets": int(np.sum(set_sizes == 0)),
        }

        return UncertaintyResult(
            predictions=predictions,
            proba_mean=proba.astype(np.float64, copy=False),
            uncertainty=uncertainty.astype(np.float64, copy=False),
            epistemic=None,
            aleatoric=None,
            method="conformal",
            metadata=metadata,
        )


__all__ = ["ConformalPredictor"]
