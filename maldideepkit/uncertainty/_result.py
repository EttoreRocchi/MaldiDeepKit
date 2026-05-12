"""Container dataclass and shared helpers for uncertainty estimators.

The :class:`UncertaintyResult` dataclass is the common return value of
every estimator in :mod:`maldideepkit.uncertainty`. The two helpers
:func:`_softmax` and :func:`_entropy` are reused across the submodules
to keep the numerical conventions identical between methods.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Return numerically-stable row-wise softmax of ``logits``.

    Parameters
    ----------
    logits : ndarray of shape (..., n_classes)
        Unnormalised log-probabilities. The last axis is the class axis.

    Returns
    -------
    ndarray
        Same shape as ``logits``; rows sum to 1 along the class axis.
    """
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def _entropy(proba: np.ndarray) -> np.ndarray:
    """Return per-row Shannon entropy of ``proba`` normalised to ``[0, 1]``.

    Divides by ``log(n_classes)`` so the value is independent of the
    cardinality of the class set; ``0`` corresponds to a one-hot vector
    and ``1`` to a uniform distribution over classes.

    Parameters
    ----------
    proba : ndarray of shape (..., n_classes)
        Probabilities that sum to 1 along the last axis.

    Returns
    -------
    ndarray of shape (...,)
        Entropy of each row in ``[0, 1]``.
    """
    proba = np.asarray(proba, dtype=np.float64)
    n_classes = int(proba.shape[-1])
    if n_classes < 2:
        return np.zeros(proba.shape[:-1], dtype=np.float64)
    safe = np.clip(proba, 1e-12, 1.0)
    h = -np.sum(safe * np.log(safe), axis=-1)
    return (h / np.log(n_classes)).astype(np.float64, copy=False)


@dataclass(frozen=True)
class UncertaintyResult:
    """Container for the output of a :class:`BaseUncertaintyEstimator`.

    Attributes
    ----------
    predictions : ndarray of shape (n_samples,)
        Hard labels drawn from the classifier's :attr:`classes_`.
    proba_mean : ndarray of shape (n_samples, n_classes)
        Mean softmax probabilities. For non-Bayesian methods this is
        simply the classifier's :meth:`predict_proba` output.
    uncertainty : ndarray of shape (n_samples,)
        Scalar per-sample uncertainty in ``[0, 1]`` where possible. The
        precise semantics are method-specific and documented on the
        producing class.
    epistemic : ndarray of shape (n_samples,) or None
        Model-uncertainty component, when the method decomposes total
        uncertainty into epistemic and aleatoric parts. ``None`` for
        methods that do not decompose.
    aleatoric : ndarray of shape (n_samples,) or None
        Data-uncertainty component, paired with :attr:`epistemic`.
        ``None`` for methods that do not decompose.
    method : str
        Name of the method that produced this result (e.g.
        ``"mc_dropout"``, ``"laplace"``, ``"conformal"``).
    metadata : dict
        Method-specific extras (per-pass samples for MC Dropout,
        prediction sets for conformal, predictive variance for
        Laplace, ...).
    """

    predictions: np.ndarray
    proba_mean: np.ndarray
    uncertainty: np.ndarray
    epistemic: np.ndarray | None
    aleatoric: np.ndarray | None
    method: str
    metadata: dict[str, Any] = field(default_factory=dict)


__all__ = ["UncertaintyResult", "_entropy", "_softmax"]
