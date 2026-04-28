"""Post-hoc calibration helpers used by :class:`BaseSpectralClassifier`.

- :func:`tune_threshold` picks the binary decision threshold on a
  validation set that maximises a chosen metric.
- :func:`fit_temperature` optimises a single temperature scalar by
  LBFGS on held-out logits for probability calibration.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_curve


def tune_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    metric: str = "balanced_accuracy",
) -> float:
    """Pick the binary decision threshold that maximises ``metric``.

    Sweeps the unique observed probabilities (capped at 1000 quantiles)
    so severely-imbalanced settings still resolve. Falls back to a
    99-point ``linspace(0.01, 0.99)`` only when no probability lies
    strictly inside ``(0, 1)``.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Binary ground-truth labels in ``{0, 1}``.
    y_proba : array-like of shape (n_samples,) or (n_samples, 2)
        Predicted positive-class probabilities. If a 2-D array is
        given, column index ``1`` is used.
    metric : {"balanced_accuracy", "f1", "youden"}, default="balanced_accuracy"
        Which metric to maximise. ``"youden"`` = TPR - FPR.

    Returns
    -------
    float
        Threshold in ``(0, 1)``. Use as ``y_pred = (y_proba >= t)``.
    """
    y_true = np.asarray(y_true).ravel().astype(int)
    y_proba_arr = np.asarray(y_proba, dtype=float)
    if y_proba_arr.ndim == 2:
        if y_proba_arr.shape[1] != 2:
            raise ValueError(
                "tune_threshold is binary-only; "
                f"got y_proba with {y_proba_arr.shape[1]} columns."
            )
        y_proba_arr = y_proba_arr[:, 1]
    y_proba_arr = y_proba_arr.ravel()

    if metric == "youden":
        fpr, tpr, thr = roc_curve(y_true, y_proba_arr)
        valid = (thr > 0) & (thr < 1)
        if not valid.any():
            return 0.5
        j = tpr[valid] - fpr[valid]
        return float(thr[valid][int(np.argmax(j))])

    unique = np.unique(y_proba_arr)
    unique = unique[(unique > 0) & (unique < 1)]
    if unique.size == 0:
        candidates = np.linspace(0.01, 0.99, 99)
    elif unique.size > 1000:
        candidates = np.quantile(unique, np.linspace(0.0, 1.0, 1000))
    else:
        candidates = unique
    best_t, best_score = 0.5, -np.inf
    for t in candidates:
        pred = (y_proba_arr >= t).astype(int)
        if metric == "balanced_accuracy":
            score = balanced_accuracy_score(y_true, pred)
        elif metric == "f1":
            score = f1_score(y_true, pred, zero_division=0)
        else:
            raise ValueError(
                f"Unknown metric={metric!r}; "
                "expected 'balanced_accuracy', 'f1', or 'youden'."
            )
        if score > best_score:
            best_score, best_t = score, float(t)
    return best_t


def fit_temperature(
    logits: torch.Tensor | np.ndarray,
    y_true: torch.Tensor | np.ndarray,
    max_iter: int = 200,
    lr: float = 1e-1,
) -> float:
    """Fit a scalar temperature by LBFGS minimisation of NLL.

    Applies to raw logits (not probabilities). Returns the temperature
    ``T`` such that ``softmax(logits / T)`` is better-calibrated than
    the unscaled softmax.

    Parameters
    ----------
    logits : torch.Tensor or ndarray of shape (n_samples, n_classes)
        Held-out logits.
    y_true : torch.Tensor or ndarray of shape (n_samples,)
        Ground-truth class indices.
    max_iter : int, default=200
        LBFGS max iterations.
    lr : float, default=1e-1
        LBFGS step size.

    Returns
    -------
    float
        Fitted temperature; strictly positive.
    """
    if not isinstance(logits, torch.Tensor):
        logits = torch.as_tensor(logits, dtype=torch.float32)
    if not isinstance(y_true, torch.Tensor):
        y_true = torch.as_tensor(np.asarray(y_true).ravel(), dtype=torch.long)
    else:
        y_true = y_true.to(torch.long).view(-1)

    log_temperature = torch.zeros(1, device=logits.device, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=lr, max_iter=max_iter)

    def _closure():
        optimizer.zero_grad()
        t = torch.exp(log_temperature)
        loss = F.cross_entropy(logits / t, y_true)
        loss.backward()
        return loss

    optimizer.step(_closure)
    return float(torch.exp(log_temperature).detach().item())
