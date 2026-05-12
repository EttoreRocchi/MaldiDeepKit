"""Monte Carlo Dropout uncertainty estimator.

Runs the underlying classifier's forward pass ``n_samples`` times with
dropout layers kept in training mode. The variance of the resulting
softmax distribution is interpreted as epistemic uncertainty (model
disagreement), while the mean of the per-pass entropies is interpreted
as aleatoric uncertainty (data noise) - following the decomposition in
Kendall and Gal (2017).
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import torch
from torch import nn

from ..base.classifier import BaseSpectralClassifier
from ._base import BaseUncertaintyEstimator, _apply_classifier_preprocessing
from ._result import UncertaintyResult, _entropy, _softmax


def _has_dropout(model: nn.Module) -> bool:
    """Return ``True`` iff ``model`` contains at least one dropout layer."""
    return any(
        isinstance(m, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d))
        for m in model.modules()
    )


class MCDropoutEstimator(BaseUncertaintyEstimator):
    """Monte Carlo Dropout estimator (Gal and Ghahramani, 2016).

    Runs ``n_samples`` stochastic forward passes through ``classifier.model_``
    with dropout layers active and aggregates the resulting softmax
    distribution into a predictive mean plus an epistemic / aleatoric
    decomposition.

    Parameters
    ----------
    classifier : BaseSpectralClassifier
        A fitted classifier whose architecture contains at least one
        :class:`torch.nn.Dropout` (or its 1-D / 2-D / 3-D variants). A
        warning is emitted if it does not, but the estimator still runs
        - in that degenerate case all ``n_samples`` passes are identical
        and epistemic uncertainty collapses to ``0``.
    n_samples : int, default=30
        Number of stochastic forward passes per call to
        :meth:`predict_with_uncertainty`. Higher values reduce variance
        of the estimate at a linear cost in compute.
    batch_size : int or None, default=None
        Mini-batch size used to chunk the input through each forward
        pass. When ``None``, falls back to ``classifier.batch_size``.

    Notes
    -----
    The scalar :attr:`UncertaintyResult.uncertainty` field is the
    normalised entropy of the predictive mean ``H(E[p])``. The
    epistemic / aleatoric decomposition follows Depeweg et al. (2018):

    - ``aleatoric = E_w[H(p)]`` (mean of per-pass entropies)
    - ``epistemic = H(E[p]) - E[H(p)]`` (mutual information)

    Both components are normalised to ``[0, 1]`` by dividing the raw
    entropies by ``log(n_classes)``.
    """

    def __init__(
        self,
        classifier: BaseSpectralClassifier,
        n_samples: int = 30,
        batch_size: int | None = None,
    ) -> None:
        super().__init__(classifier)
        if int(n_samples) < 1:
            raise ValueError(f"n_samples must be >= 1; got {n_samples!r}.")
        self.n_samples = int(n_samples)
        self.batch_size = None if batch_size is None else int(batch_size)
        if not _has_dropout(classifier.model_):
            warnings.warn(
                f"{type(classifier).__name__} contains no Dropout layers; "
                "MCDropoutEstimator will produce identical samples and zero "
                "epistemic uncertainty.",
                stacklevel=2,
            )

    def _resolve_batch_size(self) -> int:
        if self.batch_size is not None:
            return max(1, int(self.batch_size))
        return max(1, int(getattr(self.classifier, "batch_size", 32)))

    def predict_with_uncertainty(
        self, X: Any, *, store_samples: bool = False
    ) -> UncertaintyResult:
        """Run ``n_samples`` MC forward passes and decompose uncertainty.

        Parameters
        ----------
        X : array-like or MaldiSet of shape (n_samples, n_bins)
            Spectra to score.
        store_samples : bool, default=False
            If ``True``, attach the raw per-pass softmax tensor of shape
            ``(n_mc_passes, n_data_samples, n_classes)`` to
            ``metadata["per_pass_proba"]``. Off by default to keep
            memory bounded on large datasets.

        Returns
        -------
        UncertaintyResult
            ``method="mc_dropout"`` with both epistemic and aleatoric
            components populated.
        """
        X_np = _apply_classifier_preprocessing(self.classifier, X)
        device = self.classifier._device_
        X_t_full = torch.from_numpy(X_np.astype(np.float32)).to(device)
        chunk = self._resolve_batch_size()

        model = self.classifier.model_
        was_training = model.training
        per_pass_logits: list[np.ndarray] = []
        try:
            model.train()
            with torch.no_grad():
                for _ in range(self.n_samples):
                    chunks: list[np.ndarray] = []
                    for start in range(0, X_t_full.shape[0], chunk):
                        x_chunk = X_t_full[start : start + chunk]
                        chunks.append(model(x_chunk).detach().cpu().numpy())
                    if chunks:
                        per_pass_logits.append(np.concatenate(chunks, axis=0))
                    else:
                        per_pass_logits.append(
                            np.empty(
                                (0, int(self.classifier.n_classes_)), dtype=np.float32
                            )
                        )
        finally:
            model.eval()
            if was_training:
                # Should never happen given fit() leaves the model in eval(),
                # but restore the original mode if it does.
                model.train()
                model.eval()

        logits_arr = np.stack(per_pass_logits, axis=0)
        per_pass_proba = _softmax(logits_arr)
        proba_mean = per_pass_proba.mean(axis=0)

        n_classes = int(self.classifier.n_classes_)
        total = _entropy(proba_mean)
        per_pass_entropy = _entropy(per_pass_proba)
        aleatoric = per_pass_entropy.mean(axis=0)
        epistemic = np.clip(total - aleatoric, 0.0, 1.0)

        idx = np.argmax(proba_mean, axis=1)
        predictions = np.asarray(self.classifier.classes_)[idx]

        metadata: dict[str, Any] = {
            "n_samples": int(self.n_samples),
            "n_classes": n_classes,
        }
        if store_samples:
            metadata["per_pass_proba"] = per_pass_proba.astype(np.float32, copy=False)

        return UncertaintyResult(
            predictions=predictions,
            proba_mean=proba_mean.astype(np.float64, copy=False),
            uncertainty=total.astype(np.float64, copy=False),
            epistemic=epistemic.astype(np.float64, copy=False),
            aleatoric=aleatoric.astype(np.float64, copy=False),
            method="mc_dropout",
            metadata=metadata,
        )


__all__ = ["MCDropoutEstimator"]
