"""Laplace-approximation uncertainty estimator.

Wraps the ``laplace-torch`` package to fit a (last-layer or full)
Gaussian approximation of the posterior over the network's weights and
turn the predictive variance into an epistemic / aleatoric
decomposition.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..base.classifier import BaseSpectralClassifier
from ..base.data import SpectralDataset
from ._base import BaseUncertaintyEstimator, _apply_classifier_preprocessing
from ._result import UncertaintyResult, _entropy


class LaplaceEstimator(BaseUncertaintyEstimator):
    """Laplace-approximation uncertainty estimator.

    A thin wrapper around the ``laplace-torch`` package
    (`<https://github.com/aleximmer/Laplace>`_) that fits a Gaussian
    posterior over the classifier's weights and turns the predictive
    variance into a per-sample uncertainty estimate.

    Parameters
    ----------
    classifier : BaseSpectralClassifier
        A fitted classifier; its ``model_`` is reused as the network
        whose posterior is approximated.
    subset_of_weights : {"last_layer", "all"}, default="last_layer"
        Which subset of weights to model. ``"last_layer"`` is the
        standard, cheap default and works for any of the MaldiDeepKit
        backbones whose final layer is a :class:`torch.nn.Linear`.
    hessian_structure : {"diag", "kron"}, default="diag"
        Approximation structure for the Hessian. ``"diag"`` is the
        cheapest; ``"kron"`` (Kronecker-factored) is more accurate at a
        moderate compute cost.
    sigma_noise : float, default=1.0
        Forwarded to ``laplace-torch``. For classification it has no
        effect (it controls the regression noise scale) but is exposed
        for interface symmetry.

    Raises
    ------
    ImportError
        If ``laplace-torch`` is not installed.
    """

    def __init__(
        self,
        classifier: BaseSpectralClassifier,
        subset_of_weights: str = "last_layer",
        hessian_structure: str = "diag",
        sigma_noise: float = 1.0,
    ) -> None:
        super().__init__(classifier)
        if subset_of_weights not in {"last_layer", "all"}:
            raise ValueError(
                f"subset_of_weights must be 'last_layer' or 'all'; "
                f"got {subset_of_weights!r}."
            )
        if hessian_structure not in {"diag", "kron"}:
            raise ValueError(
                f"hessian_structure must be 'diag' or 'kron'; "
                f"got {hessian_structure!r}."
            )
        try:
            import laplace  # noqa: F401
        except ImportError as exc:  # pragma: no cover - guard tested via importorskip
            raise ImportError(
                "laplace-torch is required for LaplaceEstimator. "
                "Install it with: pip install laplace-torch"
            ) from exc
        self.subset_of_weights = subset_of_weights
        self.hessian_structure = hessian_structure
        self.sigma_noise = float(sigma_noise)
        self.la_: Any | None = None

    def calibrate(
        self,
        X_cal: Any,
        y_cal: Any,
        *,
        batch_size: int | None = None,
    ) -> "LaplaceEstimator":
        """Fit the Laplace approximation on ``(X_cal, y_cal)``.

        Applies the classifier's preprocessing to ``X_cal``, builds an
        internal :class:`~torch.utils.data.DataLoader`, fits the
        Laplace approximation, and runs marginal-likelihood prior
        precision optimisation.

        Parameters
        ----------
        X_cal : array-like or MaldiSet of shape (n_samples, n_bins)
            Calibration spectra.
        y_cal : array-like of shape (n_samples,)
            Calibration labels using the original label space stored in
            ``classifier.classes_``. Re-encoded to ``0..n_classes-1``
            internally.
        batch_size : int or None, default=None
            DataLoader batch size. When ``None``, falls back to
            ``classifier.batch_size``.

        Returns
        -------
        LaplaceEstimator
            ``self``, with the fitted Laplace approximation stored on
            :attr:`la_`.
        """
        from laplace import Laplace

        X_proc = _apply_classifier_preprocessing(self.classifier, X_cal)
        if hasattr(y_cal, "to_numpy"):
            y_cal = y_cal.to_numpy()
        y_np = np.asarray(y_cal).ravel()
        if X_proc.shape[0] != y_np.shape[0]:
            raise ValueError(
                f"X_cal has {X_proc.shape[0]} rows but y_cal has {y_np.shape[0]}."
            )
        classes = np.asarray(self.classifier.classes_)
        if not np.all(np.isin(y_np, classes)):
            unknown = np.setdiff1d(np.unique(y_np), classes).tolist()
            raise ValueError(f"y_cal contains labels not seen at fit time: {unknown}.")
        y_encoded = np.searchsorted(classes, y_np).astype(np.int64)

        bs = max(
            1, int(batch_size if batch_size is not None else self.classifier.batch_size)
        )
        dataset = SpectralDataset(X_proc, y_encoded, standardize=False)
        loader = DataLoader(dataset, batch_size=bs, shuffle=False, drop_last=False)

        device = self.classifier._device_
        model = self.classifier.model_
        model.eval()
        la = Laplace(
            model,
            likelihood="classification",
            subset_of_weights=self.subset_of_weights,
            hessian_structure=self.hessian_structure,
            sigma_noise=self.sigma_noise,
        )
        la.fit(loader)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"By default `link_approx` is `probit`.*",
                category=UserWarning,
            )
            try:
                la.optimize_prior_precision(method="marglik", link_approx="probit")
            except TypeError:
                la.optimize_prior_precision()
        self.la_ = la
        self._device_ = device
        return self

    def predict_with_uncertainty(self, X: Any) -> UncertaintyResult:
        """Return predictions and Laplace-derived uncertainty for ``X``.

        Uses ``pred_type="glm"`` and ``link_approx="probit"`` to map the
        weight-space posterior into a softmax-domain predictive
        distribution. Per-sample predictive variance is summarised as
        the mean diagonal entry, then squashed into ``[0, 1]`` via
        ``1 - exp(-v)`` and stored as ``epistemic``. The scalar
        :attr:`UncertaintyResult.uncertainty` field is the normalised
        entropy of the predictive mean; ``aleatoric`` is the
        non-negative residual ``uncertainty - epistemic``.

        Raises
        ------
        RuntimeError
            If :meth:`calibrate` has not been called.
        """
        if self.la_ is None:
            raise RuntimeError(
                "LaplaceEstimator has not been calibrated. "
                "Call calibrate(X_cal, y_cal) before predict_with_uncertainty."
            )
        X_proc = _apply_classifier_preprocessing(self.classifier, X)
        device = self.classifier._device_
        X_t = torch.from_numpy(X_proc.astype(np.float32)).to(device)

        with torch.no_grad():
            probs = self.la_(X_t, pred_type="glm", link_approx="probit")
        if isinstance(probs, tuple):
            probs = probs[0]
        proba_mean = probs.detach().cpu().numpy().astype(np.float64)

        epistemic_raw = self._epistemic_variance(X_t)
        epistemic = 1.0 - np.exp(-epistemic_raw)
        epistemic = np.clip(epistemic, 0.0, 1.0)

        total = _entropy(proba_mean)
        aleatoric = np.clip(total - epistemic, 0.0, 1.0)

        idx = np.argmax(proba_mean, axis=1)
        predictions = np.asarray(self.classifier.classes_)[idx]

        metadata: dict[str, Any] = {
            "subset_of_weights": self.subset_of_weights,
            "hessian_structure": self.hessian_structure,
            "predictive_variance": epistemic_raw.astype(np.float64, copy=False),
        }

        return UncertaintyResult(
            predictions=predictions,
            proba_mean=proba_mean,
            uncertainty=total.astype(np.float64, copy=False),
            epistemic=epistemic.astype(np.float64, copy=False),
            aleatoric=aleatoric.astype(np.float64, copy=False),
            method="laplace",
            metadata=metadata,
        )

    def _epistemic_variance(self, X_t: torch.Tensor) -> np.ndarray:
        """Per-sample mean predictive variance from the Laplace posterior.

        Tries the public sampling API first (works on every
        ``laplace-torch`` release that exposes
        :meth:`predictive_samples`) and falls back to the GLM
        predictive distribution otherwise.
        """
        with torch.no_grad():
            try:
                samples = self.la_.predictive_samples(
                    X_t, pred_type="glm", n_samples=100
                )
                var = samples.var(dim=0)
                return var.mean(dim=-1).detach().cpu().numpy().astype(np.float64)
            except (AttributeError, TypeError):
                pass
            f_mu, f_var = self.la_._glm_predictive_distribution(X_t)
            if f_var.dim() == 3:
                diag = torch.diagonal(f_var, dim1=-2, dim2=-1)
            else:
                diag = f_var
            return diag.mean(dim=-1).detach().cpu().numpy().astype(np.float64)


__all__ = ["LaplaceEstimator"]
