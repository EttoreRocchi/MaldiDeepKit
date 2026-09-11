"""Abstract base for sklearn-compatible **dual-input** spectral classifiers.

:class:`BaseDualSpectralClassifier` extends
:class:`maldideepkit.BaseSpectralClassifier` to classifiers that
consume a per-row **side input** alongside the spectrum (e.g. an
antibiotic fingerprint, a culture-medium descriptor, an instrument
ID one-hot). The side input is **packed into ``X`` by concatenation**
so the resulting class is fully sklearn-pipeline-friendly:

- ``fit(X, y)`` / ``predict(X)`` / ``predict_proba(X)`` /
  ``score(X, y)`` keep their single-array signatures.
- ``GridSearchCV`` and ``Pipeline`` work transparently because the
  splitter only sees ``X``; the spectrum / side split happens inside
  the model.

Subclasses override the single abstract method
:meth:`_build_dual_model(input_dim_spec, input_dim_side, n_classes)`
returning an :class:`torch.nn.Module` whose ``forward(x_spec, x_side)``
maps the two halves to logits.
"""

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch import nn

from .classifier import BaseSpectralClassifier

__all__ = ["BaseDualSpectralClassifier"]


class _DualInputSplitter(nn.Module):
    """Adapter that splits a concatenated input and dispatches to a dual module.

    The concatenated input layout is ``[ X_spec | X_side ]`` along the
    feature axis. ``n_spec_bins`` controls the split point.
    """

    def __init__(self, inner: nn.Module, n_spec_bins: int) -> None:
        super().__init__()
        self.inner = inner
        self.n_spec_bins = int(n_spec_bins)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """Split ``x`` along dim 1 and call the inner dual module."""
        x_spec = x[:, : self.n_spec_bins]
        x_side = x[:, self.n_spec_bins :]
        return self.inner(x_spec, x_side)


class BaseDualSpectralClassifier(BaseSpectralClassifier):
    """Abstract base for dual-input spectral classifiers.

    Parameters
    ----------
    n_spec_bins : int
        Number of spectrum-side feature columns in ``X``. The remaining
        ``X.shape[1] - n_spec_bins`` columns are treated as the side
        input. This **must** be supplied so the class knows where to
        split.

    Notes
    -----
    Every other keyword argument is a
    :class:`BaseSpectralClassifier` hyperparameter, re-declared here
    explicitly (rather than swallowed by ``**kwargs``) so that
    ``get_params`` / ``clone`` (and therefore ``GridSearchCV``,
    ``Pipeline`` and ``cross_val_score``) see the full hyperparameter
    surface on every subclass. See the
    :class:`BaseSpectralClassifier` docstring for the meaning of each
    (``learning_rate``, ``batch_size``, ``epochs``, ``warping``, ...).

    Concrete subclasses override exactly
    :meth:`_build_dual_model(input_dim_spec, input_dim_side, n_classes)`
    returning an :class:`torch.nn.Module` whose forward signature is
    ``forward(x_spec, x_side) -> (batch, n_classes)``. The base class
    handles the concatenation-split via an internal
    :class:`_DualInputSplitter` wrapper so the result behaves like any
    other :class:`BaseSpectralClassifier` from sklearn's perspective.

    *Preprocessing is confined to the spectrum block.* Every knob that
    assumes its input is a spectrum applies to the first
    ``n_spec_bins`` columns only; the side block is passed through
    untouched:

    - ``warping``, ``standardize`` and ``input_transform`` are fitted
      and applied on the spectrum columns alone, so
      :attr:`feature_mean_` / :attr:`feature_std_` have length
      ``n_spec_bins``, and ``input_transform="log1p"`` never rewrites
      binary fingerprint bits.
    - ``augment`` sees the spectrum block only.
    - ``cutmix_alpha`` draws its contiguous window inside the spectrum
      block, so a window can no longer straddle the spectrum / side
      boundary; the side block is convex-combined with the same
      ``effective_lam``, keeping ``(x, y)`` consistent.
    - ``mixup_alpha`` is deliberately *not* confined: it already
      convex-combines every column with one shared ``lam``, which is
      the coherent multi-modal formulation (blending the spectra
      while leaving the side features unblended would contradict the
      blended label).

    Anything you want applied to the side block (e.g. scaling a
    continuous descriptor) belongs in a ``Pipeline`` step upstream of
    the estimator.

    Examples
    --------
    A minimal subclass that concatenates the two branches' features
    and feeds them through a single linear head:

    >>> from torch import nn
    >>> from maldideepkit.base.dual import BaseDualSpectralClassifier
    >>>
    >>> class MyDualClassifier(BaseDualSpectralClassifier):
    ...     def _build_dual_model(self, *, input_dim_spec, input_dim_side, n_classes):
    ...         class _Module(nn.Module):
    ...             def __init__(self):
    ...                 super().__init__()
    ...                 self.spec = nn.Linear(input_dim_spec, 64)
    ...                 self.side = nn.Linear(input_dim_side, 32)
    ...                 self.head = nn.Linear(64 + 32, n_classes)
    ...             def forward(self, x_spec, x_side):
    ...                 import torch as _t
    ...                 fs = self.spec(x_spec).relu()
    ...                 ff = self.side(x_side).relu()
    ...                 return self.head(_t.cat([fs, ff], dim=1))
    ...         return _Module()
    """

    def __init__(
        self,
        *,
        n_spec_bins: int,
        input_dim: int | None = None,
        n_classes: int = 2,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        grad_clip_norm: float | None = None,
        label_smoothing: float = 0.0,
        loss: str = "cross_entropy",
        focal_gamma: float = 2.0,
        use_amp: bool = False,
        swa_start_epoch: int | None = None,
        tune_threshold: bool = False,
        threshold_metric: str = "balanced_accuracy",
        calibrate_temperature: bool = False,
        min_val_auroc_for_threshold_tune: float = 0.6,
        use_sam: bool = False,
        sam_rho: float = 0.05,
        batch_size: int = 32,
        epochs: int = 100,
        early_stopping_patience: int = 10,
        val_fraction: float = 0.1,
        warmup_epochs: int = 0,
        standardize: bool = False,
        input_transform: str | None = None,
        warping: Any | None = None,
        metrics_log_path: str | Path | None = None,
        track_train_metrics: bool = False,
        augment: Callable[[torch.Tensor], torch.Tensor] | None = None,
        mixup_alpha: float = 0.0,
        cutmix_alpha: float = 0.0,
        ema_decay: float | None = None,
        retry_on_val_auroc_below: float | None = None,
        max_retries: int = 2,
        class_weight: str | np.ndarray | list | None = None,
        device: str | torch.device = "auto",
        random_state: int = 0,
        verbose: bool = False,
    ) -> None:
        if int(n_spec_bins) <= 0:
            raise ValueError(
                f"n_spec_bins must be a positive integer, got {n_spec_bins!r}."
            )
        super().__init__(
            input_dim=input_dim,
            n_classes=n_classes,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            grad_clip_norm=grad_clip_norm,
            label_smoothing=label_smoothing,
            loss=loss,
            focal_gamma=focal_gamma,
            use_amp=use_amp,
            swa_start_epoch=swa_start_epoch,
            tune_threshold=tune_threshold,
            threshold_metric=threshold_metric,
            calibrate_temperature=calibrate_temperature,
            min_val_auroc_for_threshold_tune=min_val_auroc_for_threshold_tune,
            use_sam=use_sam,
            sam_rho=sam_rho,
            batch_size=batch_size,
            epochs=epochs,
            early_stopping_patience=early_stopping_patience,
            val_fraction=val_fraction,
            warmup_epochs=warmup_epochs,
            standardize=standardize,
            input_transform=input_transform,
            warping=warping,
            metrics_log_path=metrics_log_path,
            track_train_metrics=track_train_metrics,
            augment=augment,
            mixup_alpha=mixup_alpha,
            cutmix_alpha=cutmix_alpha,
            ema_decay=ema_decay,
            retry_on_val_auroc_below=retry_on_val_auroc_below,
            max_retries=max_retries,
            class_weight=class_weight,
            device=device,
            random_state=random_state,
            verbose=verbose,
        )
        self.n_spec_bins = int(n_spec_bins)

    @abstractmethod
    def _build_dual_model(
        self,
        *,
        input_dim_spec: int,
        input_dim_side: int,
        n_classes: int,
    ) -> nn.Module:
        """Return a fresh :class:`nn.Module` mapping (x_spec, x_side) -> logits.

        Parameters
        ----------
        input_dim_spec : int
            Number of spectrum-side feature columns. Equals
            ``self.n_spec_bins``.
        input_dim_side : int
            Number of side-input feature columns. Equals
            ``self.input_dim_ - self.n_spec_bins`` at ``fit`` time.
        n_classes : int
            Output logits dim. Equals ``self.n_classes_`` at ``fit`` time.

        Returns
        -------
        module : torch.nn.Module
            Module whose ``forward(x_spec, x_side)`` returns logits of
            shape ``(batch, n_classes)``.
        """

    def _n_transform_cols(self) -> int | None:
        """Confine spectrum preprocessing to the first ``n_spec_bins`` columns."""
        return int(self.n_spec_bins)

    def _build_model(self) -> nn.Module:
        if self.input_dim_ <= self.n_spec_bins:
            raise ValueError(
                f"X has {self.input_dim_} feature columns but "
                f"n_spec_bins={self.n_spec_bins}: the side input would "
                "have zero columns. Did you forget to hstack the "
                "side features onto X?"
            )
        input_dim_side = int(self.input_dim_) - int(self.n_spec_bins)
        inner = self._build_dual_model(
            input_dim_spec=int(self.n_spec_bins),
            input_dim_side=input_dim_side,
            n_classes=int(self.n_classes_),
        )
        return _DualInputSplitter(inner, self.n_spec_bins)
