"""sklearn-compatible dual-input classifier for MALDI-TOF spectra.

:class:`MaldiDualBranchClassifier` conditions classification on both
the spectrum and a per-row side feature (e.g. an antibiotic
fingerprint). The side feature is packed into ``X`` by concatenation
so the resulting class behaves like any other
:class:`maldideepkit.BaseSpectralClassifier` from sklearn's
perspective (``GridSearchCV``, ``Pipeline``, and ``cross_val_score``
all work out of the box).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from sklearn.base import clone
from torch import nn

from ..base.classifier import BaseSpectralClassifier
from ..base.dual import BaseDualSpectralClassifier

__all__ = ["MaldiDualBranchClassifier"]


_DEFAULT_SPEC_HIDDEN_DIMS = (256, 128)
_DEFAULT_FP_HIDDEN_DIMS = (256, 128)


class _FusedDualBranchModule(nn.Module):
    """Spectral branch + side branch + concat + fused linear head."""

    def __init__(
        self,
        spectral: nn.Module,
        side: nn.Module,
        head: nn.Linear,
    ) -> None:
        super().__init__()
        self.spectral = spectral
        self.side = side
        self.head = head

    def forward(self, x_spec: torch.Tensor, x_side: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """Map ``(x_spec, x_side)`` to logits via concat + head."""
        f_spec = self.spectral(x_spec)
        f_side = self.side(x_side)
        return self.head(torch.cat([f_spec, f_side], dim=1))


class MaldiDualBranchClassifier(BaseDualSpectralClassifier):
    """sklearn-compatible dual-input classifier for MALDI-TOF spectra.

    The user calls ``fit(X_full, y)`` where ``X_full`` is the column
    concatenation ``np.hstack([X_spectra, side_features])``. The class
    knows where to split via the required ``n_spec_bins`` parameter.

    Architecture:

    1. **Spectral branch**: default a small MLP
       ``Linear -> ReLU -> Dropout -> ... -> Linear -> ReLU``. When
       ``backbone=`` is supplied (a
       :class:`maldideepkit.BaseSpectralClassifier` instance), the
       backbone's ``_build_model()`` is invoked, the final
       :class:`torch.nn.Linear` is replaced with
       :class:`torch.nn.Identity`, and the remaining module is used as
       a feature extractor.
    2. **Side branch**: small MLP
       ``Linear -> ReLU -> Dropout -> ... -> Linear -> ReLU``.
    3. **Fused head**: ``Linear(d_spec_feat + d_side_feat, n_classes)``.

    Parameters
    ----------
    n_spec_bins : int
        Number of spectrum-side feature columns in ``X``. Required.
    spec_hidden_dims : sequence of int, default=(256, 128)
        Hidden-layer widths of the **default** spectral-branch MLP.
        Ignored when ``backbone=`` is supplied. Last entry is the
        output dim of the spectral branch.
    spec_dropout : float, default=0.2
        Dropout between hidden layers of the default spectral-branch
        MLP. Ignored when ``backbone=`` is supplied.
    fp_hidden_dims : sequence of int, default=(256, 128)
        Hidden-layer widths of the side-branch MLP. Last entry is
        the output dim of the side branch.
    fp_dropout : float, default=0.2
        Dropout between hidden layers of the side-branch MLP.
    backbone : maldideepkit.BaseSpectralClassifier or None, default=None
        When supplied, used as the spectral branch (head replaced by
        :class:`torch.nn.Identity` so the module returns features).
        When ``None``, the built-in MLP from ``spec_hidden_dims`` is
        used.
    **kwargs
        Forwarded to :class:`BaseSpectralClassifier` /
        :class:`BaseDualSpectralClassifier`. See those docstrings for
        the full hyperparameter list (``learning_rate``,
        ``batch_size``, ``epochs``, ``warping``, ...).

    Notes
    -----
    The class inherits everything from
    :class:`BaseDualSpectralClassifier`, which inherits from
    :class:`BaseSpectralClassifier`. ``sklearn.base.clone`` is fully
    supported.

    Spectrum-specific preprocessing stops at the ``n_spec_bins``
    boundary: ``warping``, ``standardize``, ``input_transform``,
    ``augment`` and ``cutmix_alpha`` act on the spectrum columns only,
    leaving the side block untouched (``mixup_alpha`` mixes both
    blocks with one shared ``lam``, which stays consistent with the
    mixed label). So ``input_transform="log1p"`` will not rewrite
    binary fingerprint bits, and a CutMix window cannot straddle the
    spectrum / side boundary. Preprocessing you *want* applied to the
    side block belongs in an upstream ``Pipeline`` step. See
    :class:`BaseDualSpectralClassifier` for the full breakdown.

    Examples
    --------
    >>> import numpy as np
    >>> from maldideepkit import MaldiDualBranchClassifier
    >>> rng = np.random.default_rng(0)
    >>> X_spec = rng.standard_normal((40, 64)).astype("float32")
    >>> X_side = rng.standard_normal((40, 16)).astype("float32")
    >>> X = np.hstack([X_spec, X_side])
    >>> y = rng.integers(0, 2, size=40)
    >>> clf = MaldiDualBranchClassifier(
    ...     n_spec_bins=64,
    ...     spec_hidden_dims=(32,),
    ...     fp_hidden_dims=(16,),
    ...     epochs=1, batch_size=8, random_state=0,
    ... ).fit(X, y)
    >>> clf.predict(X).shape
    (40,)

    With a CNN backbone for the spectral side:

    >>> from maldideepkit import MaldiCNNClassifier
    >>> clf = MaldiDualBranchClassifier(
    ...     n_spec_bins=64,
    ...     backbone=MaldiCNNClassifier(channels=(8, 16), head_dim=32),
    ...     fp_hidden_dims=(16,),
    ... )  # doctest: +SKIP
    """

    def __init__(
        self,
        n_spec_bins: int,
        *,
        spec_hidden_dims: Sequence[int] = _DEFAULT_SPEC_HIDDEN_DIMS,
        spec_dropout: float = 0.2,
        fp_hidden_dims: Sequence[int] = _DEFAULT_FP_HIDDEN_DIMS,
        fp_dropout: float = 0.2,
        backbone: BaseSpectralClassifier | None = None,
        # BaseSpectralClassifier kwargs (forwarded)
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
        augment: Any | None = None,
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
        super().__init__(
            n_spec_bins=n_spec_bins,
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
        self.spec_hidden_dims = spec_hidden_dims
        self.spec_dropout = spec_dropout
        self.fp_hidden_dims = fp_hidden_dims
        self.fp_dropout = fp_dropout
        self.backbone = backbone

    def _build_dual_model(
        self,
        *,
        input_dim_spec: int,
        input_dim_side: int,
        n_classes: int,
    ) -> nn.Module:
        spec_branch, spec_out = self._build_spectral_branch(input_dim_spec, n_classes)
        side_branch, side_out = self._build_side_branch(input_dim_side)
        head = nn.Linear(int(spec_out) + int(side_out), int(n_classes))
        return _FusedDualBranchModule(spec_branch, side_branch, head)

    def _build_spectral_branch(
        self,
        input_dim_spec: int,
        n_classes: int,
    ) -> tuple[nn.Module, int]:
        """Build the spectral branch + return its output dim."""
        if self.backbone is None:
            return self._build_default_mlp(
                input_dim_spec,
                hidden_dims=tuple(self.spec_hidden_dims),
                dropout=float(self.spec_dropout),
            )

        # User backbone path: clone, force input_dim_/n_classes_, build,
        # then replace the head with Identity to expose features.
        if not isinstance(self.backbone, BaseSpectralClassifier):
            raise TypeError(
                f"backbone= must be a BaseSpectralClassifier instance "
                f"(got {type(self.backbone).__name__}). Pass an instance "
                "such as MaldiMLPClassifier() or MaldiCNNClassifier()."
            )
        bb = clone(self.backbone)
        bb.input_dim_ = int(input_dim_spec)
        bb.n_classes_ = int(n_classes)
        full = bb._build_model()
        spec_out = self._find_last_linear_in_features(full)
        spec = self._replace_last_linear_with_identity(full)
        return spec, spec_out

    def _build_side_branch(
        self,
        input_dim_side: int,
    ) -> tuple[nn.Module, int]:
        """Build the side branch MLP + return its output dim."""
        return self._build_default_mlp(
            input_dim_side,
            hidden_dims=tuple(self.fp_hidden_dims),
            dropout=float(self.fp_dropout),
        )

    @staticmethod
    def _build_default_mlp(
        input_dim: int,
        *,
        hidden_dims: tuple[int, ...],
        dropout: float,
    ) -> tuple[nn.Module, int]:
        """Linear-ReLU-Dropout-Linear-ReLU MLP, returning (module, out_dim)."""
        if not hidden_dims:
            raise ValueError(
                "hidden_dims must contain at least one entry; got an empty tuple."
            )
        layers: list[nn.Module] = []
        in_dim = int(input_dim)
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, int(h)))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(float(dropout)))
            in_dim = int(h)
        if isinstance(layers[-1], nn.Dropout):
            layers.pop()
        return nn.Sequential(*layers), int(hidden_dims[-1])

    @staticmethod
    def _replace_last_linear_with_identity(module: nn.Module) -> nn.Module:
        """Replace the final :class:`nn.Linear` in ``module`` with Identity."""
        last_parent = None
        last_name = None
        for name, sub in module.named_modules():
            if isinstance(sub, nn.Linear):
                parent_path, _, last_name = name.rpartition(".")
                last_parent = (
                    module if parent_path == "" else module.get_submodule(parent_path)
                )
        if last_parent is None or last_name is None:
            raise RuntimeError(
                "MaldiDualBranchClassifier could not locate a final "
                f"nn.Linear in the spectral branch "
                f"{type(module).__name__}."
            )
        setattr(last_parent, last_name, nn.Identity())
        return module

    @staticmethod
    def _find_last_linear_in_features(module: nn.Module) -> int:
        """Return the ``in_features`` of the last :class:`nn.Linear` in ``module``."""
        last = None
        for sub in module.modules():
            if isinstance(sub, nn.Linear):
                last = sub
        if last is None:
            raise RuntimeError(
                "Could not infer spectral feature dim: backbone module "
                "has no nn.Linear layer."
            )
        return int(last.in_features)
