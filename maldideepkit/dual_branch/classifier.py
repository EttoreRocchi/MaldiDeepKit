"""sklearn-compatible dual-input classifier for MALDI-TOF spectra.

:class:`MaldiDualBranchClassifier` conditions classification on both
the spectrum and a per-row side feature (e.g. an antibiotic
fingerprint). The side feature is packed into ``X`` by concatenation,
so for scikit-learn the resulting class is like any other
:class:`~maldideepkit.BaseSpectralClassifier` (``GridSearchCV``,
``Pipeline``, and ``cross_val_score`` all accept it).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from sklearn.base import clone
from torch import nn

from ..base._registry import get_backbone_class
from ..base._serialization import _estimator_spec
from ..base._trunk import split_trunk, trunk_prefix
from ..base.classifier import BaseSpectralClassifier, _load_init_encoder
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
       ``backbone=`` is supplied (a registered name or a
       :class:`~maldideepkit.BaseSpectralClassifier` instance), the
       backbone's ``_build_model()`` is invoked, its final layer is
       replaced with :class:`torch.nn.Identity`, and the remaining
       module is used as a feature extractor.
    2. **Side branch**: small MLP
       ``Linear -> ReLU -> Dropout -> ... -> Linear -> ReLU``.
    3. **Fused head**: ``Linear(d_spec_feat + d_side_feat, n_classes)``.

    Parameters
    ----------
    n_spec_bins : int
        Number of spectrum columns at the start of ``X``; the remaining
        ``X.shape[1] - n_spec_bins`` columns are the side input.
        Required.
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
    backbone : str, maldideepkit.BaseSpectralClassifier or None, default=None
        When supplied, a name (see
        :func:`~maldideepkit.pretraining.register_backbone`) or an instance used as
        the spectral branch (head replaced by
        :class:`torch.nn.Identity` so the module returns features).
        When ``None``, the built-in MLP from ``spec_hidden_dims`` is
        used, or the architecture of the encoder given as ``init_from``.
    init_from : str, path-like, MaldiSpectrumEncoder or None, default=None
        Initialises the spectral branch from the trunk of this encoder at
        :meth:`fit`, and the spectral columns inherit its fitted
        preprocessing (checked against **this** classifier's
        ``standardize`` / ``input_transform`` / ``warping`` and
        ``n_spec_bins``, since the dual classifier owns the
        preprocessing). It may instead be set on the ``backbone=``
        template, but not on both. A path is loaded with the default
        ``trusted=False``; for a trusted checkpoint, pass
        ``MaldiSpectrumEncoder.load(path, trusted=True)``. A
        ``.warper.pkl`` is a pickle, read whatever ``trusted`` is: load
        checkpoints with a warper only from trusted sources.
    freeze_trunk_epochs : int, default=0
        Linear probing then fine-tuning (LP-FT): the spectral branch is
        frozen for the first epochs, while the side branch and the
        fusion head train. A non-zero value on the ``backbone=`` template
        is used when this one is ``0``.
    torch_compile : bool, default=False
        Train through ``torch.compile`` (the fitted module, checkpoints,
        and inference stay uncompiled); see
        :class:`~maldideepkit.BaseSpectralClassifier`.
    amp_dtype : {"float16", "bfloat16", "auto"}, default="float16"
        Autocast dtype of ``use_amp=True`` (``"bfloat16"`` cannot
        overflow and needs no ``GradScaler``); see
        :class:`~maldideepkit.BaseSpectralClassifier`.
    optimizer : None, str or callable, default=None
        See :class:`~maldideepkit.BaseSpectralClassifier`.
    scheduler : None, str or callable, default=None
        See :class:`~maldideepkit.BaseSpectralClassifier`.
    callbacks : Callback, sequence of Callback or None, default=None
        See :class:`~maldideepkit.BaseSpectralClassifier`.

    Notes
    -----
    Every parameter accepted by
    :class:`~maldideepkit.BaseSpectralClassifier` (e.g.
    ``learning_rate``, ``batch_size``, ``epochs``, ``warping``,
    ``device``, ``random_state``, ...) is forwarded to the base class.
    See its docstring for the full list.

    The class inherits everything from
    :class:`~maldideepkit.BaseDualSpectralClassifier`, which inherits
    from :class:`~maldideepkit.BaseSpectralClassifier`.
    :func:`~sklearn.base.clone` works on it.

    Spectrum-specific preprocessing stops at the ``n_spec_bins``
    boundary: ``warping``, ``standardize``, ``input_transform``,
    ``augment``, and ``cutmix_alpha`` act on the spectrum columns only,
    leaving the side block untouched (``mixup_alpha`` mixes both
    blocks with one shared ``lam``, which stays consistent with the
    mixed label). So ``input_transform="log1p"`` will not rewrite
    binary fingerprint bits, and a CutMix window cannot straddle the
    spectrum / side boundary. Preprocessing you *want* applied to the
    side block belongs in an upstream ``Pipeline`` step. See
    :class:`~maldideepkit.BaseDualSpectralClassifier` for the full
    breakdown.

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
        loss: Any = "cross_entropy",
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
        init_from: str | Path | Any | None = None,
        freeze_trunk_epochs: int = 0,
        torch_compile: bool = False,
        amp_dtype: str = "float16",
        optimizer: Any = None,
        scheduler: Any = None,
        callbacks: Any = None,
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
            init_from=init_from,
            freeze_trunk_epochs=freeze_trunk_epochs,
            torch_compile=torch_compile,
            amp_dtype=amp_dtype,
            optimizer=optimizer,
            scheduler=scheduler,
            callbacks=callbacks,
        )
        self.spec_hidden_dims = spec_hidden_dims
        self.spec_dropout = spec_dropout
        self.fp_hidden_dims = fp_hidden_dims
        self.fp_dropout = fp_dropout
        self.backbone = backbone

    def _resolve_init_encoder(self) -> Any | None:
        """Resolve ``init_from``, set here or on the ``backbone=`` template.

        The encoder is checked against this (outer) classifier's
        preprocessing config and ``n_spec_bins``, because the dual
        classifier owns the preprocessing of the spectral columns. Without
        a ``backbone=``, the spectral branch takes the architecture of the
        encoder (:attr:`spectral_template_`).
        """
        self._init_from_provenance = None
        self.__dict__.pop("spectral_template_", None)
        template = self._spectral_template()
        template_from = getattr(template, "init_from", None)
        if self.init_from is not None and template_from is not None:
            raise ValueError(
                "init_from is set both on this classifier and on its backbone= "
                "template: set it on one of them."
            )
        init_from = self.init_from if self.init_from is not None else template_from
        if init_from is None:
            return None
        encoder, provenance = _load_init_encoder(init_from)
        self._check_init_encoder(encoder, int(self.n_spec_bins))
        self._init_from_provenance = provenance
        if template is None:
            self.spectral_template_ = clone(encoder.backbone_template_)
        return encoder

    def _spectral_template(self) -> Any:
        """Return the ``backbone=`` template, a name being resolved to its class."""
        if isinstance(self.backbone, str):
            return get_backbone_class(self.backbone)()
        return self.backbone

    def _frozen_epochs(self) -> int:
        """Return this classifier's ``freeze_trunk_epochs``, else the template's."""
        own = int(self.freeze_trunk_epochs)
        if own != 0:
            return own
        return int(getattr(self._spectral_template(), "freeze_trunk_epochs", 0))

    def _hparam_dict(self) -> dict[str, Any]:
        """Store the architecture derived from the encoder as the ``backbone``."""
        params = super()._hparam_dict()
        derived = getattr(self, "spectral_template_", None)
        if self.backbone is None and derived is not None:
            params["backbone"] = _estimator_spec(derived)
        return params

    @classmethod
    def from_spectrum(
        cls, bin_width: int, input_dim: int, **overrides: Any
    ) -> MaldiDualBranchClassifier:
        """Construct a classifier for ``input_dim`` spectrum bins.

        A ``backbone`` given by name is built with the ``from_spectrum``
        of its class, so its architecture is scaled for ``bin_width``.

        Parameters
        ----------
        bin_width : int
            Bin width in Daltons.
        input_dim : int
            Number of spectrum bins (``n_spec_bins``).
        **overrides
            Any other constructor argument.

        Returns
        -------
        MaldiDualBranchClassifier
            An unfitted estimator.
        """
        backbone = overrides.get("backbone")
        if isinstance(backbone, str):
            overrides["backbone"] = get_backbone_class(backbone).from_spectrum(
                bin_width, input_dim
            )
        return super().from_spectrum(bin_width, input_dim, **overrides)

    def _trunk_target(self, model: nn.Module) -> tuple[nn.Module, str | None]:
        """Return the spectral branch; its final layer is already an Identity."""
        return model.inner.spectral, None

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
        template = self._spectral_template()
        if template is None:
            template = getattr(self, "spectral_template_", None)
        if template is None:
            return self._build_default_mlp(
                input_dim_spec,
                hidden_dims=tuple(self.spec_hidden_dims),
                dropout=float(self.spec_dropout),
            )

        # User backbone path: clone, force input_dim_/n_classes_, build,
        # then replace the head with Identity to expose features.
        if not isinstance(template, BaseSpectralClassifier):
            raise TypeError(
                f"backbone= must be a name or a BaseSpectralClassifier instance; "
                f"got {type(template).__name__}. Pass a name such as 'cnn' or "
                "an instance such as MaldiMLPClassifier()."
            )
        bb = clone(template)
        bb.input_dim_ = int(input_dim_spec)
        bb.n_classes_ = int(n_classes)
        full = bb._build_model()
        spec_out = self._find_last_linear_in_features(full, int(input_dim_spec))
        spec = self._replace_last_linear_with_identity(full, int(input_dim_spec))
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
    def _replace_last_linear_with_identity(
        module: nn.Module, input_dim: int | None = None
    ) -> nn.Module:
        """Replace the final layer of ``module`` with Identity (in place).

        Uses ``module.final_layer`` when the module defines it (the
        built-in backbones); otherwise the final :class:`nn.Linear` of a
        custom module. The lookup is
        :func:`~maldideepkit.base._trunk.trunk_prefix`, shared with the
        encoder and ``init_from``.
        """
        parent_path, _, name = trunk_prefix(module, input_dim).rpartition(".")
        parent = module if parent_path == "" else module.get_submodule(parent_path)
        setattr(parent, name, nn.Identity())
        return module

    @staticmethod
    def _find_last_linear_in_features(
        module: nn.Module, input_dim: int | None = None
    ) -> int:
        """Return the spectral feature width of ``module``.

        Uses ``module.feature_dim`` when the module defines it together
        with ``final_layer`` (the built-in backbones, whose final layer
        :meth:`_replace_last_linear_with_identity` swaps out); otherwise
        falls back to the ``in_features`` of the final :class:`nn.Linear`
        of ``module`` (custom modules), so the width always matches the
        layer that is actually replaced (see
        :func:`~maldideepkit.base._trunk.split_trunk`).
        """
        return split_trunk(module, input_dim).feature_dim
