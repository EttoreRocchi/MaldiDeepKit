"""1-D convolutional classifier for binned MALDI-TOF spectra.

A stack of ``Conv1d -> BatchNorm -> ReLU -> MaxPool`` blocks followed
by a flatten + dense classification head.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn

from .._bin_scaling import scale_odd_kernel
from ..base.classifier import BaseSpectralClassifier


def _broadcast(value: int | Sequence[int], n: int, name: str) -> tuple[int, ...]:
    """Return a length-``n`` tuple: scalars are broadcast, sequences validated."""
    if isinstance(value, int):
        if value <= 0:
            raise ValueError(f"{name} must be a positive integer; got {value}.")
        return (value,) * n
    out = tuple(int(v) for v in value)
    if len(out) != n:
        raise ValueError(
            f"{name} has length {len(out)} but must have length {n} to match channels."
        )
    if any(v <= 0 for v in out):
        raise ValueError(f"{name} must contain only positive integers; got {out}.")
    return out


class _ConvBlock(nn.Module):
    """One Conv1D + BN + ReLU + MaxPool + Dropout block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        pool_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(pool_size),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SpectralCNN1D(nn.Module):
    """Stack of Conv1D blocks with a dense classification head.

    Parameters
    ----------
    input_dim : int
        Number of input bins.
    n_classes : int, default=2
        Number of output logits.
    channels : sequence of int, default=(32, 64, 128, 128)
        Output channels of each convolutional block.
    kernel_size : int or sequence of int, default=7
        Kernel size per block. A scalar is broadcast to every block;
        a sequence must have the same length as ``channels``.
    pool_size : int or sequence of int, default=2
        Pool factor per block. A scalar is broadcast to every block;
        a sequence must have the same length as ``channels``.
    head_dim : int, default=128
        Width of the single hidden dense layer.
    dropout : float, default=0.3
        Dropout applied inside every block and before the output layer.

    Attributes
    ----------
    feature_dim : int
        Width of :meth:`forward_features` (``head_dim``).

    Notes
    -----
    Input tensors have shape ``(batch, input_dim)`` and are unsqueezed to
    ``(batch, 1, input_dim)`` internally.
    """

    def __init__(
        self,
        input_dim: int,
        n_classes: int = 2,
        channels: tuple[int, ...] = (32, 64, 128, 128),
        kernel_size: int | Sequence[int] = 7,
        pool_size: int | Sequence[int] = 2,
        head_dim: int = 128,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        n_blocks = len(channels)
        kernels = _broadcast(kernel_size, n_blocks, "kernel_size")
        pools = _broadcast(pool_size, n_blocks, "pool_size")

        blocks: list[nn.Module] = []
        prev = 1
        length = input_dim
        for out_ch, k, p in zip(channels, kernels, pools, strict=True):
            blocks.append(_ConvBlock(prev, out_ch, k, p, dropout))
            prev = out_ch
            length //= p
            if length <= 0:
                raise ValueError(
                    f"input_dim={input_dim} is too small for the given pool "
                    f"schedule {pools} (block {len(blocks)} would have 0 length)."
                )
        self.backbone = nn.Sequential(*blocks)
        self.flat_dim = prev * length
        self.head = nn.Sequential(
            nn.Linear(self.flat_dim, head_dim),
            nn.LayerNorm(head_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(head_dim, n_classes),
        )
        self.feature_dim = head_dim

    @property
    def final_layer(self) -> nn.Module:
        """Output layer mapping :meth:`forward_features` to the logits."""
        return self.head[-1]

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``(batch, input_dim)`` to ``(batch, feature_dim)`` features."""
        x = x.unsqueeze(1)
        feat = self.backbone(x).flatten(1)
        for layer in list(self.head)[:-1]:
            feat = layer(feat)
        return feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``(batch, input_dim)`` to ``(batch, n_classes)`` logits."""
        return self.head[-1](self.forward_features(x))


class MaldiCNNClassifier(BaseSpectralClassifier):
    """sklearn-compatible 1-D CNN classifier for MALDI-TOF spectra.

    Parameters
    ----------
    channels : sequence of int, default=(32, 64, 128, 128)
        Output channels of each convolutional block. The effective spatial
        resolution is divided by the corresponding ``pool_size`` after
        every block.
    kernel_size : int or sequence of int, default=7
        Kernel size per block. A scalar is broadcast; a sequence must
        match the length of ``channels``. The default is calibrated for
        ``bin_width=3``; :meth:`from_spectrum` scales it for other bin
        widths.
    pool_size : int or sequence of int, default=2
        Pool factor per block. Accepts scalar or per-block sequence.
    head_dim : int, default=128
        Width of the hidden dense layer.
    dropout : float, default=0.3
        Dropout applied inside every block and before the output layer.
    init_from : str, path-like, MaldiSpectrumEncoder or None, default=None
        Self-supervised encoder (checkpoint path or fitted instance)
        whose trunk initialises the model, with its fitted
        preprocessing inherited; see
        :class:`~maldideepkit.BaseSpectralClassifier`.
    freeze_trunk_epochs : int, default=0
        Number of initial epochs training only the final layer
        (LP-FT); see
        :class:`~maldideepkit.BaseSpectralClassifier`.
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
    :class:`~maldideepkit.BaseSpectralClassifier`
    (e.g. ``learning_rate``, ``batch_size``, ``epochs``, ``warping``,
    ``calibrate_temperature``, ``device``, ``random_state``, ...) is
    forwarded to the base class. See its docstring for the full list.

    The flat dense head scales linearly with ``input_dim``; prefer
    :class:`~maldideepkit.MaldiResNetClassifier` or
    :class:`~maldideepkit.MaldiTransformerClassifier` for a head width
    independent of the input resolution.

    See :meth:`from_spectrum` for a factory that scales
    ``kernel_size`` for a given ``(bin_width, input_dim)`` layout.

    Examples
    --------
    >>> import numpy as np
    >>> from maldideepkit import MaldiCNNClassifier
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((32, 256)).astype("float32")
    >>> y = rng.integers(0, 2, size=32)
    >>> clf = MaldiCNNClassifier(epochs=2, batch_size=8, random_state=0).fit(X, y)
    >>> clf.predict(X).shape
    (32,)

    Per-block kernel progression:

    >>> clf = MaldiCNNClassifier(
    ...     channels=(32, 64, 128, 128),
    ...     kernel_size=(11, 7, 5, 3),
    ... )
    """

    def __init__(
        self,
        input_dim: int | None = None,
        n_classes: int = 2,
        *,
        channels: tuple[int, ...] = (32, 64, 128, 128),
        kernel_size: int | Sequence[int] = 7,
        pool_size: int | Sequence[int] = 2,
        head_dim: int = 128,
        dropout: float = 0.3,
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
        self.channels = channels
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.head_dim = head_dim
        self.dropout = dropout

    def _arch_params(self) -> dict[str, Any]:
        return {
            "channels": tuple(self.channels),
            "kernel_size": self.kernel_size,
            "pool_size": self.pool_size,
            "head_dim": int(self.head_dim),
            "dropout": float(self.dropout),
        }

    def _build_model(self) -> nn.Module:
        return SpectralCNN1D(
            input_dim=self.input_dim_,
            n_classes=self.n_classes_,
            **self._arch_params(),
        )

    @classmethod
    def from_spectrum(
        cls, bin_width: int, input_dim: int, **overrides
    ) -> "MaldiCNNClassifier":
        """Construct a classifier with ``kernel_size`` scaled for ``bin_width``.

        Scales ``kernel_size`` inversely with ``bin_width`` relative to
        the package reference (``bin_width=3``, ``kernel_size=7``).

        Parameters
        ----------
        bin_width : int
            Bin width in Daltons (e.g. 3 for the MaldiAMRKit default).
        input_dim : int
            Number of bins in the input.
        **overrides
            Any other constructor argument; it wins over the scaled
            value.

        Returns
        -------
        MaldiCNNClassifier
            An unfitted estimator.
        """
        kwargs: dict[str, Any] = {
            "input_dim": input_dim,
            "kernel_size": scale_odd_kernel(bin_width),
        }
        kwargs.update(overrides)
        return cls(**kwargs)
