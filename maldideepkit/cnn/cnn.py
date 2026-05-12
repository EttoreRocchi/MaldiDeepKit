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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``(batch, input_dim)`` to ``(batch, n_classes)`` logits."""
        x = x.unsqueeze(1)
        feat = self.backbone(x)
        return self.head(feat.flatten(1))


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

    Notes
    -----
    Every parameter accepted by
    :class:`~maldideepkit.base.classifier.BaseSpectralClassifier`
    (e.g. ``learning_rate``, ``batch_size``, ``epochs``, ``warping``,
    ``calibrate_temperature``, ``device``, ``random_state``, ...) is
    forwarded to the base class. See its docstring for the full list.

    The flat dense head scales linearly with ``input_dim``; prefer
    :class:`~maldideepkit.MaldiResNetClassifier` or
    :class:`~maldideepkit.MaldiTransformerClassifier` if you want
    a head width that's independent of the input resolution.

    See :func:`from_spectrum` for a factory that auto-scales
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
        channels: tuple[int, ...] = (32, 64, 128, 128),
        kernel_size: int | Sequence[int] = 7,
        pool_size: int | Sequence[int] = 2,
        head_dim: int = 128,
        dropout: float = 0.3,
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
        self.channels = channels
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.head_dim = head_dim
        self.dropout = dropout

    def _build_model(self) -> nn.Module:
        return SpectralCNN1D(
            input_dim=self.input_dim_,
            n_classes=self.n_classes_,
            channels=tuple(self.channels),
            kernel_size=self.kernel_size,
            pool_size=self.pool_size,
            head_dim=int(self.head_dim),
            dropout=float(self.dropout),
        )

    @classmethod
    def from_spectrum(
        cls, bin_width: int, input_dim: int, **overrides
    ) -> "MaldiCNNClassifier":
        """Construct a classifier with ``kernel_size`` scaled for ``bin_width``.

        Scales ``kernel_size`` inversely with ``bin_width`` relative to
        the package reference (``bin_width=3``, ``kernel_size=7``).
        Any keyword in ``**overrides`` wins over the auto-scaled value.

        Parameters
        ----------
        bin_width : int
            Bin width in Daltons (e.g. 3 for the MaldiAMRKit default,
            6 for coarser binning).
        input_dim : int
            Number of bins in the input. Stored on the classifier for
            shape validation.
        **overrides
            Any additional keyword arguments override the scaled defaults.

        Returns
        -------
        MaldiCNNClassifier
            An unfitted estimator with ``kernel_size`` scaled for the
            given ``bin_width``.
        """
        kwargs: dict[str, Any] = {
            "input_dim": input_dim,
            "kernel_size": scale_odd_kernel(bin_width),
        }
        kwargs.update(overrides)
        return cls(**kwargs)
