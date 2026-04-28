"""1-D ResNet classifier for binned MALDI-TOF spectra.

ResNet-18 residual-block template adapted to 1-D spectral input: a
stem Conv1D, four stages of BasicBlock1D pairs with strided
downsampling between stages, global average pooling, and a single
linear head.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .._bin_scaling import scale_odd_kernel
from ..base.classifier import BaseSpectralClassifier


class BasicBlock1D(nn.Module):
    """Two Conv1D layers with a residual shortcut.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int, default=1
        Stride of the first Conv1D. A stride ``!= 1`` (or a channel
        mismatch) triggers a 1x1 projection on the shortcut path.
    kernel_size : int, default=3
        Kernel size of both Conv1D layers.
    """

    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or in_channels != out_channels:
            self.shortcut: nn.Module = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the residual block to a ``(N, C, L)`` activation tensor."""
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity
        return self.relu(out)


class SpectralResNet1D(nn.Module):
    """1-D ResNet-18 style backbone with a linear classification head.

    Parameters
    ----------
    input_dim : int
        Number of input bins.
    n_classes : int, default=2
        Number of output logits.
    stem_channels : int, default=32
        Output channels of the initial ``Conv1d`` + pool stem.
    stage_channels : sequence of int, default=(64, 128, 256, 512)
        Output channels of each stage. The first stage keeps the stem
        resolution; later stages downsample by 2.
    blocks_per_stage : sequence of int, default=(2, 2, 2, 2)
        Number of BasicBlock1D pairs per stage.
    stem_kernel_size : int, default=7
        Kernel size of the stem Conv1D.
    stem_stride : int, default=1
        Stride of the stem Conv1D. Defaults to ``1`` (vs literal
        ResNet-18's ``2``) to preserve peak-scale features through the
        first stage on ~6000-bin spectra.
    block_kernel_size : int, default=7
        Kernel size inside every :class:`BasicBlock1D`. Widened from
        ResNet-18's 3 to give each block enough local context for
        peak-scale features.
    use_stem_pool : bool, default=False
        If ``True``, append the literal ResNet-18 ``MaxPool1d(kernel=3,
        stride=2)`` after the stem Conv. Defaults to ``False`` because
        the combined stem-Conv(stride=2) + MaxPool(stride=2) initial
        4x downsampling collapses peak-scale features. Set to ``True``
        to reproduce the literal ResNet-18 backbone.
    dropout : float, default=0.2
        Dropout applied after global average pooling.
    """

    def __init__(
        self,
        input_dim: int,
        n_classes: int = 2,
        stem_channels: int = 32,
        stage_channels: tuple[int, ...] = (64, 128, 256, 512),
        blocks_per_stage: tuple[int, ...] = (2, 2, 2, 2),
        stem_kernel_size: int = 7,
        stem_stride: int = 1,
        block_kernel_size: int = 7,
        use_stem_pool: bool = False,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if len(stage_channels) != len(blocks_per_stage):
            raise ValueError(
                "stage_channels and blocks_per_stage must have the same length."
            )
        stem_layers: list[nn.Module] = [
            nn.Conv1d(
                1,
                stem_channels,
                kernel_size=stem_kernel_size,
                stride=stem_stride,
                padding=stem_kernel_size // 2,
                bias=False,
            ),
            nn.BatchNorm1d(stem_channels),
            nn.ReLU(inplace=True),
        ]
        if use_stem_pool:
            stem_layers.append(nn.MaxPool1d(kernel_size=3, stride=2, padding=1))
        self.stem = nn.Sequential(*stem_layers)

        stages: list[nn.Module] = []
        prev = stem_channels
        for i, (ch, n_blocks) in enumerate(
            zip(stage_channels, blocks_per_stage, strict=True)
        ):
            stride = 1 if i == 0 else 2
            layer = [
                BasicBlock1D(prev, ch, stride=stride, kernel_size=block_kernel_size)
            ]
            layer += [
                BasicBlock1D(ch, ch, stride=1, kernel_size=block_kernel_size)
                for _ in range(n_blocks - 1)
            ]
            stages.append(nn.Sequential(*layer))
            prev = ch
        self.stages = nn.Sequential(*stages)

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(prev, n_classes)
        self.input_dim = input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``(batch, input_dim)`` to ``(batch, n_classes)`` logits."""
        x = x.unsqueeze(1)
        x = self.stem(x)
        x = self.stages(x)
        x = self.gap(x).squeeze(-1)
        x = self.dropout(x)
        return self.fc(x)


class MaldiResNetClassifier(BaseSpectralClassifier):
    """sklearn-compatible 1-D ResNet classifier for MALDI-TOF spectra.

    Parameters
    ----------
    stem_channels : int, default=32
        Output channels of the stem.
    stage_channels : sequence of int, default=(64, 128, 256, 512)
        Output channels of each residual stage.
    blocks_per_stage : sequence of int, default=(2, 2, 2, 2)
        Number of residual blocks per stage (ResNet-18 topology).
    stem_kernel_size : int, default=7
        Kernel size of the stem Conv1D. Calibrated for ``bin_width=3``;
        :meth:`from_spectrum` scales it for other bin widths.
    stem_stride : int, default=1
        Stride of the stem Conv1D. Defaults to ``1`` so the first
        residual stage sees full-resolution input.
    block_kernel_size : int, default=7
        Kernel size inside every :class:`BasicBlock1D`. Widened from
        ResNet-18's 3 so each block has enough local context for
        peak-scale features.
    use_stem_pool : bool, default=False
        If ``True``, append the literal ResNet-18 MaxPool after the
        stem Conv. Defaults to ``False`` because the combined
        stem(stride=2) + MaxPool(stride=2) 4x downsampling is too
        aggressive for MALDI-TOF peak structure. Set to ``True`` to
        reproduce the literal ResNet-18 backbone.
    dropout : float, default=0.2
        Dropout before the final linear layer.
    **kwargs
        Forwarded to :class:`~maldideepkit.base.classifier.BaseSpectralClassifier`.

    Notes
    -----
    Defaults deviate from literal ResNet-18 (He et al., 2016) in three
    ways for MALDI-TOF: ``stem_stride=1`` (was 2),
    ``block_kernel_size=7`` (was 3), and ``use_stem_pool=False`` (was
    True). The literal backbone is reproducible via ``stem_stride=2,
    block_kernel_size=3, use_stem_pool=True``.

    The final head is a ``Linear(stage_channels[-1], n_classes)``
    after global average pooling, so the parameter count is
    **independent of ``input_dim``**.

    Examples
    --------
    >>> import numpy as np
    >>> from maldideepkit import MaldiResNetClassifier
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((32, 512)).astype("float32")
    >>> y = rng.integers(0, 2, size=32)
    >>> clf = MaldiResNetClassifier(
    ...     epochs=2, batch_size=8, stage_channels=(16, 32),
    ...     blocks_per_stage=(1, 1), random_state=0
    ... ).fit(X, y)
    >>> clf.predict(X).shape
    (32,)
    """

    def __init__(
        self,
        input_dim: int | None = None,
        n_classes: int = 2,
        stem_channels: int = 32,
        stage_channels: tuple[int, ...] = (64, 128, 256, 512),
        blocks_per_stage: tuple[int, ...] = (2, 2, 2, 2),
        stem_kernel_size: int = 7,
        stem_stride: int = 1,
        block_kernel_size: int = 7,
        use_stem_pool: bool = False,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        grad_clip_norm: float | None = 1.0,
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
        warmup_epochs: int = 5,
        standardize: bool = False,
        input_transform: str | None = "log1p",
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
        self.stem_channels = stem_channels
        self.stage_channels = stage_channels
        self.blocks_per_stage = blocks_per_stage
        self.stem_kernel_size = stem_kernel_size
        self.stem_stride = stem_stride
        self.block_kernel_size = block_kernel_size
        self.use_stem_pool = use_stem_pool
        self.dropout = dropout

    def _build_model(self) -> nn.Module:
        return SpectralResNet1D(
            input_dim=self.input_dim_,
            n_classes=self.n_classes_,
            stem_channels=int(self.stem_channels),
            stage_channels=tuple(self.stage_channels),
            blocks_per_stage=tuple(self.blocks_per_stage),
            stem_kernel_size=int(self.stem_kernel_size),
            stem_stride=int(self.stem_stride),
            block_kernel_size=int(self.block_kernel_size),
            use_stem_pool=bool(self.use_stem_pool),
            dropout=float(self.dropout),
        )

    @classmethod
    def from_spectrum(
        cls, bin_width: int, input_dim: int, **overrides
    ) -> "MaldiResNetClassifier":
        """Construct a peak-friendly ResNet for a given spectrum layout.

        Scales ``stem_kernel_size`` inversely with ``bin_width``
        relative to the package reference (``bin_width=3``,
        ``stem_kernel_size=7``). Also sets ``stem_stride=1`` and
        ``use_stem_pool=False`` (the class defaults). Any keyword in
        ``**overrides`` wins over the auto-selected values.
        """
        kwargs: dict[str, Any] = {
            "input_dim": input_dim,
            "stem_kernel_size": scale_odd_kernel(bin_width),
            "stem_stride": 1,
            "use_stem_pool": False,
        }
        kwargs.update(overrides)
        return cls(**kwargs)
