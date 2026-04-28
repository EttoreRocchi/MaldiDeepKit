"""1-D Vision Transformer for binned MALDI-TOF spectra.

A plain ViT backbone adapted to 1-D spectra:
non-overlapping patch embedding, learned positional embedding,
pre-LayerNorm residual blocks with LayerScale and stochastic depth,
global self-attention in every block, and mean-pool aggregation by
default (CLS token optional).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .._blocks import DropPath, PatchEmbed1D
from ..base.classifier import BaseSpectralClassifier


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with QK-norm + memory-efficient SDPA.

    QK-normalization applies a per-head :class:`~torch.nn.LayerNorm`
    to query and key tensors before the scaled-dot-product, bounding
    the softmax denominator regardless of input scale. Always on
    (universal stability improvement with negligible compute overhead).

    Parameters
    ----------
    dim : int
        Token embedding dimension. Must be divisible by ``num_heads``.
    num_heads : int
        Number of attention heads.
    attention_dropout : float, default=0.0
        Dropout applied inside the attention kernel during training.
    proj_dropout : float, default=0.0
        Dropout applied to the final projection.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        attention_dropout: float = 0.0,
        proj_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}.")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.q_norm = nn.LayerNorm(self.head_dim)
        self.k_norm = nn.LayerNorm(self.head_dim)
        self.attention_dropout = float(attention_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Global self-attention on ``(B, N, C)`` tokens.

        ``key_padding_mask`` (optional, shape ``(B, N)``, dtype bool):
        ``True`` = real token, ``False`` = padding to ignore.
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(dim=0)
        q = self.q_norm(q)
        k = self.k_norm(k)
        attn_mask: torch.Tensor | None = None
        if key_padding_mask is not None:
            attn_mask = torch.zeros((B, 1, 1, N), dtype=q.dtype, device=q.device)
            attn_mask = attn_mask.masked_fill(
                ~key_padding_mask[:, None, None, :], float("-inf")
            )
        dropout_p = self.attention_dropout if self.training else 0.0
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=False
        )
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(out))


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with LayerScale and stochastic depth.

    Residual pattern::

        x = x + drop_path(γ_1 * Attn(LN(x)))
        x = x + drop_path(γ_2 * MLP(LN(x)))

    ``γ_*`` are per-channel learnable scales initialised near zero so
    every block starts as an identity map.

    Parameters
    ----------
    dim : int
        Token dimension.
    num_heads : int
        Attention heads.
    mlp_ratio : int, default=4
        MLP hidden-dim multiplier.
    dropout : float, default=0.0
        MLP dropout.
    attention_dropout : float, default=0.0
        Attention-matrix dropout.
    drop_path : float, default=0.0
        Stochastic-depth probability for this block's residuals.
    layerscale_init : float, default=1e-4
        Initial value of the LayerScale gammas. Set to ``None`` to
        disable LayerScale entirely.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        drop_path: float = 0.0,
        layerscale_init: float | None = 1e-4,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(
            dim, num_heads, attention_dropout=attention_dropout, proj_dropout=dropout
        )
        self.drop_path1 = DropPath(drop_path)

        self.norm2 = nn.LayerNorm(dim)
        hidden = int(mlp_ratio * dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )
        self.drop_path2 = DropPath(drop_path)

        self.use_layerscale = layerscale_init is not None
        if self.use_layerscale:
            self.gamma1 = nn.Parameter(torch.full((dim,), float(layerscale_init)))
            self.gamma2 = nn.Parameter(torch.full((dim,), float(layerscale_init)))

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run pre-norm attention + MLP residual sub-blocks with optional LayerScale."""
        attn_out = self.attn(self.norm1(x), key_padding_mask=key_padding_mask)
        mlp_out_src = self.norm2(x)
        if self.use_layerscale:
            attn_out = attn_out * self.gamma1
        x = x + self.drop_path1(attn_out)
        mlp_out = self.mlp(mlp_out_src)
        if self.use_layerscale:
            mlp_out = mlp_out * self.gamma2
        x = x + self.drop_path2(mlp_out)
        return x


class SpectralTransformer1D(nn.Module):
    """1-D Vision Transformer backbone for binned spectra.

    Parameters
    ----------
    input_dim : int
        Number of input bins.
    n_classes : int, default=2
        Number of output logits.
    patch_size : int, default=4
        Non-overlapping patch width. Token count is
        ``ceil(input_dim / patch_size)``.
    embed_dim : int, default=64
        Token embedding dimension.
    depth : int, default=6
        Number of transformer blocks.
    num_heads : int, default=4
        Attention heads per block. ``embed_dim`` must be divisible by
        ``num_heads``.
    mlp_ratio : int, default=4
        MLP hidden-dim multiplier.
    dropout : float, default=0.1
        MLP dropout applied inside every block and before the head.
    attention_dropout : float, default=0.0
        Attention-matrix dropout.
    drop_path_rate : float, default=0.1
        End-of-stack stochastic-depth rate. Linearly interpolated
        from ``0`` at block 0 to ``drop_path_rate`` at the final block.
    layerscale_init : float or None, default=1e-4
        LayerScale initial value. ``None`` disables LayerScale.
    pool : {"cls", "mean"}, default="mean"
        Aggregation strategy for classification. ``"mean"`` averages
        over patch tokens (more robust on small data); ``"cls"``
        prepends a learned token and uses its output.
    head_dim : int, default=128
        Width of the hidden dense layer in the classification head.
    """

    def __init__(
        self,
        input_dim: int,
        n_classes: int = 2,
        patch_size: int = 4,
        embed_dim: int = 64,
        depth: int = 6,
        num_heads: int = 4,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        attention_dropout: float = 0.0,
        drop_path_rate: float = 0.1,
        layerscale_init: float | None = 1e-4,
        pool: str = "mean",
        head_dim: int = 128,
    ) -> None:
        super().__init__()
        if pool not in {"mean", "cls"}:
            raise ValueError(f"pool must be 'mean' or 'cls'; got {pool!r}.")
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}."
            )
        if depth < 1:
            raise ValueError(f"depth must be >= 1; got {depth!r}.")

        self.pool = pool
        self.patch_size = patch_size

        self.embed = PatchEmbed1D(
            patch_size=patch_size, in_channels=1, embed_dim=embed_dim
        )

        n_tokens = -(-input_dim // patch_size)
        self.n_tokens = n_tokens

        self.cls_token: nn.Parameter | None
        if pool == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
            pos_len = n_tokens + 1
        else:
            self.register_parameter("cls_token", None)
            pos_len = n_tokens
        self.pos_embed = nn.Parameter(torch.zeros(1, pos_len, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.pos_drop = nn.Dropout(dropout)

        dpr = [float(x) for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    drop_path=dpr[i],
                    layerscale_init=layerscale_init,
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, head_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``(batch, input_dim)`` to ``(batch, n_classes)`` logits.

        Notes
        -----
        Inputs whose length is not a multiple of ``patch_size`` are
        right-padded with zeros before the patch embedding.
        """
        x = x.unsqueeze(1)
        pad = (-x.shape[-1]) % self.patch_size
        if pad:
            x = torch.nn.functional.pad(x, (0, pad))
        tokens = self.embed(x)
        if self.cls_token is not None:
            cls = self.cls_token.expand(tokens.shape[0], -1, -1)
            tokens = torch.cat([cls, tokens], dim=1)
        tokens = self.pos_drop(tokens + self.pos_embed[:, : tokens.shape[1]])
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        if self.pool == "cls":
            pooled = tokens[:, 0]
        else:
            start = 1 if self.cls_token is not None else 0
            pooled = tokens[:, start:].mean(dim=1)
        return self.head(pooled)


class MaldiTransformerClassifier(BaseSpectralClassifier):
    """sklearn-compatible 1-D ViT classifier for MALDI-TOF spectra.

    Parameters
    ----------
    patch_size : int, default=4
        Patch size of the initial Conv1D embedding. Token count is
        ``ceil(input_dim / patch_size)``.
    embed_dim : int, default=64
        Token embedding dimension. Must be divisible by ``num_heads``.
    depth : int, default=6
        Number of transformer blocks.
    num_heads : int, default=4
        Attention heads per block.
    mlp_ratio : int, default=4
        MLP hidden-dim multiplier inside each block.
    dropout : float, default=0.1
        MLP dropout applied inside every block and before the head.
    attention_dropout : float, default=0.0
        Attention-matrix dropout.
    drop_path_rate : float, default=0.1
        Linearly ramped stochastic-depth rate (0 at block 0, this
        value at the final block).
    layerscale_init : float or None, default=1e-4
        LayerScale initial value. ``None`` disables LayerScale.
    pool : {"mean", "cls"}, default="mean"
        Token aggregation for classification.
    head_dim : int, default=128
        Width of the hidden dense layer in the classification head.
    **kwargs
        Forwarded to :class:`~maldideepkit.base.classifier.BaseSpectralClassifier`.

    Notes
    -----
    Transformer training recipe baked in as defaults: ``lr=3e-4``,
    ``weight_decay=0.05``, ``grad_clip_norm=1.0``, ``warmup_epochs=5``.

    Examples
    --------
    >>> import numpy as np
    >>> from maldideepkit import MaldiTransformerClassifier
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((32, 256)).astype("float32")
    >>> y = rng.integers(0, 2, size=32)
    >>> clf = MaldiTransformerClassifier(
    ...     epochs=2, batch_size=8, embed_dim=32, depth=2,
    ...     num_heads=2, patch_size=2, random_state=0,
    ... ).fit(X, y)
    >>> clf.predict(X).shape
    (32,)
    """

    def __init__(
        self,
        input_dim: int | None = None,
        n_classes: int = 2,
        patch_size: int = 4,
        embed_dim: int = 64,
        depth: int = 6,
        num_heads: int = 4,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        attention_dropout: float = 0.0,
        drop_path_rate: float = 0.1,
        layerscale_init: float | None = 1e-4,
        pool: str = "mean",
        head_dim: int = 128,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.05,
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
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.drop_path_rate = drop_path_rate
        self.layerscale_init = layerscale_init
        self.pool = pool
        self.head_dim = head_dim

    def _build_model(self) -> nn.Module:
        return SpectralTransformer1D(
            input_dim=self.input_dim_,
            n_classes=self.n_classes_,
            patch_size=int(self.patch_size),
            embed_dim=int(self.embed_dim),
            depth=int(self.depth),
            num_heads=int(self.num_heads),
            mlp_ratio=int(self.mlp_ratio),
            dropout=float(self.dropout),
            attention_dropout=float(self.attention_dropout),
            drop_path_rate=float(self.drop_path_rate),
            layerscale_init=self.layerscale_init,
            pool=str(self.pool),
            head_dim=int(self.head_dim),
        )

    @classmethod
    def from_spectrum(
        cls, bin_width: int, input_dim: int, **overrides
    ) -> "MaldiTransformerClassifier":
        """Construct a classifier for a given ``(bin_width, input_dim)`` layout.

        The transformer is architecturally scale-agnostic, so this
        factory only forwards ``input_dim`` and any ``**overrides``.
        Provided for API symmetry with the other classifiers.
        """
        del bin_width
        kwargs: dict[str, Any] = {"input_dim": input_dim}
        kwargs.update(overrides)
        return cls(**kwargs)
