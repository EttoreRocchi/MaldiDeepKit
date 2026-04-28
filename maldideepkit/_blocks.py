"""Shared ``nn.Module`` primitives used by more than one architecture."""

from __future__ import annotations

import torch
from einops import rearrange
from torch import nn


class PatchEmbed1D(nn.Module):
    """1-D patch embedding via a strided Conv1D + LayerNorm.

    Maps ``(B, C_in, L)`` to ``(B, L // patch_size, embed_dim)`` by
    convolving with a non-overlapping kernel of width ``patch_size``
    and stride ``patch_size``.

    Parameters
    ----------
    patch_size : int
        Non-overlapping patch width (and stride).
    in_channels : int
        Input channels. ``1`` for raw binned spectra after an
        ``unsqueeze(1)``.
    embed_dim : int
        Output embedding dimension.
    """

    def __init__(self, patch_size: int, in_channels: int, embed_dim: int) -> None:
        super().__init__()
        self.proj = nn.Conv1d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``(B, C, L)`` to ``(B, L // patch_size, embed_dim)``."""
        x = self.proj(x)
        x = rearrange(x, "b c l -> b l c")
        return self.norm(x)


class DropPath(nn.Module):
    """Stochastic depth (per-sample residual drop).

    During training, zero out a fraction ``drop_prob`` of samples
    uniformly at random and rescale the survivors by
    ``1 / (1 - drop_prob)`` so the expectation is unchanged. At
    inference time or when ``drop_prob == 0`` this is a no-op.

    Parameters
    ----------
    drop_prob : float, default=0.0
        Probability of dropping a sample's residual. Must be in
        ``[0, 1)``.
    generator : torch.Generator or None, default=None
        Optional explicit RNG for the Bernoulli mask. When ``None``
        PyTorch's global generator is used.
    """

    def __init__(
        self,
        drop_prob: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> None:
        super().__init__()
        if not 0.0 <= drop_prob < 1.0:
            raise ValueError(f"drop_prob must be in [0, 1); got {drop_prob!r}.")
        self.drop_prob = float(drop_prob)
        self.generator = generator

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep_prob, generator=self.generator)
        return x * mask / keep_prob

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob}"
