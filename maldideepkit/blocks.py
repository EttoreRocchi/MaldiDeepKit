"""One-stop import path for every MaldiDeepKit ``nn.Module`` primitive.

Re-exports the full backbones and composable primitives under a single
namespace so users embedding components into their own networks don't
have to know the per-family layout.

Examples
--------
>>> import torch
>>> from maldideepkit.blocks import SpectralTransformer1D, TransformerBlock
>>> backbone = SpectralTransformer1D(input_dim=6000, depth=6)
>>> block = TransformerBlock(dim=128, num_heads=4)
>>> tokens = torch.randn(2, 1500, 128)
>>> out = block(tokens)           # (2, 1500, 128)
"""

from __future__ import annotations

from ._blocks import DropPath, PatchEmbed1D
from .attention.mlp import SpectralAttentionMLP
from .cnn.cnn import SpectralCNN1D
from .resnet.resnet import BasicBlock1D, SpectralResNet1D
from .transformer.transformer import (
    MultiHeadSelfAttention,
    SpectralTransformer1D,
    TransformerBlock,
)

__all__ = [
    "SpectralAttentionMLP",
    "SpectralCNN1D",
    "SpectralResNet1D",
    "BasicBlock1D",
    "SpectralTransformer1D",
    "TransformerBlock",
    "MultiHeadSelfAttention",
    "PatchEmbed1D",
    "DropPath",
]
