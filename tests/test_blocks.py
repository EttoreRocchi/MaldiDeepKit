"""Tests for the ``maldideepkit.blocks`` re-export module.

The module is a convenience namespace that re-exports every
``nn.Module`` primitive backing the classifier catalog. The tests
verify:

- Every symbol listed in ``blocks.__all__`` is importable.
- Each re-exported symbol resolves to the same class object as the
  canonical per-family path (no accidental shadowing).
- A representative backbone and a representative composable primitive
  work when instantiated through the new import path.
"""

from __future__ import annotations

import importlib

import pytest
import torch

from maldideepkit import blocks

# Map each re-exported name to its canonical defining module.
CANONICAL = {
    "SpectralAttentionMLP": "maldideepkit.attention.mlp",
    "SpectralCNN1D": "maldideepkit.cnn.cnn",
    "SpectralResNet1D": "maldideepkit.resnet.resnet",
    "BasicBlock1D": "maldideepkit.resnet.resnet",
    "SpectralTransformer1D": "maldideepkit.transformer.transformer",
    "TransformerBlock": "maldideepkit.transformer.transformer",
    "MultiHeadSelfAttention": "maldideepkit.transformer.transformer",
    # PatchEmbed1D and DropPath live in the private ``_blocks`` module.
    "PatchEmbed1D": "maldideepkit._blocks",
    "DropPath": "maldideepkit._blocks",
}


class TestBlocksModule:
    def test_all_names_in_module(self):
        """Every name in ``__all__`` is actually an attribute."""
        for name in blocks.__all__:
            assert hasattr(blocks, name), f"blocks.{name} missing"

    @pytest.mark.parametrize("name,canonical_module", list(CANONICAL.items()))
    def test_reexport_is_canonical_object(self, name, canonical_module):
        """Re-exported class is the *same* object as the family-module class."""
        mod = importlib.import_module(canonical_module)
        canonical = getattr(mod, name)
        reexported = getattr(blocks, name)
        assert reexported is canonical, (
            f"blocks.{name} ({reexported}) "
            f"is not {canonical_module}.{name} ({canonical})"
        )

    def test_backbone_instantiates_via_blocks(self):
        """A full backbone can be built entirely from the blocks namespace."""
        model = blocks.SpectralTransformer1D(
            input_dim=128,
            embed_dim=16,
            depth=2,
            num_heads=2,
            patch_size=2,
            head_dim=8,
        )
        out = model(torch.randn(2, 128))
        assert out.shape == (2, 2)

    def test_composable_primitive_via_blocks(self):
        """A single composable block can be built from the blocks namespace."""
        block = blocks.TransformerBlock(dim=16, num_heads=2)
        x = torch.randn(2, 16, 16)
        assert block(x).shape == x.shape

    def test_patch_embed_via_blocks(self):
        pe = blocks.PatchEmbed1D(patch_size=2, in_channels=1, embed_dim=16)
        x = torch.randn(2, 1, 32)
        assert pe(x).shape == (2, 16, 16)
