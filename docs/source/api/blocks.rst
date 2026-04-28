Blocks Module
=============

.. currentmodule:: maldideepkit.blocks

One-stop import path for every ``nn.Module`` primitive backing the
classifier catalog. Symbols are re-exported from their per-family
defining modules (so the exact same class object is accessible through
either path).

Full backbones
--------------

The ``nn.Module`` subclasses the sklearn-compatible classifier wrappers
drive. Use these if you want to own the training loop but keep
MaldiDeepKit's architecture defaults.

- :class:`SpectralAttentionMLP`
- :class:`SpectralCNN1D`
- :class:`SpectralResNet1D`
- :class:`SpectralTransformer1D`

Composable primitives
---------------------

Smaller ``nn.Module``'s that the backbones are composed of. Useful for
mixing components across families or building custom architectures.

- :class:`BasicBlock1D`
- :class:`TransformerBlock`
- :class:`MultiHeadSelfAttention`
- :class:`PatchEmbed1D`
- :class:`DropPath`

Each link above resolves to the autodoc entry on the corresponding
per-family page (``mlp``, ``cnn``, ``resnet``, ``transformer``), so
there is a single source of truth for every class's documentation.

Example
-------

.. code-block:: python

    import torch
    from maldideepkit.blocks import (
        SpectralTransformer1D, TransformerBlock, PatchEmbed1D,
    )

    # Full backbone:
    backbone = SpectralTransformer1D(input_dim=6000, depth=6)

    # Or compose a single transformer block into your own stack:
    block = TransformerBlock(dim=128, num_heads=4)
    tokens = torch.randn(2, 1500, 128)
    out = block(tokens)
    assert out.shape == (2, 1500, 128)
