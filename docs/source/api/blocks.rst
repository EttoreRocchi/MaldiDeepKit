Blocks module
=============

.. module:: maldideepkit.blocks

``maldideepkit.blocks`` re-exports every ``nn.Module`` behind the
classifiers from one import path; each name is the same class object
as in its family module.

Full backbones
--------------

The ``nn.Module`` subclasses that the classifiers train. Use them to
write your own training loop with the MaldiDeepKit architecture
defaults.

- :class:`~maldideepkit.attention.mlp.SpectralAttentionMLP`
- :class:`~maldideepkit.cnn.cnn.SpectralCNN1D`
- :class:`~maldideepkit.resnet.resnet.SpectralResNet1D`
- :class:`~maldideepkit.transformer.transformer.SpectralTransformer1D`

Composable primitives
---------------------

The smaller modules that make up the backbones, for mixing components
across families or building custom architectures.

- :class:`~maldideepkit.resnet.resnet.BasicBlock1D`
- :class:`~maldideepkit.transformer.transformer.TransformerBlock`
- :class:`~maldideepkit.transformer.transformer.MultiHeadSelfAttention`
- :class:`~maldideepkit.blocks.PatchEmbed1D`
- :class:`~maldideepkit.blocks.DropPath`

The other classes are documented on their family pages (:doc:`mlp`,
:doc:`cnn`, :doc:`resnet`, and :doc:`transformer`);
:class:`~maldideepkit.blocks.PatchEmbed1D` and
:class:`~maldideepkit.blocks.DropPath` are documented here:

.. autoclass:: maldideepkit.blocks.PatchEmbed1D
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: maldideepkit.blocks.DropPath
   :members:
   :undoc-members:
   :show-inheritance:

Examples
--------

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
