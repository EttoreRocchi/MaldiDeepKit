Transformer module
==================

1-D Vision Transformer (ViT) adapted to MALDI-TOF binned spectra:
non-overlapping ``Conv1d`` patch embedding, learned positional
embedding, optional ``[CLS]`` aggregation token, a stack of pre-norm
multi-head self-attention blocks with MLP + LayerScale + stochastic
depth, and a small MLP classification head over the pooled representation.

**Why a plain ViT for MALDI?** Every token attends to every other
token in every block, so widely separated m/z peaks interact directly
at layer 1 rather than through several stages of local-window merging.

**Design deviations from the canonical ImageNet ViT.**

- ``embed_dim=64`` and ``depth=6`` by default (literature ViT-S is
  ``embed_dim=384, depth=12``): the smaller model suits MALDI-TOF
  cohorts of a few thousand spectra, where data efficiency matters
  more than capacity.
- LayerScale is **on by default** (``layerscale_init=1e-4``), so each
  block starts close to an identity map.
- CLS pooling is opt-in; the default is a mean pool over patch tokens,
  which is more robust on small data (no single aggregator token to
  overfit).
- The training recipe is in the defaults: ``learning_rate=3e-4``,
  ``weight_decay=0.05``, ``grad_clip_norm=1.0``, and
  ``warmup_epochs=5``. Without it the attention layers diverge on the
  first few batches.

MaldiTransformerClassifier
--------------------------

.. autoclass:: maldideepkit.MaldiTransformerClassifier
   :members:
   :undoc-members:
   :show-inheritance:

SpectralTransformer1D
---------------------

.. autoclass:: maldideepkit.transformer.transformer.SpectralTransformer1D
   :members:
   :undoc-members:
   :show-inheritance:

TransformerBlock
~~~~~~~~~~~~~~~~

.. autoclass:: maldideepkit.transformer.transformer.TransformerBlock
   :members:
   :undoc-members:
   :show-inheritance:

MultiHeadSelfAttention
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: maldideepkit.transformer.transformer.MultiHeadSelfAttention
   :members:
   :undoc-members:
   :show-inheritance:

Examples
--------

Default recipe:

.. code-block:: python

   import numpy as np
   from maldideepkit import MaldiTransformerClassifier

   rng = np.random.default_rng(0)
   X = rng.standard_normal((400, 6000)).astype("float32")
   y = rng.integers(0, 2, size=400)

   clf = MaldiTransformerClassifier(random_state=0).fit(X, y)

Smaller recipe for tiny cohorts:

.. code-block:: python

   clf = MaldiTransformerClassifier(
       embed_dim=32, depth=4, num_heads=2, patch_size=4,
       random_state=0,
   )

CLS-pool variant:

.. code-block:: python

   clf = MaldiTransformerClassifier(pool="cls", random_state=0)

Disable LayerScale (for comparison with a recipe without it):

.. code-block:: python

   clf = MaldiTransformerClassifier(layerscale_init=None, random_state=0)

Auto-scale for a different spectrum layout (patch size is
architecturally scale-agnostic, so ``from_spectrum`` just forwards
``input_dim``):

.. code-block:: python

   clf = MaldiTransformerClassifier.from_spectrum(
       bin_width=6, input_dim=3000, random_state=0,
   )
