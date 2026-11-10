ResNet module
=============

1-D ResNet-18-style residual stack (He et al., 2016) adapted to
MALDI-TOF spectra: a ``Conv1d`` stem (optionally followed by
``MaxPool1d``), four stages of
:class:`~maldideepkit.resnet.resnet.BasicBlock1D` pairs with strided
downsampling between stages, global average pooling, and a linear
head.

**Peak-friendly defaults.** MaldiDeepKit deliberately deviates from the
literal ResNet-18 stem in three ways that matter on ~6000-bin MALDI-TOF
input: ``stem_stride=1`` (ResNet-18: 2), ``block_kernel_size=7``
(ResNet-18: 3), and ``use_stem_pool=False`` (no stem MaxPool).
Empirically, the literal ResNet-18 configuration (4× initial
downsampling in total) collapses peak-scale features on MALDI spectra,
and the model underfits. The literal backbone is still available with
``stem_stride=2``, ``block_kernel_size=3``, and ``use_stem_pool=True``.

MaldiResNetClassifier
---------------------

.. autoclass:: maldideepkit.MaldiResNetClassifier
   :members:
   :undoc-members:
   :show-inheritance:

SpectralResNet1D
----------------

.. autoclass:: maldideepkit.resnet.resnet.SpectralResNet1D
   :members:
   :undoc-members:
   :show-inheritance:

BasicBlock1D
~~~~~~~~~~~~

.. autoclass:: maldideepkit.resnet.resnet.BasicBlock1D
   :members:
   :undoc-members:
   :show-inheritance:

Examples
--------

Peak-friendly defaults (recommended for MALDI-TOF):

.. code-block:: python

   import numpy as np
   from maldideepkit import MaldiResNetClassifier

   rng = np.random.default_rng(0)
   X = rng.standard_normal((400, 6000)).astype("float32")
   y = rng.integers(0, 2, size=400)

   # Defaults: stem_stride=1, block_kernel_size=7, use_stem_pool=False
   clf = MaldiResNetClassifier(random_state=0).fit(X, y)

Literal ResNet-18 backbone (for literature reproducibility):

.. code-block:: python

   clf = MaldiResNetClassifier(
       stem_kernel_size=7,
       stem_stride=2,
       block_kernel_size=3,
       use_stem_pool=True,
       random_state=0,
   )

Auto-scale the stem for a different spectrum layout:

.. code-block:: python

   clf = MaldiResNetClassifier.from_spectrum(
       bin_width=6, input_dim=3000, random_state=0,
   )
   # stem_kernel_size=5 (scaled from 7 at bin_width=3), stem_stride=1,
   # use_stem_pool=False (peak-friendly stem preserved regardless of
   # bin width; pass use_stem_pool=True as an override to reproduce
   # the literal ResNet-18 stem at any bin width).
