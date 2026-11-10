Base module
===========

Abstract base classes and data utilities shared by every MaldiDeepKit
classifier. Users implementing a new single-input architecture only
need to inherit from :class:`~maldideepkit.BaseSpectralClassifier` and
override ``_build_model()``. Dual-input classifiers (spectrum + a
per-row side feature) instead subclass
:class:`~maldideepkit.BaseDualSpectralClassifier` and override
``_build_dual_model()``. The ``optimizer``, ``scheduler``, ``loss``,
and ``callbacks`` parameters choose the training components of the four
architectures and of the dual-branch classifier, custom subclasses
included (with self-training, set them on its base estimator).
:meth:`~maldideepkit.BaseSpectralClassifier.get_embeddings` returns the
embeddings that the final layer of a fitted classifier reads (for a
dual-input classifier, the fused features of the two branches).

BaseSpectralClassifier
----------------------

.. autoclass:: maldideepkit.BaseSpectralClassifier
   :members:
   :inherited-members:
   :undoc-members:
   :show-inheritance:

BaseDualSpectralClassifier
--------------------------

.. autoclass:: maldideepkit.BaseDualSpectralClassifier
   :members:
   :undoc-members:
   :show-inheritance:

SpectralDataset
---------------

.. autoclass:: maldideepkit.SpectralDataset
   :members:
   :undoc-members:
   :show-inheritance:

``SpectralDataset`` accepts NumPy arrays, pandas DataFrames, and any
object with a DataFrame-like ``.X`` attribute (e.g.
:class:`maldiamrkit.MaldiSet`):

.. code-block:: python

   import numpy as np
   import pandas as pd
   from maldideepkit import SpectralDataset

   ds_array = SpectralDataset(np.zeros((10, 6000)))
   ds_frame = SpectralDataset(pd.DataFrame(np.zeros((10, 6000))))

make_loaders
------------

.. autofunction:: maldideepkit.make_loaders

Examples
~~~~~~~~

.. code-block:: python

   import numpy as np
   from maldideepkit import make_loaders

   X = np.random.default_rng(0).standard_normal((200, 6000)).astype("float32")
   y = np.random.default_rng(0).integers(0, 2, size=200)

   train, val, stats = make_loaders(
       X, y, batch_size=32, val_size=0.1, standardize=True, random_state=0,
   )

Input transforms
----------------

The fitted input transform (``standardize`` / ``input_transform``) is a
state dict from :func:`~maldideepkit.base.data.fit_input_transform`,
applied in NumPy by :func:`~maldideepkit.base.data.apply_input_transform`
or on any PyTorch device by its twin
:func:`~maldideepkit.base.data.apply_input_transform_torch` (used by the
GPU-resident pre-training pool of
:class:`~maldideepkit.pretraining.MaldiSpectrumEncoder`).

.. autofunction:: maldideepkit.base.data.fit_input_transform

.. autofunction:: maldideepkit.base.data.apply_input_transform

.. autofunction:: maldideepkit.base.data.apply_input_transform_torch
