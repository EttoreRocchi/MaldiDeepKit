API Reference
=============

Complete reference for all public classes and functions in MaldiDeepKit,
organized by module.

.. toctree::
   :maxdepth: 2
   :hidden:

   base
   mlp
   cnn
   resnet
   transformer
   blocks
   augment
   uncertainty
   utils

Base Classes
------------

Every classifier inherits from :class:`~maldideepkit.BaseSpectralClassifier`,
which handles device placement, validation splits, early stopping, and
persistence. Subclasses only need to override ``_build_model()``.

.. autosummary::
   :nosignatures:

   maldideepkit.BaseSpectralClassifier
   maldideepkit.SpectralDataset
   maldideepkit.make_loaders

Classifiers
-----------

.. autosummary::
   :nosignatures:

   maldideepkit.MaldiMLPClassifier
   maldideepkit.MaldiCNNClassifier
   maldideepkit.MaldiResNetClassifier
   maldideepkit.MaldiTransformerClassifier

Building Blocks
---------------

Low-level ``nn.Module`` classes backing each classifier are re-exported
through the :mod:`maldideepkit.blocks` namespace, so a user who wants to
embed a single component in a custom network can import from one place.
See :doc:`blocks` for the full list, grouped into full backbones vs.
composable primitives.

.. code-block:: python

   from maldideepkit.blocks import (
       SpectralTransformer1D,   # full backbones
       TransformerBlock,        # composable primitives
       PatchEmbed1D,
       BasicBlock1D,
       # ...
   )

Training Utilities
------------------

.. autosummary::
   :nosignatures:

   maldideepkit.utils.seed_everything
   maldideepkit.utils.resolve_device
   maldideepkit.utils.EarlyStopping
   maldideepkit.utils.train_loop
   maldideepkit.utils.FocalLoss
   maldideepkit.utils.SAMOptimizer
   maldideepkit.utils.tune_threshold
   maldideepkit.utils.fit_temperature
   maldideepkit.utils.find_lr
   maldideepkit.utils.SpectralEnsemble

Augmentation
------------

Per-batch augmentations applied to training batches only; bypassed
during validation and inference. See :doc:`augment` for the full API.

.. autosummary::
   :nosignatures:

   maldideepkit.augment.SpectrumAugment
   maldideepkit.augment.apply_mixup
   maldideepkit.augment.apply_cutmix

Uncertainty Quantification
--------------------------

Drop-in estimators that wrap a fitted classifier and return calibrated
predictions plus per-sample uncertainty. See :doc:`uncertainty`.

.. autosummary::
   :nosignatures:

   maldideepkit.uncertainty.MCDropoutEstimator
   maldideepkit.uncertainty.LaplaceEstimator
   maldideepkit.uncertainty.ConformalPredictor
   maldideepkit.uncertainty.UncertaintyResult
