API reference
=============

Complete reference for all public classes and functions in MaldiDeepKit,
organised by module.

.. toctree::
   :maxdepth: 2
   :hidden:

   base
   mlp
   cnn
   resnet
   transformer
   dual_branch
   semisupervised
   pretraining
   blocks
   augment
   uncertainty
   utils

Base classes
------------

Every single-input classifier inherits from
:class:`~maldideepkit.BaseSpectralClassifier`, which handles device
placement, validation splits, early stopping, and persistence.
Subclasses only need to override ``_build_model()``. Dual-input
classifiers (spectrum + per-row side feature) inherit from
:class:`~maldideepkit.BaseDualSpectralClassifier`, which extends the
single-input base with a concatenation-split layout.

.. autosummary::
   :nosignatures:

   maldideepkit.BaseSpectralClassifier
   maldideepkit.BaseDualSpectralClassifier
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
   maldideepkit.MaldiDualBranchClassifier
   maldideepkit.MaldiSelfTrainingClassifier

Self-supervised pre-training
----------------------------

Pre-train a backbone on unlabelled spectra, then start a classifier's
trunk from it with ``init_from=`` (optionally with linear probing first,
``freeze_trunk_epochs=``). See :doc:`pretraining`.

.. autosummary::
   :nosignatures:

   maldideepkit.pretraining.MaldiSpectrumEncoder
   maldideepkit.pretraining.Pretext
   maldideepkit.pretraining.MAEPretext
   maldideepkit.pretraining.ShufflePretext
   maldideepkit.pretraining.ShuffleMAEPretext
   maldideepkit.pretraining.ContrastivePretext
   maldideepkit.pretraining.register_pretext
   maldideepkit.pretraining.get_pretext_class
   maldideepkit.pretraining.register_backbone
   maldideepkit.pretraining.get_backbone_class
   maldideepkit.pretraining.make_pretraining_loaders

Building blocks
---------------

Low-level ``nn.Module`` classes backing each classifier are re-exported
through the :mod:`maldideepkit.blocks` namespace, so a user who wants to
embed a single component in a custom network can import from one place.
See :doc:`blocks` for the full list, grouped into full backbones vs
composable primitives.

.. code-block:: python

   from maldideepkit.blocks import (
       SpectralTransformer1D,   # full backbones
       TransformerBlock,        # composable primitives
       PatchEmbed1D,
       BasicBlock1D,
       # ...
   )

Training utilities
------------------

.. autosummary::
   :nosignatures:

   maldideepkit.utils.seed_everything
   maldideepkit.utils.resolve_device
   maldideepkit.utils.Callback
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

Uncertainty quantification
--------------------------

Drop-in estimators that wrap a fitted classifier and return calibrated
predictions plus per-sample uncertainty. See :doc:`uncertainty`.

.. autosummary::
   :nosignatures:

   maldideepkit.uncertainty.MCDropoutEstimator
   maldideepkit.uncertainty.LaplaceEstimator
   maldideepkit.uncertainty.ConformalPredictor
   maldideepkit.uncertainty.UncertaintyResult
