Augment Module
==============

Per-batch data augmentation for binned MALDI-TOF spectra. All
augmentations are callables that take and return a tensor of shape
``(batch, n_bins)``. Wire them into a classifier via the ``augment=``
keyword on :class:`~maldideepkit.BaseSpectralClassifier`; they apply
to training batches only and are bypassed during validation and
inference.

The MixUp / CutMix helpers are exposed for users who roll their own
training loop; the classifier wrappers also accept ``mixup_alpha`` /
``cutmix_alpha`` keywords that engage them automatically.

SpectrumAugment
---------------

.. autoclass:: maldideepkit.augment.SpectrumAugment
   :members:
   :undoc-members:
   :show-inheritance:

MixUp / CutMix helpers
----------------------

.. autofunction:: maldideepkit.augment.apply_mixup

.. autofunction:: maldideepkit.augment.apply_cutmix

.. autofunction:: maldideepkit.augment.to_one_hot

Example
-------

.. code-block:: python

    import numpy as np
    from maldideepkit import MaldiCNNClassifier
    from maldideepkit.augment import SpectrumAugment

    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 6000)).astype("float32")
    y = rng.integers(0, 2, size=200)

    augment = SpectrumAugment(
        noise_std=0.02,
        intensity_jitter=0.05,
        mz_shift_max_bins=2,
        random_state=0,
    )
    clf = MaldiCNNClassifier(augment=augment, random_state=0).fit(X, y)
