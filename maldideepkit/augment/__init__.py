"""Data-augmentation utilities for binned MALDI-TOF spectra.

All augmentations are callables that transform a training-batch tensor
of shape ``(batch, n_bins)`` and return a tensor of the same shape.
Wire them into :class:`~maldideepkit.BaseSpectralClassifier` via the
``augment=`` kwarg; they apply to training batches only and are
bypassed during validation and inference.
"""

from __future__ import annotations

from .mixing import apply_cutmix, apply_mixup, to_one_hot
from .spectra import SpectrumAugment

__all__ = ["SpectrumAugment", "apply_cutmix", "apply_mixup", "to_one_hot"]
