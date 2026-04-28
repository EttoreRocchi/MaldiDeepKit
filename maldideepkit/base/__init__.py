"""Shared primitives for MaldiDeepKit classifiers.

Exposes :class:`BaseSpectralClassifier`, the abstract base for all six
model families in the package, together with the :class:`SpectralDataset`
/ :func:`make_loaders` data utilities.
"""

from .classifier import BaseSpectralClassifier
from .data import SpectralDataset, make_loaders

__all__ = [
    "BaseSpectralClassifier",
    "SpectralDataset",
    "make_loaders",
]
