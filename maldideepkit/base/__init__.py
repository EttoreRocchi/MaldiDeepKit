"""Shared primitives for MaldiDeepKit classifiers.

Exposes :class:`BaseSpectralClassifier`, the abstract base for the four
classifier families in the package (MLP, CNN, ResNet, Transformer),
together with the :class:`SpectralDataset` / :func:`make_loaders` data
utilities.
"""

from .classifier import BaseSpectralClassifier
from .data import SpectralDataset, make_loaders

__all__ = [
    "BaseSpectralClassifier",
    "SpectralDataset",
    "make_loaders",
]
