"""Shared primitives for MaldiDeepKit classifiers.

Exposes:

- :class:`BaseSpectralClassifier` - abstract base for the four
  single-input classifier families (MLP, CNN, ResNet, Transformer).
- :class:`BaseDualSpectralClassifier` - abstract base for dual-input
  classifiers (e.g. spectrum + antibiotic fingerprint); the side
  input is packed into ``X`` by concatenation to stay
  sklearn-pipeline-friendly.
- :class:`SpectralDataset` / :func:`make_loaders` - data utilities.
"""

from .classifier import BaseSpectralClassifier
from .data import SpectralDataset, make_loaders
from .dual import BaseDualSpectralClassifier

__all__ = [
    "BaseDualSpectralClassifier",
    "BaseSpectralClassifier",
    "SpectralDataset",
    "make_loaders",
]
