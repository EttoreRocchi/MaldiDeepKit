"""MaldiDeepKit - deep learning classifiers for MALDI-TOF binned spectra.

Provides a catalog of PyTorch architectures (MLP, CNN, ResNet,
Transformer) adapted to 1-D binned MALDI-TOF spectra, each wrapped in
a scikit-learn compatible estimator with sensible defaults.

Subpackages
-----------
- ``maldideepkit.base`` - ``BaseSpectralClassifier``, ``SpectralDataset``,
  ``make_loaders``.
- ``maldideepkit.attention`` - ``MaldiMLPClassifier`` (MLP with optional
  sigmoid-gated attention).
- ``maldideepkit.cnn`` - ``MaldiCNNClassifier`` (Conv1D blocks).
- ``maldideepkit.resnet`` - ``MaldiResNetClassifier`` (1-D ResNet-18).
- ``maldideepkit.transformer`` - ``MaldiTransformerClassifier`` (1-D ViT).
- ``maldideepkit.blocks`` - re-exports of every backbone and
  composable primitive for users embedding components into their own
  networks.
- ``maldideepkit.utils`` - reproducibility helpers and shared
  training primitives.
- ``maldideepkit.uncertainty`` - uncertainty-quantification
  estimators (MC Dropout, Laplace approximation, split conformal
  prediction) for fitted classifiers.

Examples
--------
>>> import numpy as np
>>> from maldideepkit import MaldiMLPClassifier
>>> rng = np.random.default_rng(0)
>>> X = rng.standard_normal((64, 256)).astype("float32")
>>> y = rng.integers(0, 2, size=64)
>>> clf = MaldiMLPClassifier(epochs=2, batch_size=16, random_state=0)
>>> _ = clf.fit(X, y)
>>> proba = clf.predict_proba(X)
"""

from . import uncertainty
from .attention.mlp import MaldiMLPClassifier
from .base.classifier import BaseSpectralClassifier
from .base.data import SpectralDataset, make_loaders
from .cnn.cnn import MaldiCNNClassifier
from .resnet.resnet import MaldiResNetClassifier
from .transformer.transformer import MaldiTransformerClassifier

__version__ = "0.2.0"
__author__ = "Ettore Rocchi"

__all__ = [
    "BaseSpectralClassifier",
    "MaldiCNNClassifier",
    "MaldiMLPClassifier",
    "MaldiResNetClassifier",
    "MaldiTransformerClassifier",
    "SpectralDataset",
    "__author__",
    "__version__",
    "make_loaders",
    "uncertainty",
]
