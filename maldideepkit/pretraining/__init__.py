"""Self-supervised pre-training of MaldiDeepKit backbones on binned spectra.

- :class:`MaldiSpectrumEncoder`: wraps a backbone (``"mlp"``,
  ``"cnn"``, ``"resnet"``, ``"transformer"``, or a classifier
  template), pre-trains it with a self-supervised pre-text task on
  unlabeled spectra, and exposes embeddings, the fitted preprocessing
  state, and a checkpoint.
- :class:`Pretext`: base class of the pre-text tasks, with the
  built-in :class:`MAEPretext`, :class:`ShufflePretext`,
  :class:`ShuffleMAEPretext`, and :class:`ContrastivePretext`;
  :func:`register_pretext` / :func:`get_pretext_class` map names to
  classes, as :func:`register_backbone` / :func:`get_backbone_class` do
  for the architectures. :class:`SSLModel` and :class:`SSLBatch` are
  what a pre-text loss receives.
- :func:`make_pretraining_loaders`: lazy train / validation loaders
  over an unlabeled (or partially labeled) spectra pool, held on the
  host or on a GPU, with a ``collate_fn`` seam for custom batching.
"""

from ..base._registry import get_backbone_class, register_backbone
from ._pretext import (
    ContrastivePretext,
    MAEPretext,
    Pretext,
    ShuffleMAEPretext,
    ShufflePretext,
    SSLModel,
    get_pretext_class,
    register_pretext,
)
from ._training import SSLBatch
from .data import make_pretraining_loaders
from .encoder import MaldiSpectrumEncoder

__all__ = [
    "ContrastivePretext",
    "MAEPretext",
    "MaldiSpectrumEncoder",
    "Pretext",
    "SSLBatch",
    "SSLModel",
    "ShuffleMAEPretext",
    "ShufflePretext",
    "get_backbone_class",
    "get_pretext_class",
    "make_pretraining_loaders",
    "register_backbone",
    "register_pretext",
]
