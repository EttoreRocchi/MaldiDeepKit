"""1-D convolutional classifier for MALDI-TOF spectra."""

from .cnn import MaldiCNNClassifier, SpectralCNN1D

__all__ = ["MaldiCNNClassifier", "SpectralCNN1D"]
