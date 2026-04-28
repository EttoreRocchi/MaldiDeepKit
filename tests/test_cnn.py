"""Tests specific to MaldiCNNClassifier."""

from __future__ import annotations

import pytest
import torch

from maldideepkit import MaldiCNNClassifier
from maldideepkit.cnn.cnn import SpectralCNN1D


class TestMaldiCNNClassifier:
    def test_defaults(self):
        clf = MaldiCNNClassifier()
        assert clf.channels == (32, 64, 128, 128)
        assert clf.kernel_size == 7
        assert clf.pool_size == 2
        assert clf.input_transform is None

    def test_fit_predict(self, synthetic_binary):
        X, y = synthetic_binary
        clf = MaldiCNNClassifier(
            channels=(8, 16),
            kernel_size=3,
            head_dim=8,
            epochs=2,
            batch_size=8,
            random_state=0,
        ).fit(X, y)
        assert clf.predict(X).shape == (len(X),)
        assert clf.predict_proba(X).shape == (len(X), 2)

    def test_forward_shape(self):
        model = SpectralCNN1D(input_dim=256, n_classes=3, channels=(8, 16))
        out = model(torch.randn(4, 256))
        assert out.shape == (4, 3)

    def test_too_small_input_raises(self):
        with pytest.raises(ValueError, match="too small"):
            SpectralCNN1D(input_dim=4, channels=(8, 16, 32, 64, 128))

    def test_multiclass(self, synthetic_multiclass):
        X, y = synthetic_multiclass
        clf = MaldiCNNClassifier(
            channels=(8, 16),
            kernel_size=3,
            head_dim=8,
            epochs=2,
            batch_size=8,
            random_state=0,
        ).fit(X, y)
        assert clf.predict_proba(X).shape == (len(X), 3)
