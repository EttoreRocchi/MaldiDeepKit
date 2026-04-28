"""Tests specific to MaldiMLPClassifier."""

from __future__ import annotations

import numpy as np
import pytest

from maldideepkit import MaldiMLPClassifier


class TestMaldiMLPClassifier:
    def test_default_hyperparams(self):
        clf = MaldiMLPClassifier()
        assert clf.hidden_dim == 512
        assert clf.head_dims == (256, 128)
        assert clf.use_attention is True
        assert clf.dropout_high == 0.3
        assert clf.dropout_low == 0.2

    def test_attention_weights_shape(self, synthetic_binary):
        X, y = synthetic_binary
        clf = MaldiMLPClassifier(
            hidden_dim=32, head_dims=(16,), epochs=2, batch_size=8, random_state=0
        ).fit(X, y)
        assert clf.attention_weights_ is not None
        assert clf.attention_weights_.shape[1] == 32

    def test_get_attention_weights(self, synthetic_binary):
        X, y = synthetic_binary
        clf = MaldiMLPClassifier(
            hidden_dim=32, head_dims=(16,), epochs=2, batch_size=8, random_state=0
        ).fit(X, y)
        w = clf.get_attention_weights(X[:5])
        assert w.shape == (5, 32)
        assert np.all((w >= 0) & (w <= 1))  # sigmoid range

    def test_attention_disabled(self, synthetic_binary):
        X, y = synthetic_binary
        clf = MaldiMLPClassifier(
            hidden_dim=32,
            head_dims=(16,),
            epochs=2,
            batch_size=8,
            use_attention=False,
            random_state=0,
        ).fit(X, y)
        assert clf.attention_weights_ is None
        with pytest.raises(RuntimeError, match="use_attention=True"):
            clf.get_attention_weights(X[:5])

    def test_multiclass(self, synthetic_multiclass):
        X, y = synthetic_multiclass
        clf = MaldiMLPClassifier(
            hidden_dim=32, head_dims=(16,), epochs=2, batch_size=8, random_state=0
        ).fit(X, y)
        proba = clf.predict_proba(X)
        assert proba.shape == (len(X), 3)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_forward_dim(self):
        import torch

        from maldideepkit.attention.mlp import SpectralAttentionMLP

        model = SpectralAttentionMLP(input_dim=64, n_classes=2, hidden_dim=16)
        out = model(torch.randn(4, 64))
        assert out.shape == (4, 2)
