"""Tests for the SpectralEnsemble utility."""

from __future__ import annotations

import numpy as np
import pytest

from maldideepkit import MaldiCNNClassifier, MaldiMLPClassifier
from maldideepkit.utils import SpectralEnsemble


def _small_mlp(**kw):
    return MaldiMLPClassifier(
        hidden_dim=8, head_dims=(4,), epochs=1, batch_size=8, random_state=0, **kw
    )


def _small_cnn(**kw):
    return MaldiCNNClassifier(
        channels=(4, 8),
        kernel_size=5,
        pool_size=2,
        head_dim=4,
        epochs=1,
        batch_size=8,
        random_state=0,
        **kw,
    )


class TestSpectralEnsemble:
    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            SpectralEnsemble([])

    def test_fit_fits_every_member(self, synthetic_binary):
        X, y = synthetic_binary
        members = [_small_mlp(), _small_mlp(use_attention=False)]
        SpectralEnsemble(members).fit(X, y)
        assert all(hasattr(m, "model_") for m in members)

    def test_predict_proba_shape_matches_members(self, synthetic_binary):
        X, y = synthetic_binary
        ens = SpectralEnsemble([_small_mlp(), _small_cnn()]).fit(X, y)
        proba = ens.predict_proba(X[:4])
        assert proba.shape == (4, 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_predict_argmax_of_mean(self, synthetic_binary):
        X, y = synthetic_binary
        ens = SpectralEnsemble([_small_mlp(), _small_mlp(use_attention=False)]).fit(
            X, y
        )
        proba = ens.predict_proba(X[:4])
        np.testing.assert_array_equal(
            ens.predict(X[:4]), ens.classes_[np.argmax(proba, axis=1)]
        )

    def test_score_is_float_in_unit_interval(self, synthetic_binary):
        X, y = synthetic_binary
        ens = SpectralEnsemble([_small_mlp(), _small_mlp(use_attention=False)]).fit(
            X, y
        )
        s = ens.score(X, y)
        assert isinstance(s, float)
        assert 0.0 <= s <= 1.0

    def test_mismatched_classes_raise(self):
        """If two members emit different `classes_` after fit, we raise."""

        class _Stub:
            """Minimal fake classifier that captures a preset classes_ on fit."""

            def __init__(self, classes):
                self._classes = np.asarray(classes)

            def fit(self, X, y):
                self.classes_ = self._classes
                return self

        ens = SpectralEnsemble([_Stub([0, 1]), _Stub([0, 1, 2])])
        with pytest.raises(ValueError, match="All members must see the same labels"):
            ens.fit(np.zeros((4, 4)), np.zeros(4))

    def test_save_load_roundtrip(self, tmp_path, synthetic_binary):
        X, y = synthetic_binary
        ens = SpectralEnsemble([_small_mlp(), _small_mlp(use_attention=False)]).fit(
            X, y
        )
        path = tmp_path / "ens"
        ens.save(path)
        assert (tmp_path / "ens.ensemble.json").exists()
        assert (tmp_path / "ens_0.pt").exists()
        assert (tmp_path / "ens_1.pt").exists()
        reloaded = SpectralEnsemble.load(path)
        np.testing.assert_allclose(
            ens.predict_proba(X[:4]),
            reloaded.predict_proba(X[:4]),
            atol=1e-5,
        )
