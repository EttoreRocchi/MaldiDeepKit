"""Tests for the SpectralEnsemble utility."""

from __future__ import annotations

import json
import warnings

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

    def test_dotted_names_stay_distinct(self, tmp_path, synthetic_binary):
        X, y = synthetic_binary
        ens_a = SpectralEnsemble([_small_mlp(), _small_cnn()]).fit(X, y)
        ens_b = SpectralEnsemble([_small_mlp(use_attention=False)]).fit(X, y)
        ens_a.save(tmp_path / "ens0.05")
        ens_b.save(tmp_path / "ens0.10")
        for name in (
            "ens0.05.ensemble.json",
            "ens0.05_0.pt",
            "ens0.05_1.pt",
            "ens0.10.ensemble.json",
            "ens0.10_0.pt",
        ):
            assert (tmp_path / name).exists()
        assert not (tmp_path / "ens0.ensemble.json").exists()
        for ens, name in ((ens_a, "ens0.05"), (ens_b, "ens0.10")):
            reloaded = SpectralEnsemble.load(tmp_path / name)
            assert [type(c) for c in reloaded.classifiers] == [
                type(c) for c in ens.classifiers
            ]
            np.testing.assert_allclose(
                ens.predict_proba(X[:4]), reloaded.predict_proba(X[:4]), atol=1e-5
            )

    def test_load_v030_dotted_name(self, tmp_path, synthetic_binary):
        """v0.3.0 wrote ``save("ens0.05")`` as ``ens0.ensemble.json`` + ``ens0_<i>``."""
        X, y = synthetic_binary
        ens = SpectralEnsemble([_small_mlp(), _small_cnn()]).fit(X, y)
        ens.save(tmp_path / "ens0")  # the v0.3.0 file names of "ens0.05"
        assert not (tmp_path / "ens0.05.ensemble.json").exists()
        index = tmp_path / "ens0.ensemble.json"
        meta = json.loads(index.read_text())
        assert meta["version"] == 2
        index.write_text(json.dumps({**meta, "version": 1}))  # as v0.3.0 wrote it
        with pytest.warns(UserWarning, match=r"ens0\.ensemble\.json instead"):
            reloaded = SpectralEnsemble.load(tmp_path / "ens0.05")
        np.testing.assert_allclose(
            ens.predict_proba(X[:4]), reloaded.predict_proba(X[:4]), atol=1e-5
        )

    def test_legacy_name_of_a_v040_ensemble_is_not_loaded(
        self, tmp_path, synthetic_binary
    ):
        """``load("ens0.99")`` must not return the v0.4.0 ``ens0`` ensemble."""
        X, y = synthetic_binary
        SpectralEnsemble([_small_mlp()]).fit(X, y).save(tmp_path / "ens0")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(FileNotFoundError, match=r"ens0\.99\.ensemble\.json"):
                SpectralEnsemble.load(tmp_path / "ens0.99")

    def test_load_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            SpectralEnsemble.load(tmp_path / "ens0.05")

    def test_fitted_members_with_different_labels_raise(self, synthetic_binary):
        X, y = synthetic_binary
        a = _small_mlp().fit(X, y)
        b = _small_mlp().fit(X, np.array(["R", "S"])[y])
        ensemble = SpectralEnsemble([a, b])
        with pytest.raises(ValueError, match="same labels"):
            ensemble.predict(X[:4])

    def test_failing_save_leaves_nothing_behind(self, tmp_path, synthetic_binary):
        X, y = synthetic_binary
        good = _small_mlp().fit(X, y)
        bad = _small_mlp(loss=lambda logits, target: logits.sum()).fit(X, y)
        with pytest.raises(ValueError):
            SpectralEnsemble([good, bad]).save(tmp_path / "ens")
        assert list(tmp_path.iterdir()) == []

    def test_failing_save_keeps_existing_ensemble(self, tmp_path, synthetic_binary):
        X, y = synthetic_binary
        good = _small_mlp().fit(X, y)
        SpectralEnsemble([good, _small_mlp().fit(X, y)]).save(tmp_path / "ens")
        before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
        changed = _small_mlp(use_attention=False).fit(X, y)
        bad = _small_mlp(loss=lambda logits, target: logits.sum()).fit(X, y)
        with pytest.raises(ValueError):
            SpectralEnsemble([changed, bad]).save(tmp_path / "ens")
        assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before
