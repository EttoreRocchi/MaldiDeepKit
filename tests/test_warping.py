"""Tests for the spectral warping hook on `BaseSpectralClassifier`."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, TransformerMixin

from maldideepkit import MaldiMLPClassifier
from maldideepkit.base import data as data_module
from maldideepkit.base.data import _warp_numpy, make_loaders


class ShiftWarper(BaseEstimator, TransformerMixin):
    """Tiny deterministic warper used to test the leakage-safe pipeline.

    `fit()` stores the per-column mean (`ref_`) of the input. `transform()`
    returns ``X - ref_``. That is enough to check that:

    - `fit` only sees the training split in `make_loaders` (the val
      split's values do not enter `ref_`).
    - `transform` is applied to both splits and to new data at predict
      time.
    """

    def fit(self, X, y=None):
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(np.asarray(X))
        self.ref_ = df.to_numpy().mean(axis=0)
        return self

    def transform(self, X):
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(np.asarray(X))
        out = df.to_numpy() - self.ref_
        return pd.DataFrame(out, index=df.index, columns=df.columns)


class TestWarperPipelineOrdering:
    def test_warper_is_fit_on_training_split_only(self):
        """`ref_` should reflect the train split's mean, not the full X."""
        rng = np.random.default_rng(0)
        n = 40
        X = rng.standard_normal((n, 16)).astype(np.float32)
        y = np.repeat([0, 1], n // 2)

        warper = ShiftWarper()
        _, _, stats = make_loaders(
            X,
            y,
            batch_size=8,
            val_size=0.25,
            random_state=0,
            standardize=False,
            warper=warper,
        )
        fitted = stats["warper"]
        assert not np.allclose(fitted.ref_, X.mean(axis=0))

    def test_warper_default_is_none(self):
        clf = MaldiMLPClassifier()
        assert clf.warping is None

    def test_fit_without_warper_unchanged(self, synthetic_binary):
        X, y = synthetic_binary
        clf = MaldiMLPClassifier(
            hidden_dim=8,
            head_dims=(4,),
            epochs=1,
            batch_size=8,
            random_state=0,
        ).fit(X, y)
        assert clf.warper_ is None

    def test_fit_with_warper_stores_fitted_transformer(self, synthetic_binary):
        X, y = synthetic_binary
        clf = MaldiMLPClassifier(
            hidden_dim=8,
            head_dims=(4,),
            epochs=1,
            batch_size=8,
            random_state=0,
            warping=ShiftWarper(),
        ).fit(X, y)
        assert clf.warper_ is not None
        assert hasattr(clf.warper_, "ref_")
        assert not hasattr(clf.warping, "ref_")

    def test_predict_applies_fitted_warper(self, synthetic_binary):
        """`predict_proba` should route through `warper_.transform` at inference."""
        X, y = synthetic_binary
        clf = MaldiMLPClassifier(
            hidden_dim=8,
            head_dims=(4,),
            epochs=1,
            batch_size=8,
            random_state=0,
            warping=ShiftWarper(),
        ).fit(X, y)
        calls = {"n": 0}
        orig = clf.warper_.transform

        def spy(X_):
            calls["n"] += 1
            return orig(X_)

        clf.warper_.transform = spy
        clf.predict_proba(X[:4])
        assert calls["n"] >= 1

    def test_save_load_roundtrip_preserves_warper(self, tmp_path, synthetic_binary):
        X, y = synthetic_binary
        clf = MaldiMLPClassifier(
            hidden_dim=8,
            head_dims=(4,),
            epochs=1,
            batch_size=8,
            random_state=0,
            warping=ShiftWarper(),
        ).fit(X, y)
        ref_before = clf.warper_.ref_.copy()
        path = tmp_path / "clf"
        clf.save(path)
        reloaded = MaldiMLPClassifier.load(path)
        assert reloaded.warper_ is not None
        np.testing.assert_array_equal(reloaded.warper_.ref_, ref_before)
        # Predictions are reproducible after reload.
        np.testing.assert_allclose(
            clf.predict_proba(X[:4]),
            reloaded.predict_proba(X[:4]),
            atol=1e-5,
        )

    def test_no_warper_no_pickle_file(self, tmp_path, synthetic_binary):
        X, y = synthetic_binary
        clf = MaldiMLPClassifier(
            hidden_dim=8,
            head_dims=(4,),
            epochs=1,
            batch_size=8,
            random_state=0,
        ).fit(X, y)
        path = tmp_path / "clf"
        clf.save(path)
        assert not path.with_suffix(".warper.pkl").exists()
        reloaded = MaldiMLPClassifier.load(path)
        assert reloaded.warper_ is None


class TestRealMaldiAMRKitWarping:
    """Smoke test the integration with `maldiamrkit.alignment.Warping`."""

    def test_end_to_end_with_shift_method(self, synthetic_binary):
        pytest.importorskip("maldiamrkit.alignment")
        from maldiamrkit.alignment import Warping

        X, y = synthetic_binary
        clf = MaldiMLPClassifier(
            hidden_dim=8,
            head_dims=(4,),
            epochs=1,
            batch_size=8,
            random_state=0,
            warping=Warping(method="shift"),
        ).fit(X, y)
        assert clf.warper_ is not None
        assert hasattr(clf.warper_, "ref_spec_")
        assert clf.predict(X[:4]).shape == (4,)


def _single_call(warper, X):
    """Reference output: one ``transform`` call on the whole matrix."""
    return np.asarray(warper.transform(pd.DataFrame(X)).to_numpy(), dtype=np.float32)


def _peaked_spectra(n, n_bins=128, seed=0):
    """Non-negative spectra with seven randomly shifted Gaussian peaks."""
    rng = np.random.default_rng(seed)
    grid = np.arange(n_bins)
    X = 0.1 * np.abs(rng.standard_normal((n, n_bins)))
    for i in range(n):
        shift = rng.integers(-4, 5)
        for center in (10, 25, 40, 60, 80, 100, 115):
            X[i] += 10.0 * np.exp(-0.5 * ((grid - center - shift) / 1.5) ** 2)
    return X.astype(np.float32)


class NonMonotonicWarper(ShiftWarper):
    """Warns as ``maldiamrkit.alignment.Warping`` for rows with a positive first bin."""

    def transform(self, X):
        warnings.warn("an unrelated warning", RuntimeWarning, stacklevel=2)
        n_bad = int((X.to_numpy()[:, 0] > 0).sum())
        if n_bad:
            warnings.warn(
                f"Warping produced non-monotonic m/z mappings for {n_bad}/{len(X)} "
                "sample(s). This may indicate poor alignment quality.",
                UserWarning,
                stacklevel=2,
            )
        return super().transform(X)


class TestChunkedWarping:
    """``_warp_numpy`` applies the warper in row chunks with identical output."""

    def test_non_monotonic_warnings_are_merged(self, monkeypatch):
        monkeypatch.setattr(data_module, "_WARP_CHUNK_ROWS", 3)
        X = np.zeros((10, 4), dtype=np.float32)
        X[[0, 4, 5, 9], 0] = 1.0
        warper = NonMonotonicWarper().fit(np.zeros((2, 4)))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = _warp_numpy(warper, X)
        np.testing.assert_array_equal(out, X)
        assert [(w.category, str(w.message)) for w in caught] == [
            (RuntimeWarning, "an unrelated warning")
        ] * 4 + [
            (
                UserWarning,
                "Warping produced non-monotonic m/z mappings for 4/10 sample(s). "
                "This may indicate poor alignment quality.",
            )
        ]

    @pytest.mark.parametrize("n", [1, 3, 4, 9, 10])
    def test_shift_warper_chunks_match_single_call(self, monkeypatch, n):
        monkeypatch.setattr(data_module, "_WARP_CHUNK_ROWS", 3)
        rng = np.random.default_rng(0)
        X = rng.standard_normal((n, 16)).astype(np.float32)
        warper = ShiftWarper().fit(rng.standard_normal((20, 16)))
        expected = _single_call(warper, X)
        calls = []
        orig = warper.transform

        def spy(df):
            calls.append((len(df), int(df.index[0])))
            return orig(df)

        warper.transform = spy
        out = _warp_numpy(warper, X)
        assert calls == [(min(3, n - s), s) for s in range(0, n, 3)]
        assert out.dtype == np.float32
        assert np.array_equal(out, expected)

    @pytest.mark.parametrize("method", ["shift", "linear", "piecewise"])
    def test_maldiamrkit_warping_chunks_match_single_call(self, monkeypatch, method):
        pytest.importorskip("maldiamrkit.alignment")
        from maldiamrkit.alignment import Warping

        warper = Warping(method=method).fit(pd.DataFrame(_peaked_spectra(12)))
        X = _peaked_spectra(11, seed=1)
        expected = _single_call(warper, X)
        monkeypatch.setattr(data_module, "_WARP_CHUNK_ROWS", 4)
        out = _warp_numpy(warper, X)
        assert out.dtype == np.float32
        assert np.array_equal(out, expected)

    def test_empty_input(self, monkeypatch):
        monkeypatch.setattr(data_module, "_WARP_CHUNK_ROWS", 3)
        rng = np.random.default_rng(0)
        warper = ShiftWarper().fit(rng.standard_normal((5, 16)))
        X = np.empty((0, 16), dtype=np.float32)
        out = _warp_numpy(warper, X)
        assert out.shape == (0, 16)
        assert out.dtype == np.float32
        assert np.array_equal(out, _single_call(warper, X))
