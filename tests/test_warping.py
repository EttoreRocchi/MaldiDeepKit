"""Tests for the spectral warping hook on `BaseSpectralClassifier`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, TransformerMixin

from maldideepkit import MaldiMLPClassifier
from maldideepkit.base.data import make_loaders


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
