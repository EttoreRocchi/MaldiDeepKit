"""Integration tests for MaldiSet-shaped inputs.

MaldiAMRKit's ``MaldiSet`` exposes a DataFrame-like ``.X``. These tests
use a lightweight fake with the same duck-typed interface so the suite
does not need a real MaldiSet fixture.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from maldideepkit import MaldiMLPClassifier


class FakeMaldiSet:
    """Minimal duck-typed stand-in for ``maldiamrkit.MaldiSet``.

    Exposes ``.X`` (DataFrame of binned spectra), ``.y`` (DataFrame),
    and ``.meta`` - enough for MaldiDeepKit's consumer code.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        n, d = X.shape
        idx = [f"s{i:03d}" for i in range(n)]
        cols = [f"bin_{k}" for k in range(d)]
        self._X = pd.DataFrame(X, index=idx, columns=cols)
        self._y = pd.Series(y, index=idx, name="label").to_frame()
        self._meta = pd.DataFrame({"batch": ["A"] * n}, index=idx)

    @property
    def X(self) -> pd.DataFrame:
        return self._X

    @property
    def y(self) -> pd.DataFrame:
        return self._y

    @property
    def meta(self) -> pd.DataFrame:
        return self._meta


@pytest.fixture
def fake_maldiset(synthetic_binary):
    X, y = synthetic_binary
    return FakeMaldiSet(X, y), y


class TestMaldiSetIntegration:
    def test_fit_on_maldiset(self, fake_maldiset):
        ds, y = fake_maldiset
        clf = MaldiMLPClassifier(
            hidden_dim=16, head_dims=(8,), epochs=2, batch_size=8, random_state=0
        ).fit(ds, y)
        assert clf.input_dim_ == ds.X.shape[1]

    def test_predict_on_maldiset(self, fake_maldiset):
        ds, y = fake_maldiset
        clf = MaldiMLPClassifier(
            hidden_dim=16, head_dims=(8,), epochs=2, batch_size=8, random_state=0
        ).fit(ds, y)
        preds = clf.predict(ds)
        assert preds.shape == (len(ds.X),)

    def test_fit_with_y_from_dataframe(self, fake_maldiset):
        ds, _ = fake_maldiset
        clf = MaldiMLPClassifier(
            hidden_dim=16, head_dims=(8,), epochs=2, batch_size=8, random_state=0
        ).fit(ds, ds.y.squeeze())
        assert clf.predict(ds).shape == (len(ds.X),)

    def test_dataframe_accepted_directly(self, synthetic_binary):
        X, y = synthetic_binary
        df = pd.DataFrame(X)
        clf = MaldiMLPClassifier(
            hidden_dim=16, head_dims=(8,), epochs=2, batch_size=8, random_state=0
        ).fit(df, y)
        assert clf.predict(df).shape == (len(df),)
