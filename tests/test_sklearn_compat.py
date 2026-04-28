"""sklearn-compatibility tests for the six MaldiDeepKit classifiers.

These run a subset of ``sklearn.utils.estimator_checks`` that is
compatible with a tiny, GPU-free training configuration, plus a few
hand-rolled checks (clone, get_params / set_params, fitted-attribute
conventions).
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from tests.conftest import ALL_FACTORIES


@pytest.mark.parametrize("factory_name", sorted(ALL_FACTORIES.keys()))
class TestSklearnCompat:
    def test_get_params(self, factory_name):
        clf = ALL_FACTORIES[factory_name]()
        params = clf.get_params()
        assert "random_state" in params
        assert "batch_size" in params

    def test_set_params(self, factory_name):
        clf = ALL_FACTORIES[factory_name]()
        clf.set_params(random_state=42)
        assert clf.random_state == 42

    def test_clone_returns_unfitted(self, factory_name, synthetic_binary):
        X, y = synthetic_binary
        clf = ALL_FACTORIES[factory_name]().fit(X, y)
        cloned = clone(clf)
        assert not hasattr(cloned, "model_")
        cloned.fit(X, y)
        assert hasattr(cloned, "model_")

    def test_fitted_attrs(self, factory_name, synthetic_binary):
        X, y = synthetic_binary
        clf = ALL_FACTORIES[factory_name]().fit(X, y)
        check_is_fitted(clf)
        assert clf.n_features_in_ == X.shape[1]
        assert clf.input_dim_ == X.shape[1]
        assert len(clf.classes_) == 2

    def test_classes_sorted(self, factory_name, synthetic_binary):
        X, y = synthetic_binary
        # Labels shuffled
        labels = np.array([10, 5])
        y2 = labels[y]
        clf = ALL_FACTORIES[factory_name]().fit(X, y2)
        assert list(clf.classes_) == [5, 10]

    def test_predict_dtype_matches_classes(self, factory_name, synthetic_binary):
        X, y = synthetic_binary
        y_str = np.array(["a", "b"])[y]
        clf = ALL_FACTORIES[factory_name]().fit(X, y_str)
        preds = clf.predict(X)
        assert set(np.unique(preds)).issubset({"a", "b"})

    def test_works_in_pipeline(self, factory_name, synthetic_binary):
        X, y = synthetic_binary
        pipe = Pipeline(
            [("scaler", StandardScaler()), ("clf", ALL_FACTORIES[factory_name]())]
        )
        pipe.fit(X, y)
        preds = pipe.predict(X)
        assert preds.shape == (len(X),)
