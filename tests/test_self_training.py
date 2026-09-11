"""Tests for :class:`maldideepkit.MaldiSelfTrainingClassifier`."""

from __future__ import annotations

import numpy as np
import pytest

from maldideepkit import MaldiMLPClassifier, MaldiSelfTrainingClassifier


@pytest.fixture
def semisup_data(rng):
    n = 80
    X = rng.standard_normal((n, 24)).astype("float32")
    y_full = rng.integers(0, 2, size=n).astype(np.int64)
    y = y_full.copy()
    # Mark the last 60% rows as unlabeled.
    y[int(0.4 * n) :] = -1
    return {"X": X, "y": y, "y_full": y_full, "n_labeled": int(0.4 * n)}


def _tiny_backbone():
    return MaldiMLPClassifier(
        epochs=1, batch_size=8, hidden_dim=16, head_dims=(8,), random_state=0
    )


def test_defaults():
    clf = MaldiSelfTrainingClassifier()
    assert clf.base_estimator is None
    assert clf.n_iter == 3
    assert clf.confidence_threshold == 0.9
    assert clf.unlabeled_marker == -1


def test_get_params_includes_self_training_specific():
    clf = MaldiSelfTrainingClassifier(
        n_iter=5,
        confidence_threshold=0.85,
        unlabeled_marker=-99,
    )
    params = clf.get_params(deep=False)
    assert params["n_iter"] == 5
    assert params["confidence_threshold"] == 0.85
    assert params["unlabeled_marker"] == -99


def test_clone_preserves_hyperparameters():
    from sklearn.base import clone

    clf = MaldiSelfTrainingClassifier(n_iter=7, confidence_threshold=0.95)
    twin = clone(clf)
    assert twin.n_iter == 7
    assert twin.confidence_threshold == 0.95


def test_fit_runs_with_unlabeled_rows(semisup_data):
    clf = MaldiSelfTrainingClassifier(
        base_estimator=_tiny_backbone(),
        n_iter=2,
        confidence_threshold=0.6,
        random_state=0,
    )
    clf.fit(semisup_data["X"], semisup_data["y"])
    assert hasattr(clf, "estimator_")
    assert hasattr(clf, "history_")
    proba = clf.predict_proba(semisup_data["X"])
    assert proba.shape == (semisup_data["X"].shape[0], 2)


def test_all_labels_emits_warning_and_falls_back(semisup_data):
    """When y has no unlabeled rows, fall back to plain supervised + warn."""
    y_fully_labeled = semisup_data["y_full"]  # no -1s
    clf = MaldiSelfTrainingClassifier(
        base_estimator=_tiny_backbone(),
        n_iter=3,
    )
    with pytest.warns(UserWarning, match="no unlabeled rows"):
        clf.fit(semisup_data["X"], y_fully_labeled)
    assert clf.history_["n_pseudolabels"] == []


def test_all_unlabeled_raises(semisup_data):
    """All -1 -> ValueError (we need at least one labeled row)."""
    y_all_unlabeled = np.full_like(semisup_data["y"], -1)
    clf = MaldiSelfTrainingClassifier(base_estimator=_tiny_backbone())
    with pytest.raises(ValueError, match="at least one labeled"):
        clf.fit(semisup_data["X"], y_all_unlabeled)


def test_length_mismatch_raises():
    X = np.zeros((10, 8), dtype=np.float32)
    y = np.zeros(7, dtype=np.int64)
    clf = MaldiSelfTrainingClassifier(base_estimator=_tiny_backbone())
    with pytest.raises(ValueError, match="rows but y"):
        clf.fit(X, y)


def test_threshold_zero_accepts_all_on_first_iter(semisup_data):
    """With threshold=0.0, every unlabeled row is accepted at iter 1."""
    clf = MaldiSelfTrainingClassifier(
        base_estimator=_tiny_backbone(),
        n_iter=3,
        confidence_threshold=0.0,
    )
    clf.fit(semisup_data["X"], semisup_data["y"])
    initial_unlabeled = int((semisup_data["y"] == -1).sum())
    assert clf.history_["n_pseudolabels"][0] == initial_unlabeled


def test_impossible_threshold_early_stops(semisup_data):
    clf = MaldiSelfTrainingClassifier(
        base_estimator=_tiny_backbone(),
        n_iter=5,
        confidence_threshold=1.5,  # impossible
    )
    clf.fit(semisup_data["X"], semisup_data["y"])
    # Should record exactly one iter with zero accepted, then stop.
    assert clf.history_["n_pseudolabels"] == [0]


def test_custom_unlabeled_marker(rng):
    n = 40
    X = rng.standard_normal((n, 8)).astype("float32")
    y = rng.integers(0, 2, size=n).astype(np.int64)
    y[20:] = 99  # mark last 20 with sentinel 99
    clf = MaldiSelfTrainingClassifier(
        base_estimator=_tiny_backbone(),
        unlabeled_marker=99,
        n_iter=1,
        confidence_threshold=0.0,
    )
    clf.fit(X, y)
    assert clf.history_["n_pseudolabels"][0] == 20


def test_default_base_estimator_is_mlp(semisup_data):
    """When base_estimator=None, the default MLP is used."""
    clf = MaldiSelfTrainingClassifier(
        base_estimator=None, n_iter=1, confidence_threshold=0.0
    )
    clf.fit(semisup_data["X"], semisup_data["y"])
    assert isinstance(clf.estimator_, MaldiMLPClassifier)


def test_predict_after_fit(semisup_data):
    clf = MaldiSelfTrainingClassifier(
        base_estimator=_tiny_backbone(), n_iter=1, confidence_threshold=0.0
    )
    clf.fit(semisup_data["X"], semisup_data["y"])
    preds = clf.predict(semisup_data["X"])
    assert preds.shape == (semisup_data["X"].shape[0],)


def test_score_returns_float(semisup_data):
    clf = MaldiSelfTrainingClassifier(
        base_estimator=_tiny_backbone(), n_iter=1, confidence_threshold=0.0
    )
    clf.fit(semisup_data["X"], semisup_data["y"])
    # Score on the labeled subset (so y has no -1)
    labeled = semisup_data["y"] != -1
    s = clf.score(semisup_data["X"][labeled], semisup_data["y"][labeled])
    assert isinstance(s, float)
    assert 0.0 <= s <= 1.0


class TestNonIntegerLabels:
    """Labels keep the caller's dtype instead of being forced to int64."""

    def test_string_labels_in_object_array(self, rng):
        X = rng.standard_normal((60, 24)).astype("float32")
        y = np.array(["res", "sus"] * 30, dtype=object)
        y[30:] = -1  # unlabeled, using the default numeric marker
        clf = MaldiSelfTrainingClassifier(
            base_estimator=_tiny_backbone(), n_iter=1, confidence_threshold=0.0
        ).fit(X, y)
        assert set(clf.classes_) == {"res", "sus"}
        assert set(np.unique(clf.predict(X))) <= {"res", "sus"}

    def test_custom_string_marker(self, rng):
        X = rng.standard_normal((60, 24)).astype("float32")
        y = np.array(["res", "sus"] * 30)
        y[30:] = "?"
        clf = MaldiSelfTrainingClassifier(
            base_estimator=_tiny_backbone(),
            n_iter=1,
            confidence_threshold=0.0,
            unlabeled_marker="?",
        ).fit(X, y)
        assert set(clf.classes_) == {"res", "sus"}
        assert clf.history_["n_pseudolabels"] == [30]

    def test_integer_labels_still_normalised(self, semisup_data):
        clf = MaldiSelfTrainingClassifier(
            base_estimator=_tiny_backbone(), n_iter=1, confidence_threshold=0.0
        ).fit(semisup_data["X"], semisup_data["y"])
        assert clf.classes_.dtype.kind in "iu"


class TestInputTypes:
    """fit() accepts the same input types as the wrapped estimator."""

    def test_maldiset_like_input(self, semisup_data):
        import pandas as pd

        class FakeMaldiSet:
            def __init__(self, X):
                self.X = pd.DataFrame(X)

        clf = MaldiSelfTrainingClassifier(
            base_estimator=_tiny_backbone(), n_iter=1, confidence_threshold=0.0
        ).fit(FakeMaldiSet(semisup_data["X"]), semisup_data["y"])
        assert clf.predict(semisup_data["X"]).shape == (semisup_data["X"].shape[0],)

    def test_dataframe_input(self, semisup_data):
        import pandas as pd

        clf = MaldiSelfTrainingClassifier(
            base_estimator=_tiny_backbone(), n_iter=1, confidence_threshold=0.0
        ).fit(pd.DataFrame(semisup_data["X"]), semisup_data["y"])
        assert clf.predict(semisup_data["X"]).shape == (semisup_data["X"].shape[0],)


class TestPseudoLabelsComeFromPredict:
    """The loop trains on the labels the base estimator would predict,
    not on a local argmax that ignores a tuned decision threshold."""

    class _ThresholdedStub:
        """Binary classifier whose predict() contradicts argmax on
        purpose: it applies a 0.2 threshold on class 1."""

        def __init__(self):
            self.fit_ys: list[np.ndarray] = []

        def get_params(self, deep=False):
            return {}

        def set_params(self, **kw):
            raise TypeError("no params")

        def fit(self, X, y):
            self.fit_ys.append(np.asarray(y))
            self.classes_ = np.array([0, 1])
            return self

        def predict_proba(self, X):
            n = np.asarray(X).shape[0]
            proba = np.tile([0.6, 0.4], (n, 1))
            return proba

        def predict(self, X):
            proba = self.predict_proba(X)
            return (proba[:, 1] >= 0.2).astype(np.int64)

    def test_accepted_labels_match_predict(self, semisup_data):
        stub = self._ThresholdedStub()
        clf = MaldiSelfTrainingClassifier(
            base_estimator=stub, n_iter=1, confidence_threshold=0.0
        )
        clf.fit(semisup_data["X"], semisup_data["y"])
        # sklearn.clone on the stub returns a fresh instance, so grab
        # the labels the *final* estimator was fitted with.
        final = clf.estimator_
        assert final.fit_ys, "stub was never fitted"
        y_final = final.fit_ys[-1]
        n_labeled = semisup_data["n_labeled"]
        pseudo = y_final[n_labeled:]
        # predict() says class 1 everywhere (0.4 >= 0.2); argmax says 0.
        assert pseudo.size > 0
        assert np.all(pseudo == 1)

    def test_confidence_gate_uses_predicted_class_probability(self, semisup_data):
        stub = self._ThresholdedStub()
        # predict() picks class 1 with probability 0.4; a 0.5 gate must
        # therefore reject every pseudo-label even though the *max*
        # probability (class 0 at 0.6) would pass it.
        clf = MaldiSelfTrainingClassifier(
            base_estimator=stub, n_iter=1, confidence_threshold=0.5
        )
        clf.fit(semisup_data["X"], semisup_data["y"])
        assert clf.history_["n_pseudolabels"] == [0]
