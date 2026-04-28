"""Tests for post-hoc calibration utilities and classifier integration."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from maldideepkit import MaldiMLPClassifier
from maldideepkit.utils import fit_temperature, tune_threshold


class TestTuneThreshold:
    def test_returns_float_in_unit_interval(self):
        y = np.array([0, 0, 0, 1, 1])
        p = np.array([0.1, 0.2, 0.6, 0.7, 0.9])
        t = tune_threshold(y, p, metric="balanced_accuracy")
        assert isinstance(t, float)
        assert 0.0 < t < 1.0

    def test_imbalanced_moves_threshold_down(self):
        """On a skewed-positive problem, optimal threshold < 0.5."""
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, size=400)
        proba = np.where(
            y == 1, 0.4 + rng.normal(0, 0.05, 400), rng.normal(0.15, 0.05, 400)
        )
        proba = np.clip(proba, 0.01, 0.99)
        t = tune_threshold(y, proba, metric="balanced_accuracy")
        assert 0.1 < t < 0.5

    def test_accepts_2d_proba(self):
        y = np.array([0, 1, 0, 1])
        p = np.array([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4], [0.2, 0.8]])
        assert 0.0 < tune_threshold(y, p) < 1.0

    def test_three_column_proba_raises(self):
        y = np.array([0, 1])
        p = np.array([[0.3, 0.3, 0.4], [0.1, 0.2, 0.7]])
        with pytest.raises(ValueError, match="binary"):
            tune_threshold(y, p)

    def test_unknown_metric_raises(self):
        y = np.array([0, 1])
        p = np.array([0.2, 0.8])
        with pytest.raises(ValueError, match="Unknown metric"):
            tune_threshold(y, p, metric="accuracy")

    def test_youden_option(self):
        y = np.array([0, 0, 1, 1])
        p = np.array([0.1, 0.4, 0.6, 0.9])
        t = tune_threshold(y, p, metric="youden")
        assert 0.0 < t < 1.0


class TestFitTemperature:
    def test_well_calibrated_logits_give_t_near_one(self):
        """If logits are already calibrated, T should land near 1.0."""
        torch.manual_seed(0)
        n = 500
        y = torch.randint(0, 2, (n,))
        logits = torch.stack(
            [-torch.randn(n) - y.float(), torch.randn(n) + y.float()], dim=1
        )
        t = fit_temperature(logits, y)
        assert 0.5 < t < 2.0

    def test_overconfident_logits_give_t_greater_than_one(self):
        """Over-confident logits (spread-scaled up 10x) should soften via T>1."""
        torch.manual_seed(0)
        n = 500
        y = torch.randint(0, 2, (n,))
        base = torch.stack(
            [-torch.randn(n) - y.float(), torch.randn(n) + y.float()], dim=1
        )
        overconfident = base * 10.0
        t = fit_temperature(overconfident, y)
        assert t > 1.0

    def test_accepts_numpy(self):
        rng = np.random.default_rng(0)
        logits = rng.standard_normal((50, 3)).astype(np.float32)
        y = rng.integers(0, 3, size=50)
        t = fit_temperature(logits, y)
        assert t > 0


class TestClassifierPostHoc:
    def test_defaults_are_off(self):
        clf = MaldiMLPClassifier()
        assert clf.tune_threshold is False
        assert clf.calibrate_temperature is False

    def test_threshold_set_after_fit(self, synthetic_binary):
        X, y = synthetic_binary
        clf = MaldiMLPClassifier(
            hidden_dim=8,
            head_dims=(4,),
            epochs=2,
            batch_size=8,
            tune_threshold=True,
            random_state=0,
        ).fit(X, y)
        assert clf.threshold_ is not None
        assert 0.0 < float(clf.threshold_) < 1.0

    def test_temperature_set_after_fit(self, synthetic_binary):
        X, y = synthetic_binary
        clf = MaldiMLPClassifier(
            hidden_dim=8,
            head_dims=(4,),
            epochs=2,
            batch_size=8,
            calibrate_temperature=True,
            random_state=0,
        ).fit(X, y)
        assert clf.temperature_ is not None
        assert float(clf.temperature_) > 0

    def test_temperature_does_not_change_argmax(self, synthetic_binary):
        """Temperature scaling preserves the argmax of predict_proba."""
        X, y = synthetic_binary
        clf_plain = MaldiMLPClassifier(
            hidden_dim=8,
            head_dims=(4,),
            epochs=2,
            batch_size=8,
            random_state=0,
        ).fit(X, y)
        clf_temp = MaldiMLPClassifier(
            hidden_dim=8,
            head_dims=(4,),
            epochs=2,
            batch_size=8,
            calibrate_temperature=True,
            random_state=0,
        ).fit(X, y)
        # Same seed -> same model weights up to temperature scaling of logits.
        # The argmax of predict_proba should be the same (temperature is
        # positive and monotonic-preserving).
        assert np.array_equal(
            np.argmax(clf_plain.predict_proba(X), axis=1),
            np.argmax(clf_temp.predict_proba(X), axis=1),
        )

    def test_threshold_changes_predict(self, synthetic_binary):
        """With a non-0.5 threshold, predict() differs from argmax."""
        X, y = synthetic_binary
        clf = MaldiMLPClassifier(
            hidden_dim=8,
            head_dims=(4,),
            epochs=2,
            batch_size=8,
            random_state=0,
        ).fit(X, y)
        clf.threshold_ = 0.99
        proba = clf.predict_proba(X)
        expected = (proba[:, 1] >= 0.99).astype(int)
        np.testing.assert_array_equal(clf.predict(X), clf.classes_[expected])

    def test_save_load_roundtrip_preserves_calibration(
        self, tmp_path, synthetic_binary
    ):
        X, y = synthetic_binary
        clf = MaldiMLPClassifier(
            hidden_dim=8,
            head_dims=(4,),
            epochs=2,
            batch_size=8,
            tune_threshold=True,
            calibrate_temperature=True,
            random_state=0,
        ).fit(X, y)
        thr = clf.threshold_
        temp = clf.temperature_
        path = tmp_path / "clf"
        clf.save(path)
        reloaded = MaldiMLPClassifier.load(path)
        assert reloaded.threshold_ == thr
        assert reloaded.temperature_ == temp


class TestThresholdTuningGuardrail:
    """`min_val_auroc_for_threshold_tune` falls back to 0.5 when val AUROC is low."""

    def test_default_gate_value(self):
        clf = MaldiMLPClassifier()
        assert clf.min_val_auroc_for_threshold_tune == 0.6

    def test_low_val_auroc_falls_back_to_half(self, synthetic_binary):
        """With gate=0.99 the tuner should never fire on a short-fit model."""
        X, y = synthetic_binary
        clf = MaldiMLPClassifier(
            hidden_dim=8,
            head_dims=(4,),
            epochs=1,
            batch_size=8,
            tune_threshold=True,
            min_val_auroc_for_threshold_tune=0.99,  # effectively always fall back
            random_state=0,
        ).fit(X, y)
        assert clf.threshold_ == 0.5

    def test_gate_zero_always_tunes(self, synthetic_binary):
        """With gate=0.0 the tuner fires regardless of val AUROC."""
        X, y = synthetic_binary
        clf = MaldiMLPClassifier(
            hidden_dim=8,
            head_dims=(4,),
            epochs=1,
            batch_size=8,
            tune_threshold=True,
            min_val_auroc_for_threshold_tune=0.0,
            random_state=0,
        ).fit(X, y)
        assert clf.threshold_ is not None
        assert 0.0 < float(clf.threshold_) < 1.0
