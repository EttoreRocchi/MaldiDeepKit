"""Tests for the LR-finder utility."""

from __future__ import annotations

import numpy as np
import pytest

from maldideepkit import MaldiMLPClassifier
from maldideepkit.utils import find_lr
from maldideepkit.utils.loss import FocalLoss


def _tiny_clf(**overrides):
    defaults = dict(hidden_dim=16, head_dims=(8,), batch_size=8, random_state=0)
    defaults.update(overrides)
    return MaldiMLPClassifier(**defaults)


class TestFindLR:
    def test_returns_expected_shape(self, synthetic_binary):
        X, y = synthetic_binary
        out = find_lr(_tiny_clf(), X, y, num_iter=30, start_lr=1e-5, end_lr=1e-1)
        assert set(out) == {"lrs", "losses", "suggested_lr"}
        assert len(out["lrs"]) == len(out["losses"])
        assert len(out["lrs"]) > 0
        assert len(out["lrs"]) <= 30  # may stop early on divergence

    def test_returns_numpy_arrays(self, synthetic_binary):
        X, y = synthetic_binary
        out = find_lr(_tiny_clf(), X, y, num_iter=20, start_lr=1e-5, end_lr=1e-1)
        assert isinstance(out["lrs"], np.ndarray)
        assert isinstance(out["losses"], np.ndarray)
        assert isinstance(out["suggested_lr"], float)

    def test_lrs_geometrically_spaced(self, synthetic_binary):
        X, y = synthetic_binary
        out = find_lr(_tiny_clf(), X, y, num_iter=20, start_lr=1e-5, end_lr=1e-1)
        lrs = out["lrs"]
        if len(lrs) >= 3:
            log_diffs = np.diff(np.log(lrs))
            assert np.allclose(log_diffs, log_diffs[0], atol=1e-6)

    def test_lrs_span_requested_range_when_no_divergence(self, synthetic_binary):
        """A short, narrow sweep should run to completion and cover [start_lr, end_lr]."""
        X, y = synthetic_binary
        out = find_lr(_tiny_clf(), X, y, num_iter=10, start_lr=1e-6, end_lr=1e-5)
        assert out["lrs"][0] == pytest.approx(1e-6, rel=1e-6)
        assert out["lrs"][-1] == pytest.approx(1e-5, rel=1e-6)

    def test_suggested_lr_is_in_range(self, synthetic_binary):
        X, y = synthetic_binary
        out = find_lr(_tiny_clf(), X, y, num_iter=40, start_lr=1e-5, end_lr=1e-1)
        assert out["lrs"][0] <= out["suggested_lr"] <= out["lrs"][-1]

    def test_finite_losses(self, synthetic_binary):
        X, y = synthetic_binary
        out = find_lr(_tiny_clf(), X, y, num_iter=20, start_lr=1e-5, end_lr=1e-2)
        assert np.isfinite(out["losses"]).all()


class TestEarlyStop:
    """Divergence early-stop and the ``< 3`` fallback for ``suggested_lr``."""

    def test_aggressive_end_lr_stops_early(self, synthetic_binary):
        """A wildly large end_lr drives the loss to diverge well before num_iter."""
        X, y = synthetic_binary
        out = find_lr(
            _tiny_clf(),
            X,
            y,
            num_iter=200,
            start_lr=1e-3,
            end_lr=1e6,
            divergence_factor=2.0,
        )
        assert len(out["lrs"]) < 200

    def test_short_run_uses_argmin_for_suggestion(self, synthetic_binary):
        """With < 3 points we can't compute gradients, so we fall back to argmin."""
        X, y = synthetic_binary
        out = find_lr(_tiny_clf(), X, y, num_iter=2, start_lr=1e-6, end_lr=1e-5)
        assert len(out["lrs"]) <= 2
        if len(out["lrs"]) > 0:
            idx = int(np.argmin(out["losses"]))
            assert out["suggested_lr"] == pytest.approx(float(out["lrs"][idx]))


class TestLossWiring:
    """``find_lr`` honours the classifier's configured loss."""

    def test_focal_loss_path(self, synthetic_binary, monkeypatch):
        """Setting ``loss='focal'`` builds a FocalLoss criterion."""
        captured: dict = {}
        real_init = FocalLoss.__init__

        def spy_init(self, *args, **kwargs):
            captured["gamma"] = kwargs.get("gamma", args[1] if len(args) > 1 else None)
            real_init(self, *args, **kwargs)

        monkeypatch.setattr(FocalLoss, "__init__", spy_init)

        X, y = synthetic_binary
        out = find_lr(
            _tiny_clf(loss="focal", focal_gamma=2.0),
            X,
            y,
            num_iter=10,
            start_lr=1e-5,
            end_lr=1e-3,
        )
        assert captured.get("gamma") == 2.0
        assert len(out["lrs"]) > 0

    def test_class_weight_propagates_to_criterion(self, synthetic_binary, monkeypatch):
        """``class_weight='balanced'`` flows into the CE criterion's weight buffer."""
        from torch import nn

        captured: dict = {}
        real_init = nn.CrossEntropyLoss.__init__

        def spy_init(self, *args, **kwargs):
            captured["weight"] = kwargs.get("weight")
            real_init(self, *args, **kwargs)

        monkeypatch.setattr(nn.CrossEntropyLoss, "__init__", spy_init)

        X, y = synthetic_binary
        find_lr(
            _tiny_clf(class_weight="balanced"),
            X,
            y,
            num_iter=5,
            start_lr=1e-5,
            end_lr=1e-3,
        )
        assert captured["weight"] is not None
        assert captured["weight"].numel() == 2  # binary task


class TestSideEffects:
    """``find_lr`` populates the standardization stats on the classifier."""

    def test_writes_feature_stats(self, synthetic_binary):
        X, y = synthetic_binary
        clf = _tiny_clf(standardize=True)
        assert (
            not hasattr(clf, "feature_mean_")
            or clf.__dict__.get("feature_mean_") is None
        )
        find_lr(clf, X, y, num_iter=5, start_lr=1e-5, end_lr=1e-3)
        assert clf.feature_mean_ is not None
        assert clf.feature_std_ is not None
        assert clf.feature_mean_.shape == (X.shape[1],)


class TestReproducibility:
    def test_same_seed_same_curve(self, synthetic_binary):
        X, y = synthetic_binary
        out_a = find_lr(_tiny_clf(), X, y, num_iter=15, start_lr=1e-5, end_lr=1e-2)
        out_b = find_lr(_tiny_clf(), X, y, num_iter=15, start_lr=1e-5, end_lr=1e-2)
        assert np.allclose(out_a["losses"], out_b["losses"])
        assert out_a["suggested_lr"] == pytest.approx(out_b["suggested_lr"])


class TestPlot:
    def test_plot_calls_matplotlib_show(self, synthetic_binary, monkeypatch):
        """``plot=True`` lazily imports matplotlib and shows a figure."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        called = {"show": 0}
        monkeypatch.setattr(
            plt, "show", lambda *a, **kw: called.__setitem__("show", called["show"] + 1)
        )

        X, y = synthetic_binary
        find_lr(
            _tiny_clf(),
            X,
            y,
            num_iter=10,
            start_lr=1e-5,
            end_lr=1e-3,
            plot=True,
        )
        assert called["show"] == 1
        plt.close("all")
