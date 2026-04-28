"""Tests for BaseSpectralClassifier, SpectralDataset, make_loaders, utils."""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest
import torch

from maldideepkit import (
    BaseSpectralClassifier,
    MaldiCNNClassifier,
    MaldiMLPClassifier,
    SpectralDataset,
    make_loaders,
)
from maldideepkit.utils.reproducibility import resolve_device, seed_everything
from maldideepkit.utils.training import EarlyStopping, train_loop


class TestSpectralDataset:
    def test_from_numpy(self, synthetic_binary):
        X, y = synthetic_binary
        ds = SpectralDataset(X, y)
        assert len(ds) == len(X)
        xb, yb = ds[0]
        assert xb.shape == (X.shape[1],)
        assert xb.dtype == torch.float32
        assert yb.dtype == torch.int64

    def test_no_labels(self, synthetic_binary):
        X, _ = synthetic_binary
        ds = SpectralDataset(X)
        assert ds.y is None
        out = ds[0]
        assert isinstance(out, torch.Tensor)

    def test_standardize(self, synthetic_binary):
        X, y = synthetic_binary
        ds = SpectralDataset(X, y, standardize=True)
        mean = ds.X.mean(dim=0).numpy()
        assert np.allclose(mean, 0.0, atol=1e-5)

    def test_from_dataframe(self, synthetic_binary):
        X, y = synthetic_binary
        df = pd.DataFrame(X)
        ds = SpectralDataset(df, y)
        assert len(ds) == len(X)

    def test_accepts_maldiset_like(self, synthetic_binary):
        class FakeMaldiSet:
            def __init__(self, X):
                self._X = pd.DataFrame(X)

            @property
            def X(self):
                return self._X

        X, y = synthetic_binary
        ds = SpectralDataset(FakeMaldiSet(X), y)
        assert len(ds) == len(X)

    def test_1d_input_reshaped(self):
        ds = SpectralDataset(np.zeros(32))
        assert ds.X.shape == (1, 32)

    def test_bad_shape_raises(self):
        with pytest.raises(ValueError, match="2-D"):
            SpectralDataset(np.zeros((2, 3, 4)))


class TestMakeLoaders:
    def test_shapes(self, synthetic_binary):
        X, y = synthetic_binary
        tr, val, stats = make_loaders(X, y, batch_size=8, val_size=0.2)
        total = sum(xb.shape[0] for xb, _ in tr) + sum(xb.shape[0] for xb, _ in val)
        assert total == len(X)
        assert stats["mean"] is None
        assert stats["std"] is None

    def test_standardize_returns_stats(self, synthetic_binary):
        X, y = synthetic_binary
        tr, val, stats = make_loaders(X, y, batch_size=8, standardize=True)
        assert stats["mean"].shape == (X.shape[1],)
        assert stats["std"].shape == (X.shape[1],)

    def test_stratify_single_class_falls_back(self):
        X = np.random.randn(16, 8).astype(np.float32)
        y = np.ones(16, dtype=int)
        y[0] = 0  # one sample in class 0 -> can't stratify
        tr, val, _ = make_loaders(X, y, batch_size=4, val_size=0.2)
        # Should not raise.
        assert len(list(tr)) >= 1


class TestReproducibility:
    def test_seed_everything(self):
        seed_everything(42)
        a = torch.randn(3)
        seed_everything(42)
        b = torch.randn(3)
        assert torch.allclose(a, b)

    def test_seed_everything_deterministic_flag_sets_env(self):
        """``deterministic=True`` must set the CUBLAS env var + use-det flag."""
        import os

        # Save and restore so we leave the environment untouched.
        prev_env = os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        prev_flag = torch.are_deterministic_algorithms_enabled()
        try:
            seed_everything(0, deterministic=True)
            assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
            assert torch.are_deterministic_algorithms_enabled() is True
        finally:
            if prev_env is not None:
                os.environ["CUBLAS_WORKSPACE_CONFIG"] = prev_env
            else:
                os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
            # Restore the deterministic flag so unrelated tests
            # downstream don't hit the CuBLAS-warning path.
            torch.use_deterministic_algorithms(prev_flag, warn_only=True)

    def test_seed_everything_no_deterministic_by_default(self):
        """Plain ``seed_everything(seed)`` must not set the CUBLAS env var."""
        import os

        # Clear first to isolate the effect of this single call.
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        seed_everything(0)
        assert "CUBLAS_WORKSPACE_CONFIG" not in os.environ


class TestRNGDriftInvariants:
    """Regression tests pinning the invariant that classifier / augment
    object construction and fit path setup don't consume global-RNG
    draws - so that presetting presets, mixing helpers, or extra augment
    objects in the preset table cannot accidentally shift the torch
    global-RNG state mid-sweep.
    """

    def _torch_state(self) -> tuple:
        cpu = torch.get_rng_state().clone()
        if torch.cuda.is_available():
            cuda = [s.clone() for s in torch.cuda.get_rng_state_all()]
        else:
            cuda = []
        return (cpu, cuda)

    def _numpy_state(self) -> tuple:
        return np.random.get_state()

    def _states_equal(self, a, b) -> bool:
        # Compare tuple-of-bytes for torch, state-tuple for numpy.
        return str(a) == str(b)

    def test_spectrum_augment_init_does_not_consume_global_rng(self):
        """Constructing ``SpectrumAugment`` must not draw from the global RNGs."""
        from maldideepkit.augment import SpectrumAugment

        torch.manual_seed(1234)
        np.random.seed(1234)
        t0 = self._torch_state()
        n0 = self._numpy_state()
        # Construct several variants, including ones with domain knobs
        # set, to catch any lazy init that would touch global RNG.
        _ = SpectrumAugment()
        _ = SpectrumAugment(noise_std=0.05, intensity_jitter=0.1, random_state=42)
        _ = SpectrumAugment(
            mz_shift_max_bins=3, mz_warp_max_bins=2, blur_sigma=1.5, random_state=7
        )
        t1 = self._torch_state()
        n1 = self._numpy_state()
        assert self._states_equal(t0, t1), "SpectrumAugment.__init__ shifted torch RNG"
        assert self._states_equal(n0, n1), "SpectrumAugment.__init__ shifted numpy RNG"

    def test_classifier_init_does_not_consume_global_rng(self):
        """Constructing a ``MaldiMLPClassifier`` must not draw from global RNGs."""
        torch.manual_seed(5678)
        np.random.seed(5678)
        t0 = self._torch_state()
        n0 = self._numpy_state()
        _ = MaldiMLPClassifier(
            hidden_dim=8,
            head_dims=(4,),
            mixup_alpha=0.2,
            cutmix_alpha=1.0,
            ema_decay=0.99,
        )
        t1 = self._torch_state()
        n1 = self._numpy_state()
        assert self._states_equal(t0, t1)
        assert self._states_equal(n0, n1)

    def test_fit_is_robust_to_prior_global_rng_state(self):
        """Two fits with the same ``random_state`` produce identical outputs
        regardless of what consumed the global RNG before them - because
        ``fit()`` re-seeds at entry. This pins the invariant that makes
        candidate-2 RNG drift (e.g. from preset-table construction)
        irrelevant."""
        X = np.random.RandomState(0).randn(48, 32).astype("float32")
        y = np.array([0, 1] * 24)

        def _fit_once(prior_draws: int) -> np.ndarray:
            torch.manual_seed(999)
            np.random.seed(999)
            for _ in range(prior_draws):
                _ = torch.randn(1).item()
                _ = np.random.rand()
            clf = MaldiMLPClassifier(
                hidden_dim=8,
                head_dims=(4,),
                epochs=2,
                batch_size=8,
                random_state=0,
            ).fit(X, y)
            return clf.predict_proba(X[:4])

        p_a = _fit_once(prior_draws=0)
        p_b = _fit_once(prior_draws=37)
        np.testing.assert_allclose(p_a, p_b, atol=1e-5)

    def test_resolve_device_auto(self):
        dev = resolve_device("auto")
        assert isinstance(dev, torch.device)

    def test_resolve_device_cpu(self):
        assert resolve_device("cpu").type == "cpu"

    def test_resolve_device_bad(self):
        with pytest.raises(ValueError):
            resolve_device(42)  # type: ignore[arg-type]


class TestEarlyStopping:
    def test_tracks_best(self):
        es = EarlyStopping(patience=2)
        model = torch.nn.Linear(4, 2)
        assert es.step(1.0, model) is True
        assert es.step(0.5, model) is True
        # No improvement
        assert es.step(0.6, model) is False
        assert not es.should_stop
        assert es.step(0.7, model) is False
        assert es.should_stop


class TestTrainingHygiene:
    """`weight_decay`, `grad_clip_norm`, `label_smoothing` knobs."""

    def _capture_optimizer(self, monkeypatch):
        captured: dict = {}
        import maldideepkit.base.classifier as _mod
        from maldideepkit.utils.training import train_loop as _tl

        def spy(*args, **kwargs):
            captured["optimizer"] = args[4]
            return _tl(*args, **kwargs)

        monkeypatch.setattr(_mod, "train_loop", spy)
        return captured

    def test_adam_when_weight_decay_zero(self, synthetic_binary, monkeypatch):
        captured = self._capture_optimizer(monkeypatch)
        X, y = synthetic_binary
        MaldiMLPClassifier(
            hidden_dim=8,
            head_dims=(4,),
            epochs=1,
            batch_size=8,
            weight_decay=0.0,
            random_state=0,
        ).fit(X, y)
        assert isinstance(captured["optimizer"], torch.optim.Adam)
        assert not isinstance(captured["optimizer"], torch.optim.AdamW)

    def test_adamw_when_weight_decay_positive(self, synthetic_binary, monkeypatch):
        captured = self._capture_optimizer(monkeypatch)
        X, y = synthetic_binary
        MaldiMLPClassifier(
            hidden_dim=8,
            head_dims=(4,),
            epochs=1,
            batch_size=8,
            weight_decay=0.05,
            random_state=0,
        ).fit(X, y)
        assert isinstance(captured["optimizer"], torch.optim.AdamW)
        assert captured["optimizer"].param_groups[0]["weight_decay"] == 0.05

    def test_grad_clip_norm_bounds_grad(self):
        """After clip, each param's grad norm should be ≤ max_norm."""
        torch.manual_seed(0)
        model = torch.nn.Linear(4, 2)
        # Force a large loss so gradients would explode without clipping.
        X = torch.randn(16, 4) * 1e3
        y = torch.randint(0, 2, (16,))
        ds = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        recorded: list[float] = []

        def record(_epoch, _val_loss):
            params = [p for g in opt.param_groups for p in g["params"]]
            total = torch.norm(
                torch.stack(
                    [p.grad.detach().norm() for p in params if p.grad is not None]
                )
            ).item()
            recorded.append(total)

        train_loop(
            model,
            loader,
            (X, y),
            torch.nn.CrossEntropyLoss(),
            opt,
            scheduler=None,
            device=torch.device("cpu"),
            epochs=2,
            early_stopping=EarlyStopping(patience=100),
            on_epoch_end=record,
            grad_clip_norm=0.5,
        )
        for g in recorded:
            assert g <= 0.5 + 1e-5, f"grad norm {g} exceeded clip 0.5"

    def test_label_smoothing_changes_loss(self, synthetic_binary):
        """`label_smoothing > 0` produces a different loss than 0 for confident predictions."""
        import torch.nn.functional as F

        logits = torch.tensor([[10.0, -10.0], [-10.0, 10.0]])
        target = torch.tensor([0, 1])
        ce_hard = F.cross_entropy(logits, target, label_smoothing=0.0)
        ce_smooth = F.cross_entropy(logits, target, label_smoothing=0.1)
        assert ce_smooth > ce_hard  # smoothing pulls target away from one-hot

    def test_label_smoothing_default_is_zero(self):
        clf = MaldiMLPClassifier()
        assert clf.label_smoothing == 0.0


class TestWarmup:
    """Linear LR warmup in ``train_loop`` and ``BaseSpectralClassifier``."""

    def _build_loop_args(self, base_lr: float):
        torch.manual_seed(0)
        model = torch.nn.Linear(4, 2)
        X = torch.randn(16, 4)
        y = torch.randint(0, 2, (16,))
        ds = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)
        return model, loader, X, y, optimizer

    def test_train_loop_warmup_scales_lr_per_epoch(self):
        """lr at the end of each warmup epoch equals (epoch+1)/warmup * base."""
        base_lr = 1e-2
        warmup = 4
        observed: list[float] = []
        model, loader, X, y, opt = self._build_loop_args(base_lr)

        def record(epoch: int, _val_loss: float) -> None:
            observed.append(opt.param_groups[0]["lr"])

        train_loop(
            model,
            loader,
            (X, y),
            torch.nn.CrossEntropyLoss(),
            opt,
            scheduler=None,
            device=torch.device("cpu"),
            epochs=6,
            early_stopping=EarlyStopping(patience=100),
            on_epoch_end=record,
            warmup_epochs=warmup,
        )
        assert observed[0] == pytest.approx(base_lr * 1 / warmup)
        assert observed[1] == pytest.approx(base_lr * 2 / warmup)
        assert observed[2] == pytest.approx(base_lr * 3 / warmup)
        assert observed[3] == pytest.approx(base_lr)
        assert observed[4] == pytest.approx(base_lr)
        assert observed[5] == pytest.approx(base_lr)

    def test_train_loop_warmup_preserves_param_group_ratios(self):
        """Each param group's lr is scaled relative to its own target lr."""
        torch.manual_seed(0)
        model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 2))
        X = torch.randn(16, 4)
        y = torch.randint(0, 2, (16,))
        ds = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False)
        opt = torch.optim.Adam(
            [
                {"params": model[0].parameters(), "lr": 1e-2},
                {"params": model[1].parameters(), "lr": 1e-3},
            ]
        )
        observed: list[tuple[float, float]] = []

        def record(_epoch: int, _val_loss: float) -> None:
            observed.append((opt.param_groups[0]["lr"], opt.param_groups[1]["lr"]))

        train_loop(
            model,
            loader,
            (X, y),
            torch.nn.CrossEntropyLoss(),
            opt,
            scheduler=None,
            device=torch.device("cpu"),
            epochs=3,
            early_stopping=EarlyStopping(patience=100),
            on_epoch_end=record,
            warmup_epochs=2,
        )
        assert observed[0] == (pytest.approx(5e-3), pytest.approx(5e-4))
        assert observed[1] == (pytest.approx(1e-2), pytest.approx(1e-3))

    def test_classifier_warmup_default_is_zero(self):
        """Default behaviour is unchanged when ``warmup_epochs`` is not set."""
        clf = MaldiMLPClassifier()
        assert clf.warmup_epochs == 0

    def test_classifier_warmup_fit_runs(self, synthetic_binary):
        """Fit with warmup completes and produces a usable classifier."""
        X, y = synthetic_binary
        clf = MaldiMLPClassifier(
            hidden_dim=16,
            head_dims=(8,),
            epochs=4,
            batch_size=8,
            warmup_epochs=2,
            random_state=0,
        )
        clf.fit(X, y)
        proba = clf.predict_proba(X[:4])
        assert proba.shape == (4, 2)


class TestCosineSchedule:
    """Cosine annealing (post-warmup) is the default LR schedule."""

    def _build(self, base_lr: float, warmup: int):
        torch.manual_seed(0)
        model = torch.nn.Linear(4, 2)
        X = torch.randn(16, 4)
        y = torch.randint(0, 2, (16,))
        ds = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False)
        opt = torch.optim.Adam(model.parameters(), lr=base_lr)
        return model, loader, X, y, opt

    def test_cosine_steps_only_after_warmup(self):
        """During warmup the manual ramp owns lr; cosine engages afterwards."""
        base_lr = 1e-2
        warmup = 3
        epochs = 6
        model, loader, X, y, opt = self._build(base_lr, warmup)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=epochs - warmup, eta_min=0.0
        )
        observed: list[float] = []

        def record(_epoch: int, _val_loss: float) -> None:
            observed.append(opt.param_groups[0]["lr"])

        train_loop(
            model,
            loader,
            (X, y),
            torch.nn.CrossEntropyLoss(),
            opt,
            scheduler,
            torch.device("cpu"),
            epochs,
            EarlyStopping(patience=100),
            on_epoch_end=record,
            warmup_epochs=warmup,
        )
        # Warmup epochs: linear ramp 1/3, 2/3, 3/3.
        assert observed[0] == pytest.approx(base_lr / 3)
        assert observed[1] == pytest.approx(base_lr * 2 / 3)
        assert observed[2] == pytest.approx(base_lr)
        # After warmup cosine decays monotonically each step.
        assert observed[3] > observed[4] > observed[5]
        assert observed[5] < base_lr

    def test_base_classifier_uses_cosine_scheduler(self, synthetic_binary, monkeypatch):
        """``BaseSpectralClassifier.fit`` ends with lr < base_lr (cosine decay)."""
        import maldideepkit.base.classifier as _mod
        from maldideepkit.utils.training import train_loop as _tl

        final_lrs: list[float] = []

        def patched_tl(*args, **kwargs):
            optimizer = args[4]
            result = _tl(*args, **kwargs)
            final_lrs.append(optimizer.param_groups[0]["lr"])
            return result

        monkeypatch.setattr(_mod, "train_loop", patched_tl)

        X, y = synthetic_binary
        clf = MaldiMLPClassifier(
            hidden_dim=8,
            head_dims=(4,),
            epochs=6,
            early_stopping_patience=100,  # disable early stopping
            batch_size=8,
            learning_rate=1e-2,
            warmup_epochs=0,
            random_state=0,
        )
        clf.fit(X, y)
        assert final_lrs, "train_loop patch did not fire"
        assert final_lrs[0] < clf.learning_rate


class TestFitPredict:
    def test_fit_returns_self(self, synthetic_binary):
        X, y = synthetic_binary
        clf = MaldiMLPClassifier(
            hidden_dim=16, head_dims=(8,), epochs=2, batch_size=8, random_state=0
        )
        assert clf.fit(X, y) is clf

    def test_classes_preserved(self, synthetic_binary):
        X, y = synthetic_binary
        y_str = np.array(["neg", "pos"])[y]
        clf = MaldiMLPClassifier(
            hidden_dim=16, head_dims=(8,), epochs=2, batch_size=8, random_state=0
        ).fit(X, y_str)
        assert set(clf.classes_.tolist()) == {"neg", "pos"}
        preds = clf.predict(X)
        assert set(np.unique(preds)).issubset({"neg", "pos"})

    def test_class_weight_balanced(self, synthetic_binary):
        X, y = synthetic_binary
        clf = MaldiMLPClassifier(
            hidden_dim=16,
            head_dims=(8,),
            epochs=2,
            batch_size=8,
            class_weight="balanced",
            random_state=0,
        ).fit(X, y)
        assert clf.predict(X).shape == (len(X),)

    def test_class_weight_bad_string(self, synthetic_binary):
        X, y = synthetic_binary
        with pytest.raises(ValueError, match="class_weight"):
            MaldiMLPClassifier(
                class_weight="not-a-mode",
                hidden_dim=16,
                head_dims=(8,),
                epochs=1,
                batch_size=8,
                random_state=0,
            ).fit(X, y)

    def test_class_weight_array(self, synthetic_binary):
        X, y = synthetic_binary
        clf = MaldiMLPClassifier(
            class_weight=[1.0, 2.0],
            hidden_dim=16,
            head_dims=(8,),
            epochs=1,
            batch_size=8,
            random_state=0,
        ).fit(X, y)
        assert clf.predict(X).shape == (len(X),)

    def test_class_weight_wrong_shape(self, synthetic_binary):
        X, y = synthetic_binary
        with pytest.raises(ValueError, match="class_weight has shape"):
            MaldiMLPClassifier(
                class_weight=[1.0, 2.0, 3.0],
                hidden_dim=16,
                head_dims=(8,),
                epochs=1,
                batch_size=8,
                random_state=0,
            ).fit(X, y)

    def test_verbose_fit(self, synthetic_binary, capsys):
        X, y = synthetic_binary
        MaldiMLPClassifier(
            hidden_dim=16,
            head_dims=(8,),
            epochs=2,
            batch_size=8,
            verbose=True,
            random_state=0,
        ).fit(X, y)
        out = capsys.readouterr().out
        assert "val_loss" in out

    def test_single_class_y_raises(self, synthetic_binary):
        X, _ = synthetic_binary
        y = np.zeros(len(X), dtype=int)
        with pytest.raises(ValueError, match="at least 2 classes"):
            MaldiMLPClassifier(epochs=1, batch_size=8, random_state=0).fit(X, y)

    def test_y_shape_mismatch_raises(self, synthetic_binary):
        X, y = synthetic_binary
        with pytest.raises(ValueError, match="X has"):
            MaldiMLPClassifier(epochs=1, batch_size=8, random_state=0).fit(X, y[:10])

    def test_input_dim_mismatch_at_predict(self, synthetic_binary):
        X, y = synthetic_binary
        clf = MaldiMLPClassifier(
            hidden_dim=16, head_dims=(8,), epochs=1, batch_size=8, random_state=0
        ).fit(X, y)
        with pytest.raises(ValueError, match="features"):
            clf.predict(X[:, :32])

    def test_input_dim_explicit_matches(self, synthetic_binary):
        X, y = synthetic_binary
        clf = MaldiMLPClassifier(
            input_dim=X.shape[1],
            hidden_dim=16,
            head_dims=(8,),
            epochs=1,
            batch_size=8,
            random_state=0,
        ).fit(X, y)
        assert clf.input_dim_ == X.shape[1]

    def test_input_dim_explicit_wrong(self, synthetic_binary):
        X, y = synthetic_binary
        clf = MaldiMLPClassifier(
            input_dim=X.shape[1] + 1,
            hidden_dim=16,
            head_dims=(8,),
            epochs=1,
            batch_size=8,
            random_state=0,
        )
        with pytest.raises(ValueError, match="input_dim"):
            clf.fit(X, y)

    def test_predict_proba_sums_to_one(self, synthetic_binary):
        X, y = synthetic_binary
        clf = MaldiMLPClassifier(
            hidden_dim=16, head_dims=(8,), epochs=1, batch_size=8, random_state=0
        ).fit(X, y)
        proba = clf.predict_proba(X)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_score_is_accuracy(self, synthetic_binary):
        X, y = synthetic_binary
        clf = MaldiMLPClassifier(
            hidden_dim=16, head_dims=(8,), epochs=1, batch_size=8, random_state=0
        ).fit(X, y)
        preds = clf.predict(X)
        assert clf.score(X, y) == pytest.approx(np.mean(preds == y))

    def test_reproducibility(self, synthetic_binary):
        X, y = synthetic_binary
        clf1 = MaldiMLPClassifier(
            hidden_dim=16, head_dims=(8,), epochs=2, batch_size=8, random_state=0
        ).fit(X, y)
        clf2 = MaldiMLPClassifier(
            hidden_dim=16, head_dims=(8,), epochs=2, batch_size=8, random_state=0
        ).fit(X, y)
        assert np.allclose(clf1.predict_proba(X), clf2.predict_proba(X), atol=1e-6)

    def test_standardize_stats_stored(self, synthetic_binary):
        X, y = synthetic_binary
        clf = MaldiCNNClassifier(
            channels=(4, 8),
            kernel_size=3,
            head_dim=8,
            epochs=1,
            batch_size=8,
            input_transform="standardize",
            random_state=0,
        ).fit(X, y)
        assert clf.feature_mean_ is not None
        assert clf.feature_std_ is not None


class TestSaveLoad:
    def test_roundtrip(self, tmp_path, synthetic_binary):
        X, y = synthetic_binary
        clf = MaldiMLPClassifier(
            hidden_dim=16, head_dims=(8,), epochs=2, batch_size=8, random_state=0
        ).fit(X, y)
        path = tmp_path / "model"
        clf.save(path)
        loaded = MaldiMLPClassifier.load(path)
        assert np.allclose(clf.predict_proba(X), loaded.predict_proba(X), atol=1e-6)

    def test_load_via_base_class(self, tmp_path, synthetic_binary):
        X, y = synthetic_binary
        clf = MaldiMLPClassifier(
            hidden_dim=16, head_dims=(8,), epochs=2, batch_size=8, random_state=0
        ).fit(X, y)
        clf.save(tmp_path / "model.pt")
        loaded = BaseSpectralClassifier.load(tmp_path / "model.json")
        assert type(loaded).__name__ == "MaldiMLPClassifier"
        assert np.allclose(clf.predict_proba(X), loaded.predict_proba(X), atol=1e-6)

    def test_load_wrong_class_raises(self, tmp_path, synthetic_binary):
        X, y = synthetic_binary
        clf = MaldiMLPClassifier(
            hidden_dim=16, head_dims=(8,), epochs=1, batch_size=8, random_state=0
        ).fit(X, y)
        clf.save(tmp_path / "model")
        with pytest.raises(ValueError, match="MaldiMLPClassifier"):
            MaldiCNNClassifier.load(tmp_path / "model")

    def test_load_missing_pt_raises(self, tmp_path, synthetic_binary):
        X, y = synthetic_binary
        clf = MaldiMLPClassifier(
            hidden_dim=16, head_dims=(8,), epochs=1, batch_size=8, random_state=0
        ).fit(X, y)
        clf.save(tmp_path / "model")
        os.remove(tmp_path / "model.pt")
        with pytest.raises(FileNotFoundError):
            MaldiMLPClassifier.load(tmp_path / "model")

    def test_load_missing_json_raises(self, tmp_path, synthetic_binary):
        X, y = synthetic_binary
        clf = MaldiMLPClassifier(
            hidden_dim=16, head_dims=(8,), epochs=1, batch_size=8, random_state=0
        ).fit(X, y)
        clf.save(tmp_path / "model")
        os.remove(tmp_path / "model.json")
        with pytest.raises(FileNotFoundError):
            MaldiMLPClassifier.load(tmp_path / "model")

    def test_json_contains_metadata(self, tmp_path, synthetic_binary):
        X, y = synthetic_binary
        clf = MaldiMLPClassifier(
            hidden_dim=16, head_dims=(8,), epochs=1, batch_size=8, random_state=0
        ).fit(X, y)
        clf.save(tmp_path / "model")
        with open(tmp_path / "model.json") as fh:
            meta = json.load(fh)
        assert meta["class_name"] == "MaldiMLPClassifier"
        assert meta["fitted"]["input_dim_"] == X.shape[1]
        assert meta["fitted"]["n_classes_"] == 2

    def test_save_unfitted_raises(self, tmp_path):
        clf = MaldiMLPClassifier(
            hidden_dim=16, head_dims=(8,), epochs=1, random_state=0
        )
        from sklearn.exceptions import NotFittedError

        with pytest.raises(NotFittedError):
            clf.save(tmp_path / "model")


class TestFocalLoss:
    """``loss="focal"`` wiring on BaseSpectralClassifier.

    Unit tests for the ``FocalLoss`` module itself live in
    ``tests/test_loss.py``.
    """

    def test_base_classifier_uses_focal_when_loss_focal(
        self, synthetic_binary, monkeypatch
    ):
        """`loss='focal'` path builds the FocalLoss criterion."""
        from maldideepkit.utils import FocalLoss

        captured = {}
        import maldideepkit.base.classifier as _mod
        from maldideepkit.utils.training import train_loop as _tl

        def spy(*args, **kwargs):
            captured["criterion"] = args[3]
            return _tl(*args, **kwargs)

        monkeypatch.setattr(_mod, "train_loop", spy)

        X, y = synthetic_binary
        MaldiMLPClassifier(
            hidden_dim=8,
            head_dims=(4,),
            epochs=1,
            batch_size=8,
            loss="focal",
            focal_gamma=2.0,
            random_state=0,
        ).fit(X, y)
        assert isinstance(captured["criterion"], FocalLoss)
        assert captured["criterion"].gamma == 2.0

    def test_unknown_loss_raises(self, synthetic_binary):
        X, y = synthetic_binary
        with pytest.raises(ValueError, match="Unknown loss"):
            MaldiMLPClassifier(
                hidden_dim=8,
                head_dims=(4,),
                epochs=1,
                loss="nonsense",
                random_state=0,
            ).fit(X, y)


class TestAMP:
    """Mixed precision training (`use_amp=True`)."""

    def test_cpu_use_amp_is_noop(self, synthetic_binary):
        """On CPU, use_amp=True runs to completion without autocast."""
        X, y = synthetic_binary
        clf = MaldiMLPClassifier(
            hidden_dim=8,
            head_dims=(4,),
            epochs=2,
            batch_size=8,
            use_amp=True,
            device="cpu",
            random_state=0,
        ).fit(X, y)
        assert clf.predict(X).shape == (len(X),)

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="AMP fast path requires CUDA"
    )
    def test_cuda_use_amp_runs(self, synthetic_binary):
        X, y = synthetic_binary
        clf = MaldiMLPClassifier(
            hidden_dim=8,
            head_dims=(4,),
            epochs=2,
            batch_size=8,
            use_amp=True,
            device="cuda",
            random_state=0,
        ).fit(X, y)
        assert clf.predict(X).shape == (len(X),)


class TestSWA:
    """`swa_start_epoch` wiring via `BaseSpectralClassifier`."""

    def test_default_is_none(self):
        assert MaldiMLPClassifier().swa_start_epoch is None

    def test_swa_replaces_final_weights(self, synthetic_binary):
        """SWA-averaged weights differ from the plain best-val checkpoint."""
        X, y = synthetic_binary
        common = dict(
            hidden_dim=8,
            head_dims=(4,),
            epochs=6,
            early_stopping_patience=100,  # disable ES so SWA has room
            batch_size=8,
            learning_rate=1e-2,
            random_state=0,
        )
        clf_plain = MaldiMLPClassifier(**common).fit(X, y)
        clf_swa = MaldiMLPClassifier(swa_start_epoch=2, **common).fit(X, y)
        # Compare first-layer weights; SWA average should differ bit-wise.
        p_plain = next(clf_plain.model_.parameters()).detach().cpu()
        p_swa = next(clf_swa.model_.parameters()).detach().cpu()
        assert not torch.allclose(p_plain, p_swa)


class TestMetricsLogging:
    """Per-epoch metrics CSV written when ``metrics_log_path`` is set."""

    def _fit_mlp(self, X, y, **extra) -> MaldiMLPClassifier:
        return MaldiMLPClassifier(
            hidden_dim=8,
            head_dims=(4,),
            epochs=3,
            batch_size=8,
            early_stopping_patience=100,  # disable ES so the CSV has all rows
            random_state=0,
            **extra,
        ).fit(X, y)

    def test_csv_written_when_path_set(self, synthetic_binary, tmp_path):
        X, y = synthetic_binary
        path = tmp_path / "m.csv"
        self._fit_mlp(X, y, metrics_log_path=path)
        df = pd.read_csv(path)
        assert len(df) == 3
        assert list(df.columns) == [
            "epoch",
            "train_loss",
            "val_loss",
            "lr",
            "mean_grad_norm",
            "n_grad_updates",
        ]
        assert (df["n_grad_updates"] >= 1).all()

    def test_grad_norm_recorded_finite(self, synthetic_binary, tmp_path):
        """`mean_grad_norm` is finite + positive even when grad_clip_norm is None."""
        X, y = synthetic_binary
        path = tmp_path / "m.csv"
        self._fit_mlp(X, y, metrics_log_path=path)
        df = pd.read_csv(path)
        assert np.isfinite(df["mean_grad_norm"]).all()
        assert (df["mean_grad_norm"] > 0).all()

    def test_track_train_metrics_adds_auroc_columns(self, synthetic_binary, tmp_path):
        X, y = synthetic_binary
        path = tmp_path / "m.csv"
        self._fit_mlp(X, y, metrics_log_path=path, track_train_metrics=True)
        df = pd.read_csv(path)
        assert "train_auroc" in df.columns
        assert "val_auroc" in df.columns
        assert np.isfinite(df["train_auroc"]).all()
        assert np.isfinite(df["val_auroc"]).all()

    def test_no_csv_when_path_unset(self, synthetic_binary, tmp_path):
        """Default path (metrics_log_path=None) writes nothing."""
        X, y = synthetic_binary
        before = set(tmp_path.iterdir())
        self._fit_mlp(X, y)
        after = set(tmp_path.iterdir())
        assert before == after

    def test_stale_csv_overwritten(self, synthetic_binary, tmp_path):
        """Re-fitting with the same `metrics_log_path` truncates the old file."""
        X, y = synthetic_binary
        path = tmp_path / "m.csv"
        path.write_text("garbage\n1,2,3\n")
        self._fit_mlp(X, y, metrics_log_path=path)
        df = pd.read_csv(path)
        assert "epoch" in df.columns
        assert len(df) == 3

    def test_default_metrics_log_path_is_none(self):
        clf = MaldiMLPClassifier()
        assert clf.metrics_log_path is None
        assert clf.track_train_metrics is False

    def test_post_fit_sidecar_written(self, synthetic_binary, tmp_path):
        """When ``metrics_log_path`` is set, a ``.post_fit.json`` sidecar
        is written next to the CSV with val metrics for the deployed
        model. Essential for measuring SWA / EMA effects, which are
        invisible in the per-epoch CSV (that tracks the base model)."""
        X, y = synthetic_binary
        path = tmp_path / "m.csv"
        self._fit_mlp(X, y, metrics_log_path=path)
        sidecar = tmp_path / "m.csv.post_fit.json"
        assert sidecar.exists()
        payload = json.loads(sidecar.read_text())
        assert "val_loss" in payload
        assert "val_auroc" in payload
        assert "weights_source" in payload
        assert payload["weights_source"] == "best_val"
        assert np.isfinite(payload["val_loss"])
        assert 0.0 <= payload["val_auroc"] <= 1.0

    def test_post_fit_sidecar_marks_ema_source(self, synthetic_binary, tmp_path):
        """EMA training => sidecar records ``weights_source=\"ema\"``."""
        X, y = synthetic_binary
        path = tmp_path / "m.csv"
        self._fit_mlp(X, y, metrics_log_path=path, ema_decay=0.99)
        sidecar = tmp_path / "m.csv.post_fit.json"
        payload = json.loads(sidecar.read_text())
        assert payload["weights_source"] == "ema"

    def test_post_fit_sidecar_marks_swa_source(self, synthetic_binary, tmp_path):
        """SWA training (when it actually kicks in) => ``weights_source=\"swa\"``."""
        X, y = synthetic_binary
        path = tmp_path / "m.csv"
        self._fit_mlp(X, y, metrics_log_path=path, swa_start_epoch=0)
        sidecar = tmp_path / "m.csv.post_fit.json"
        payload = json.loads(sidecar.read_text())
        assert payload["weights_source"] == "swa"


class TestInputTransform:
    """`input_transform` modes on make_loaders + BaseSpectralClassifier."""

    def _fit_mode(self, mode: str, X, y, **kw):
        return MaldiMLPClassifier(
            hidden_dim=8,
            head_dims=(4,),
            epochs=1,
            batch_size=8,
            random_state=0,
            input_transform=mode,
            **kw,
        ).fit(X, y)

    def test_none_is_identity(self, synthetic_binary):
        X, y = synthetic_binary
        clf = self._fit_mode("none", X, y)
        state = clf.input_transform_state_
        assert state["mode"] == "none"
        assert "mean" not in state or state.get("mean") is None

    def test_standardize_stores_per_bin_stats(self, synthetic_binary):
        X, y = synthetic_binary
        clf = self._fit_mode("standardize", X, y)
        state = clf.input_transform_state_
        assert state["mode"] == "standardize"
        assert state["mean"].shape == (X.shape[1],)
        assert state["std"].shape == (X.shape[1],)

    def test_log1p_is_stateless(self, synthetic_binary):
        X, y = synthetic_binary
        clf = self._fit_mode("log1p", X, y)
        state = clf.input_transform_state_
        assert state == {"mode": "log1p"}

    def test_robust_stores_median_iqr(self, synthetic_binary):
        X, y = synthetic_binary
        clf = self._fit_mode("robust", X, y)
        state = clf.input_transform_state_
        assert state["mode"] == "robust"
        assert state["median"].shape == (X.shape[1],)
        assert state["iqr"].shape == (X.shape[1],)
        assert (state["iqr"] > 0).all()

    def test_log1p_standardize_stores_stats_on_log_scale(self, synthetic_binary):
        X, y = synthetic_binary
        clf = self._fit_mode("log1p+standardize", X, y)
        state = clf.input_transform_state_
        assert state["mode"] == "log1p+standardize"
        assert state["mean"].shape == (X.shape[1],)
        clf_plain = self._fit_mode("standardize", X, y)
        assert not np.allclose(state["mean"], clf_plain.input_transform_state_["mean"])

    def test_standardize_legacy_bool_still_works(self, synthetic_binary):
        """`standardize=True` alone (no input_transform) retains current behavior."""
        X, y = synthetic_binary
        clf = MaldiMLPClassifier(
            hidden_dim=8,
            head_dims=(4,),
            epochs=1,
            batch_size=8,
            standardize=True,
            random_state=0,
        ).fit(X, y)
        assert clf.input_transform_state_["mode"] == "standardize"
        assert clf.feature_mean_ is not None

    def test_input_transform_wins_over_standardize(self, synthetic_binary):
        """Explicit `input_transform` overrides the legacy `standardize` bool."""
        X, y = synthetic_binary
        clf = MaldiMLPClassifier(
            hidden_dim=8,
            head_dims=(4,),
            epochs=1,
            batch_size=8,
            standardize=True,
            input_transform="log1p",
            random_state=0,
        ).fit(X, y)
        assert clf.input_transform_state_["mode"] == "log1p"

    def test_unknown_mode_raises(self, synthetic_binary):
        X, y = synthetic_binary
        with pytest.raises(ValueError, match="input_transform"):
            self._fit_mode("bogus", X, y)

    def test_predict_applies_fitted_transform(self, synthetic_binary):
        """`predict_proba` routes through the fitted state at inference."""
        X, y = synthetic_binary
        clf = self._fit_mode("log1p", X, y)
        proba = clf.predict_proba(X[:4])
        assert proba.shape == (4, 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_save_load_roundtrip_preserves_transform_state(
        self, synthetic_binary, tmp_path
    ):
        X, y = synthetic_binary
        clf = self._fit_mode("robust", X, y)
        path = tmp_path / "clf"
        clf.save(path)
        reloaded = MaldiMLPClassifier.load(path)
        assert reloaded.input_transform_state_["mode"] == "robust"
        np.testing.assert_array_equal(
            reloaded.input_transform_state_["median"],
            clf.input_transform_state_["median"],
        )
        np.testing.assert_allclose(
            clf.predict_proba(X[:4]), reloaded.predict_proba(X[:4]), atol=1e-5
        )


class TestMixupCutmix:
    """MixUp / CutMix wiring on ``BaseSpectralClassifier``."""

    def test_defaults_are_off(self):
        clf = MaldiMLPClassifier()
        assert clf.mixup_alpha == 0.0
        assert clf.cutmix_alpha == 0.0

    def test_fit_with_mixup(self, synthetic_binary):
        X, y = synthetic_binary
        clf = MaldiMLPClassifier(
            hidden_dim=8,
            head_dims=(4,),
            epochs=2,
            batch_size=8,
            mixup_alpha=0.2,
            random_state=0,
        ).fit(X, y)
        assert clf.predict(X).shape == (len(X),)

    def test_fit_with_cutmix(self, synthetic_binary):
        X, y = synthetic_binary
        clf = MaldiMLPClassifier(
            hidden_dim=8,
            head_dims=(4,),
            epochs=2,
            batch_size=8,
            cutmix_alpha=1.0,
            random_state=0,
        ).fit(X, y)
        assert clf.predict(X).shape == (len(X),)

    def test_fit_with_both(self, synthetic_binary):
        """When both are enabled, each batch picks one randomly - both branches must be sound."""
        X, y = synthetic_binary
        clf = MaldiMLPClassifier(
            hidden_dim=8,
            head_dims=(4,),
            epochs=2,
            batch_size=8,
            mixup_alpha=0.2,
            cutmix_alpha=1.0,
            random_state=0,
        ).fit(X, y)
        assert clf.predict(X).shape == (len(X),)

    def test_mixup_with_focal_loss(self, synthetic_binary):
        """FocalLoss supports soft targets, so mixup + focal must work."""
        X, y = synthetic_binary
        clf = MaldiMLPClassifier(
            hidden_dim=8,
            head_dims=(4,),
            epochs=2,
            batch_size=8,
            loss="focal",
            focal_gamma=2.0,
            mixup_alpha=0.2,
            random_state=0,
        ).fit(X, y)
        assert clf.predict(X).shape == (len(X),)

    def test_mixup_seeded_is_reproducible(self, synthetic_binary):
        """Two fits with the same seed + mixup_alpha must produce identical outputs."""
        X, y = synthetic_binary
        clf1 = MaldiMLPClassifier(
            hidden_dim=8,
            head_dims=(4,),
            epochs=2,
            batch_size=8,
            mixup_alpha=0.2,
            random_state=0,
        ).fit(X, y)
        clf2 = MaldiMLPClassifier(
            hidden_dim=8,
            head_dims=(4,),
            epochs=2,
            batch_size=8,
            mixup_alpha=0.2,
            random_state=0,
        ).fit(X, y)
        np.testing.assert_allclose(
            clf1.predict_proba(X[:4]), clf2.predict_proba(X[:4]), atol=1e-5
        )


class TestEMA:
    """Weight-EMA wiring on ``BaseSpectralClassifier``."""

    def test_default_is_off(self):
        clf = MaldiMLPClassifier()
        assert clf.ema_decay is None

    def test_fit_with_ema(self, synthetic_binary):
        X, y = synthetic_binary
        clf = MaldiMLPClassifier(
            hidden_dim=8,
            head_dims=(4,),
            epochs=2,
            batch_size=8,
            ema_decay=0.99,
            random_state=0,
        ).fit(X, y)
        assert clf.predict(X).shape == (len(X),)

    def test_invalid_decay_raises(self, synthetic_binary):
        X, y = synthetic_binary
        clf = MaldiMLPClassifier(
            hidden_dim=8,
            head_dims=(4,),
            epochs=1,
            batch_size=8,
            ema_decay=1.0,
            random_state=0,
        )
        with pytest.raises(ValueError, match="ema_decay"):
            clf.fit(X, y)

    def test_ema_changes_output_vs_plain(self, synthetic_binary):
        """A nonzero ema_decay produces different final weights than no EMA."""
        X, y = synthetic_binary
        common = dict(
            hidden_dim=8,
            head_dims=(4,),
            epochs=3,
            batch_size=8,
            random_state=0,
        )
        plain = MaldiMLPClassifier(**common).fit(X, y)
        emaed = MaldiMLPClassifier(ema_decay=0.5, **common).fit(X, y)
        assert not np.allclose(
            plain.predict_proba(X), emaed.predict_proba(X), atol=1e-5
        )

    def test_save_load_preserves_ema_weights(self, synthetic_binary, tmp_path):
        """After EMA training, save/load must reproduce predictions exactly."""
        X, y = synthetic_binary
        clf = MaldiMLPClassifier(
            hidden_dim=8,
            head_dims=(4,),
            epochs=2,
            batch_size=8,
            ema_decay=0.99,
            random_state=0,
        ).fit(X, y)
        path = tmp_path / "ema_model"
        clf.save(path)
        from maldideepkit.base.classifier import BaseSpectralClassifier

        loaded = BaseSpectralClassifier.load(path)
        np.testing.assert_allclose(
            clf.predict_proba(X[:4]),
            loaded.predict_proba(X[:4]),
            atol=1e-6,
        )


class TestInitRetry:
    """``retry_on_val_auroc_below`` reinitialises the model when a fit
    ends below the threshold. Designed for architectures like the
    Transformer that occasionally collapse to a trivial basin on
    small-data folds; unused on well-behaved architectures because
    the threshold never triggers.
    """

    def test_defaults_disable_retry(self):
        clf = MaldiMLPClassifier()
        assert clf.retry_on_val_auroc_below is None
        assert clf.max_retries == 2

    def test_retry_not_triggered_when_val_auroc_high(
        self, synthetic_binary, monkeypatch
    ):
        """Healthy fits should call ``_build_model`` exactly once."""
        X, y = synthetic_binary

        call_count = {"n": 0}
        import maldideepkit.attention.mlp as _mod

        original_build = _mod.MaldiMLPClassifier._build_model

        def spy_build(self):
            call_count["n"] += 1
            return original_build(self)

        monkeypatch.setattr(_mod.MaldiMLPClassifier, "_build_model", spy_build)

        clf = MaldiMLPClassifier(
            hidden_dim=16,
            head_dims=(8,),
            epochs=2,
            batch_size=8,
            retry_on_val_auroc_below=0.5,  # low threshold -> very likely to be met
            max_retries=2,
            random_state=0,
        ).fit(X, y)
        assert clf.predict(X).shape == (len(X),)
        assert call_count["n"] == 1

    def test_retry_fires_and_stops_at_max_retries(self, synthetic_binary, monkeypatch):
        """If threshold is impossibly high, we must fit exactly
        ``1 + max_retries`` times and still return a fitted classifier.
        """
        X, y = synthetic_binary

        call_count = {"n": 0}
        import maldideepkit.attention.mlp as _mod

        original_build = _mod.MaldiMLPClassifier._build_model

        def spy_build(self):
            call_count["n"] += 1
            return original_build(self)

        monkeypatch.setattr(_mod.MaldiMLPClassifier, "_build_model", spy_build)

        clf = MaldiMLPClassifier(
            hidden_dim=16,
            head_dims=(8,),
            epochs=2,
            batch_size=8,
            retry_on_val_auroc_below=1.01,  # impossible threshold
            max_retries=2,
            random_state=0,
        ).fit(X, y)
        # Exactly 1 initial + 2 retries = 3 model builds
        assert call_count["n"] == 3
        assert clf.predict(X).shape == (len(X),)

    def test_retry_fires_with_max_retries_zero_does_not_retry(
        self, synthetic_binary, monkeypatch
    ):
        """``max_retries=0`` means the threshold check is inert (no retry)."""
        X, y = synthetic_binary

        call_count = {"n": 0}
        import maldideepkit.attention.mlp as _mod

        original_build = _mod.MaldiMLPClassifier._build_model

        def spy_build(self):
            call_count["n"] += 1
            return original_build(self)

        monkeypatch.setattr(_mod.MaldiMLPClassifier, "_build_model", spy_build)

        MaldiMLPClassifier(
            hidden_dim=16,
            head_dims=(8,),
            epochs=2,
            batch_size=8,
            retry_on_val_auroc_below=1.01,
            max_retries=0,
            random_state=0,
        ).fit(X, y)
        assert call_count["n"] == 1
