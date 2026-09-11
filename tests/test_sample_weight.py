"""Tests for the ``sample_weight`` keyword on :meth:`BaseSpectralClassifier.fit`.

``sample_weight`` applies per-sample weights inside the loss (sklearn-style
weighted mean) and is split into a val portion that flows through the
early-stopping signal, the retry-on-collapse guardrail, AUROC reporting,
``tune_threshold`` and ``calibrate_temperature``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from maldideepkit import MaldiMLPClassifier
from maldideepkit.augment.mixing import apply_cutmix, apply_mixup, to_one_hot
from maldideepkit.utils.loss import _gather_cw_per_sample, _weighted_loss_reduce


def _tiny_mlp(**overrides) -> MaldiMLPClassifier:
    defaults = dict(
        input_dim=128,
        hidden_dim=8,
        head_dims=(8,),
        epochs=3,
        batch_size=8,
        val_fraction=0.25,
        warmup_epochs=0,
        early_stopping_patience=3,
        random_state=0,
        device="cpu",
    )
    defaults.update(overrides)
    return MaldiMLPClassifier(**defaults)


class TestWeightedLossReduce:
    def test_unweighted_unweighted_matches_mean(self):
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        out = _weighted_loss_reduce(x, None, None)
        torch.testing.assert_close(out, torch.tensor(2.5))

    def test_class_weight_only_matches_pytorch_mean_reduction(self):
        # Per-sample already includes cw factor; denom = cw_per_sample.sum().
        per_sample = torch.tensor([2.0, 4.0, 6.0])  # = cw[y] * l
        cw_per_sample = torch.tensor([2.0, 1.0, 3.0])  # = cw[y]
        out = _weighted_loss_reduce(per_sample, None, cw_per_sample)
        # sum(2+4+6) / sum(2+1+3) = 12 / 6 = 2.0
        torch.testing.assert_close(out, torch.tensor(2.0))

    def test_sample_weight_only_matches_sklearn_convention(self):
        per_sample = torch.tensor([1.0, 2.0, 3.0])
        sw = torch.tensor([1.0, 2.0, 1.0])
        out = _weighted_loss_reduce(per_sample, sw, None)
        # (1*1 + 2*2 + 1*3) / (1+2+1) = 8/4 = 2.0
        torch.testing.assert_close(out, torch.tensor(2.0))

    def test_compose_sw_cw_duplication_equivalence(self):
        # sklearn invariant: doubling sample_weight is equivalent to
        # duplicating that row.
        per_sample = torch.tensor([1.0, 2.0, 3.0])
        cw_per_sample = torch.tensor([2.0, 1.0, 3.0])
        sw_doubled = torch.tensor([2.0, 1.0, 1.0])
        a = _weighted_loss_reduce(per_sample, sw_doubled, cw_per_sample)

        # Equivalent: duplicate the first row.
        per_sample_dup = torch.tensor([1.0, 1.0, 2.0, 3.0])
        cw_dup = torch.tensor([2.0, 2.0, 1.0, 3.0])
        b = _weighted_loss_reduce(per_sample_dup, None, cw_dup)
        torch.testing.assert_close(a, b)

    def test_zero_weight_row_is_ignored(self):
        per_sample = torch.tensor([1.0, 100.0, 3.0])  # row 1 should not matter
        sw = torch.tensor([1.0, 0.0, 1.0])
        out = _weighted_loss_reduce(per_sample, sw, None)
        torch.testing.assert_close(out, torch.tensor(2.0))


# _gather_cw_per_sample -------------------------------------------------------


class TestGatherCwPerSample:
    def test_none_class_weight(self):
        target = torch.tensor([0, 1, 0])
        assert _gather_cw_per_sample(target, None) is None

    def test_int_target(self):
        target = torch.tensor([0, 1, 1, 0])
        cw = torch.tensor([2.0, 3.0])
        out = _gather_cw_per_sample(target, cw)
        torch.testing.assert_close(out, torch.tensor([2.0, 3.0, 3.0, 2.0]))

    def test_soft_target(self):
        # 2-class soft targets
        target = torch.tensor([[0.5, 0.5], [1.0, 0.0], [0.0, 1.0]])
        cw = torch.tensor([2.0, 4.0])
        out = _gather_cw_per_sample(target, cw)
        # row 0: 0.5*2 + 0.5*4 = 3
        # row 1: 1*2 = 2
        # row 2: 1*4 = 4
        torch.testing.assert_close(out, torch.tensor([3.0, 2.0, 4.0]))


class TestMixApplyWithWeight:
    def test_apply_mixup_returns_three_tuple_with_sw(self):
        gen = torch.Generator().manual_seed(0)
        x = torch.randn(4, 16)
        y = to_one_hot(torch.tensor([0, 1, 0, 1]), 2)
        sw = torch.tensor([1.0, 2.0, 3.0, 4.0])
        out = apply_mixup(x, y, alpha=0.5, generator=gen, sample_weight=sw)
        assert len(out) == 3
        x_mixed, y_mixed, sw_mixed = out
        assert x_mixed.shape == x.shape
        assert y_mixed.shape == y.shape
        assert sw_mixed.shape == sw.shape
        # sw should be a convex combination -> within the convex hull
        assert sw_mixed.min() >= 1.0
        assert sw_mixed.max() <= 4.0

    def test_apply_mixup_two_tuple_when_no_sw(self):
        x = torch.randn(4, 16)
        y = to_one_hot(torch.tensor([0, 1, 0, 1]), 2)
        out = apply_mixup(x, y, alpha=0.5, generator=torch.Generator().manual_seed(0))
        assert len(out) == 2

    def test_apply_cutmix_returns_three_tuple_with_sw(self):
        gen = torch.Generator().manual_seed(0)
        x = torch.randn(4, 16)
        y = to_one_hot(torch.tensor([0, 1, 0, 1]), 2)
        sw = torch.ones(4)
        out = apply_cutmix(x, y, alpha=1.0, generator=gen, sample_weight=sw)
        assert len(out) == 3


@pytest.fixture
def synthetic_binary_local():
    rng = np.random.default_rng(0)
    n, d = 64, 128
    X = rng.standard_normal((n, d)).astype(np.float32)
    y = rng.integers(0, 2, size=n)
    return X, y


class TestFitSampleWeight:
    def test_all_ones_matches_no_weights(self, synthetic_binary_local):
        X, y = synthetic_binary_local

        clf_a = _tiny_mlp()
        clf_a.fit(X, y)

        clf_b = _tiny_mlp()
        clf_b.fit(X, y, sample_weight=np.ones(len(y), dtype=np.float32))

        # Same seed + same data + uniform weights should yield identical
        # trajectories.
        p_a = next(clf_a.model_.parameters()).detach().cpu().numpy()
        p_b = next(clf_b.model_.parameters()).detach().cpu().numpy()
        np.testing.assert_allclose(p_a, p_b, rtol=0, atol=1e-6)

    def test_non_uniform_weights_diverge_from_unweighted(self, synthetic_binary_local):
        X, y = synthetic_binary_local
        sw = np.where(y == 1, 5.0, 1.0).astype(np.float32)

        clf_a = _tiny_mlp()
        clf_a.fit(X, y)

        clf_b = _tiny_mlp()
        clf_b.fit(X, y, sample_weight=sw)

        p_a = next(clf_a.model_.parameters()).detach().cpu().numpy()
        p_b = next(clf_b.model_.parameters()).detach().cpu().numpy()
        # Weights actually changed the training objective.
        assert not np.allclose(p_a, p_b, atol=1e-6)

    def test_class_weight_composes_with_sample_weight(self, synthetic_binary_local):
        X, y = synthetic_binary_local
        sw = np.ones(len(y), dtype=np.float32)

        clf = _tiny_mlp(class_weight="balanced")
        clf.fit(X, y, sample_weight=sw)
        assert hasattr(clf, "model_")

    def test_mixup_with_sample_weight_smoke(self, synthetic_binary_local):
        X, y = synthetic_binary_local
        sw = np.where(y == 1, 2.0, 1.0).astype(np.float32)
        clf = _tiny_mlp(mixup_alpha=0.2)
        clf.fit(X, y, sample_weight=sw)
        assert hasattr(clf, "model_")

    def test_cutmix_with_sample_weight_smoke(self, synthetic_binary_local):
        X, y = synthetic_binary_local
        sw = np.where(y == 1, 2.0, 1.0).astype(np.float32)
        clf = _tiny_mlp(cutmix_alpha=1.0)
        clf.fit(X, y, sample_weight=sw)
        assert hasattr(clf, "model_")

    def test_sam_with_sample_weight_smoke(self, synthetic_binary_local):
        X, y = synthetic_binary_local
        sw = np.where(y == 1, 2.0, 1.0).astype(np.float32)
        clf = _tiny_mlp(use_sam=True, sam_rho=0.05)
        clf.fit(X, y, sample_weight=sw)
        assert hasattr(clf, "model_")

    def test_threshold_and_temperature_under_sw(self, synthetic_binary_local):
        X, y = synthetic_binary_local
        sw = np.where(y == 1, 3.0, 1.0).astype(np.float32)
        clf = _tiny_mlp(
            tune_threshold=True,
            calibrate_temperature=True,
            min_val_auroc_for_threshold_tune=0.0,
        )
        clf.fit(X, y, sample_weight=sw)
        # Both post-hoc artifacts should be populated; tune_threshold's gate
        # is disabled so a threshold gets set regardless of val AUROC.
        assert clf.temperature_ is not None and clf.temperature_ > 0
        assert clf.threshold_ is not None and 0.0 < clf.threshold_ < 1.0


class TestSampleWeightValidation:
    def test_wrong_shape_raises(self, synthetic_binary_local):
        X, y = synthetic_binary_local
        clf = _tiny_mlp()
        with pytest.raises(ValueError, match="sample_weight"):
            clf.fit(X, y, sample_weight=np.ones(len(y) + 1, dtype=np.float32))

    def test_negative_weight_raises(self, synthetic_binary_local):
        X, y = synthetic_binary_local
        sw = np.ones(len(y), dtype=np.float32)
        sw[0] = -1.0
        clf = _tiny_mlp()
        with pytest.raises(ValueError, match="non-negative"):
            clf.fit(X, y, sample_weight=sw)

    def test_nan_weight_raises(self, synthetic_binary_local):
        X, y = synthetic_binary_local
        sw = np.ones(len(y), dtype=np.float32)
        sw[0] = np.nan
        clf = _tiny_mlp()
        with pytest.raises(ValueError, match="non-finite"):
            clf.fit(X, y, sample_weight=sw)


class TestSampleWeightSignature:
    def test_keyword_only(self, synthetic_binary_local):
        X, y = synthetic_binary_local
        clf = _tiny_mlp()
        # Passing sample_weight positionally must error.
        with pytest.raises(TypeError):
            clf.fit(X, y, np.ones(len(y), dtype=np.float32))  # noqa: FBT003

    def test_default_is_none(self, synthetic_binary_local):
        # fit(X, y) without sample_weight still works; smoke check.
        X, y = synthetic_binary_local
        clf = _tiny_mlp()
        clf.fit(X, y)
        assert hasattr(clf, "model_")


class TestScalarLossPassthrough:
    """A mean/sum-reduced criterion (the pre-0.3.0 public train_loop
    contract) must flow through _weighted_loss_reduce unchanged, never
    silently divided a second time."""

    def test_scalar_loss_is_returned_as_is(self):
        from maldideepkit.utils.loss import _weighted_loss_reduce

        scalar = torch.tensor(0.7)
        out = _weighted_loss_reduce(scalar, None, torch.ones(32))
        assert out.item() == pytest.approx(0.7)
        out = _weighted_loss_reduce(scalar, None, None)
        assert out.item() == pytest.approx(0.7)

    def test_scalar_loss_with_sample_weight_raises(self):
        from maldideepkit.utils.loss import _weighted_loss_reduce

        with pytest.raises(ValueError, match="reduction='none'"):
            _weighted_loss_reduce(torch.tensor(0.7), torch.ones(4), None)
