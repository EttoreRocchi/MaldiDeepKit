"""Unit tests for ``maldideepkit.utils.loss.FocalLoss``."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from maldideepkit.utils import FocalLoss


class TestConstructorValidation:
    """Constructor must reject invalid hyperparameters."""

    def test_negative_gamma_raises(self):
        with pytest.raises(ValueError, match="gamma must be >= 0"):
            FocalLoss(gamma=-0.1)

    def test_label_smoothing_at_one_raises(self):
        with pytest.raises(ValueError, match=r"label_smoothing must be in \[0, 1\)"):
            FocalLoss(label_smoothing=1.0)

    def test_label_smoothing_above_one_raises(self):
        with pytest.raises(ValueError, match=r"label_smoothing must be in \[0, 1\)"):
            FocalLoss(label_smoothing=1.5)

    def test_negative_label_smoothing_raises(self):
        with pytest.raises(ValueError, match=r"label_smoothing must be in \[0, 1\)"):
            FocalLoss(label_smoothing=-0.01)

    def test_unknown_reduction_raises(self):
        with pytest.raises(ValueError, match="reduction must be"):
            FocalLoss(reduction="median")

    def test_zero_gamma_is_allowed(self):
        FocalLoss(gamma=0.0)

    def test_gamma_stored_as_float(self):
        loss = FocalLoss(gamma=3)
        assert isinstance(loss.gamma, float)
        assert loss.gamma == 3.0


class TestClassWeightBuffer:
    """``weight`` is registered as a non-persistent, detached buffer."""

    def test_class_weight_registered_as_buffer(self):
        w = torch.tensor([0.5, 2.0])
        loss = FocalLoss(weight=w)
        assert "class_weight" in dict(loss.named_buffers())
        assert torch.equal(loss.class_weight, w)

    def test_class_weight_buffer_is_non_persistent(self):
        """Non-persistent buffers must not appear in the state_dict."""
        w = torch.tensor([0.5, 2.0])
        loss = FocalLoss(weight=w)
        assert "class_weight" not in loss.state_dict()

    def test_class_weight_is_cloned(self):
        """Mutating the input tensor afterward must not affect the loss."""
        w = torch.tensor([1.0, 1.0])
        loss = FocalLoss(weight=w, gamma=0.0)
        w[1] = 999.0  # mutate caller's tensor
        logits = torch.tensor([[2.0, -2.0], [-2.0, 2.0]])
        target = torch.tensor([0, 1])
        out = loss(logits, target)
        ce = F.cross_entropy(logits, target)
        assert torch.allclose(out, ce, atol=1e-6)

    def test_no_class_weight_buffer_is_none(self):
        loss = FocalLoss()
        assert loss.class_weight is None


class TestHardTargetGammaZero:
    """``gamma=0`` and no smoothing reduces to standard cross-entropy."""

    def test_matches_cross_entropy(self):
        torch.manual_seed(0)
        logits = torch.randn(16, 3)
        target = torch.randint(0, 3, (16,))
        ce = F.cross_entropy(logits, target)
        focal = FocalLoss(gamma=0.0)(logits, target)
        assert torch.allclose(ce, focal, atol=1e-6)

    def test_class_weighted_matches_cross_entropy(self):
        weight = torch.tensor([1.0, 10.0])
        logits = torch.tensor([[5.0, -5.0], [-5.0, 5.0]])
        target = torch.tensor([0, 0])
        focal = FocalLoss(weight=weight, gamma=0.0)(logits, target)
        ce = F.cross_entropy(logits, target, weight=weight)
        assert torch.allclose(focal, ce, atol=1e-6)


class TestHardTargetFocalTerm:
    """``gamma > 0`` down-weights easy examples relative to CE."""

    def test_gamma_positive_below_cross_entropy(self):
        logits = torch.tensor([[2.0, 0.0]])
        target = torch.tensor([0])
        ce = F.cross_entropy(logits, target)
        focal = FocalLoss(gamma=2.0)(logits, target)
        assert focal < ce

    def test_higher_gamma_reduces_loss_for_easy_example(self):
        logits = torch.tensor([[3.0, -3.0]])  # very confident, correct
        target = torch.tensor([0])
        loss_low = FocalLoss(gamma=1.0)(logits, target)
        loss_high = FocalLoss(gamma=5.0)(logits, target)
        assert loss_high < loss_low

    def test_gamma_does_not_change_loss_at_pt_one(self):
        """At p_t = 1, focal == CE == 0 regardless of gamma."""
        logits = torch.tensor([[100.0, -100.0]])
        target = torch.tensor([0])
        for gamma in (0.0, 1.0, 5.0):
            assert FocalLoss(gamma=gamma)(logits, target).item() == pytest.approx(
                0.0, abs=1e-6
            )


class TestSoftTargets:
    """Soft (probability) targets path."""

    def test_one_hot_matches_hard(self):
        torch.manual_seed(0)
        logits = torch.randn(8, 3)
        target = torch.randint(0, 3, (8,))
        target_oh = F.one_hot(target, 3).float()
        hard = FocalLoss(gamma=2.0)(logits, target)
        soft = FocalLoss(gamma=2.0)(logits, target_oh)
        assert torch.allclose(hard, soft, atol=1e-6)

    def test_mixed_target_between_extremes(self):
        logits = torch.tensor([[2.0, -2.0]])
        t_class0 = torch.tensor([[1.0, 0.0]])
        t_class1 = torch.tensor([[0.0, 1.0]])
        t_mix = torch.tensor([[0.5, 0.5]])
        loss_0 = FocalLoss(gamma=2.0)(logits, t_class0)
        loss_1 = FocalLoss(gamma=2.0)(logits, t_class1)
        loss_mix = FocalLoss(gamma=2.0)(logits, t_mix)
        lo, hi = (loss_0, loss_1) if loss_0 < loss_1 else (loss_1, loss_0)
        assert lo <= loss_mix <= hi

    def test_soft_target_class_weight_uniform(self):
        """Identical sample weights collapse weighted-mean to the unweighted mean."""
        weight = torch.tensor([0.5, 2.0])
        logits = torch.tensor([[1.0, -1.0], [-1.0, 1.0]])
        target_oh = torch.tensor([[1.0, 0.0], [1.0, 0.0]])  # both class 0
        weighted = FocalLoss(weight=weight, gamma=0.0)(logits, target_oh)
        unweighted = FocalLoss(gamma=0.0)(logits, target_oh)
        assert torch.allclose(weighted, unweighted, atol=1e-6)

    def test_soft_target_gamma_zero_matches_kl_against_log_softmax(self):
        """At gamma=0 the soft-target path is the standard cross-entropy with soft labels."""
        torch.manual_seed(0)
        logits = torch.randn(8, 4)
        soft = torch.softmax(torch.randn(8, 4), dim=-1)
        focal = FocalLoss(gamma=0.0)(logits, soft)
        log_probs = F.log_softmax(logits, dim=-1)
        manual = -(soft * log_probs).sum(dim=-1).mean()
        assert torch.allclose(focal, manual, atol=1e-6)

    def test_soft_target_label_smoothing_is_ignored(self):
        """``label_smoothing`` must not mutate the soft-target distribution."""
        torch.manual_seed(0)
        logits = torch.randn(4, 3)
        soft = torch.softmax(torch.randn(4, 3), dim=-1)
        no_smooth = FocalLoss(gamma=2.0, label_smoothing=0.0)(logits, soft)
        smooth = FocalLoss(gamma=2.0, label_smoothing=0.2)(logits, soft)
        assert torch.allclose(no_smooth, smooth, atol=1e-6)


class TestLabelSmoothing:
    """Label smoothing path (``label_smoothing > 0`` with hard targets)."""

    def test_gamma_zero_matches_pytorch_cross_entropy(self):
        torch.manual_seed(0)
        logits = torch.randn(16, 4)
        target = torch.randint(0, 4, (16,))
        focal = FocalLoss(gamma=0.0, label_smoothing=0.1)(logits, target)
        ce = F.cross_entropy(logits, target, label_smoothing=0.1)
        assert torch.allclose(focal, ce, atol=1e-6)

    def test_smoothing_increases_loss_on_confident_correct_predictions(self):
        """Smoothed targets are never one-hot, so a perfectly-correct logit still incurs loss."""
        logits = torch.tensor([[10.0, -10.0, -10.0]])
        target = torch.tensor([0])
        no_smooth = FocalLoss(gamma=0.0, label_smoothing=0.0)(logits, target)
        smooth = FocalLoss(gamma=0.0, label_smoothing=0.1)(logits, target)
        assert smooth > no_smooth

    def test_label_smoothing_with_class_weight_uses_per_sample_weighting(self):
        """The label-smoothing path weights each sample by ``weight[y_i]``,
        not per-class. This differs from PyTorch CE's per-class weighting
        but matches the documented FocalLoss convention.
        """
        weight = torch.tensor([1.0, 4.0])
        logits = torch.tensor([[2.0, -2.0], [-2.0, 2.0]])
        target = torch.tensor([0, 1])

        per_sample = FocalLoss(
            weight=weight, gamma=0.0, label_smoothing=0.1, reduction="none"
        )(logits, target)
        meaned = FocalLoss(
            weight=weight, gamma=0.0, label_smoothing=0.1, reduction="mean"
        )(logits, target)
        denom = weight[target].sum()
        assert torch.allclose(meaned, per_sample.sum() / denom, atol=1e-6)


class TestReduction:
    """``reduction`` parameter shape and consistency."""

    def test_none_returns_per_sample_tensor(self):
        torch.manual_seed(0)
        logits = torch.randn(5, 3)
        target = torch.randint(0, 3, (5,))
        loss = FocalLoss(gamma=2.0, reduction="none")(logits, target)
        assert loss.shape == (5,)

    def test_sum_equals_sum_of_none(self):
        torch.manual_seed(0)
        logits = torch.randn(5, 3)
        target = torch.randint(0, 3, (5,))
        per_sample = FocalLoss(gamma=2.0, reduction="none")(logits, target)
        summed = FocalLoss(gamma=2.0, reduction="sum")(logits, target)
        assert torch.allclose(summed, per_sample.sum(), atol=1e-6)

    def test_mean_equals_mean_of_none(self):
        torch.manual_seed(0)
        logits = torch.randn(5, 3)
        target = torch.randint(0, 3, (5,))
        per_sample = FocalLoss(gamma=2.0, reduction="none")(logits, target)
        meaned = FocalLoss(gamma=2.0, reduction="mean")(logits, target)
        assert torch.allclose(meaned, per_sample.mean(), atol=1e-6)

    def test_mean_with_class_weight_uses_weighted_denominator(self):
        """Weighted-mean denominator is ``Σ_i w[y_i]``, matching CE."""
        weight = torch.tensor([1.0, 4.0])
        logits = torch.tensor([[2.0, -2.0], [-2.0, 2.0]])
        target = torch.tensor([0, 1])
        per_sample = FocalLoss(weight=weight, gamma=0.0, reduction="none")(
            logits, target
        )
        meaned = FocalLoss(weight=weight, gamma=0.0, reduction="mean")(logits, target)
        # per-sample loss already includes the gather'd weight
        denom = weight[target].sum()
        assert torch.allclose(meaned, per_sample.sum() / denom, atol=1e-6)


class TestGradients:
    """Loss is differentiable w.r.t. logits in every code path."""

    @pytest.mark.parametrize(
        "gamma,label_smoothing", [(0.0, 0.0), (2.0, 0.0), (2.0, 0.1)]
    )
    def test_gradient_flows_through_hard_targets(self, gamma, label_smoothing):
        torch.manual_seed(0)
        logits = torch.randn(4, 3, requires_grad=True)
        target = torch.randint(0, 3, (4,))
        loss = FocalLoss(gamma=gamma, label_smoothing=label_smoothing)(logits, target)
        loss.backward()
        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()
        assert logits.grad.abs().sum() > 0

    def test_gradient_flows_through_soft_targets(self):
        torch.manual_seed(0)
        logits = torch.randn(4, 3, requires_grad=True)
        target = torch.softmax(torch.randn(4, 3), dim=-1)
        loss = FocalLoss(gamma=2.0)(logits, target)
        loss.backward()
        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()
        assert logits.grad.abs().sum() > 0


class TestExtraRepr:
    def test_repr_contains_hyperparameters(self):
        loss = FocalLoss(gamma=1.5, label_smoothing=0.05, reduction="sum")
        text = repr(loss)
        assert "gamma=1.5" in text
        assert "label_smoothing=0.05" in text
        assert "'sum'" in text
