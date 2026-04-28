"""Tests for the spectrum-augmentation module."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from maldideepkit import MaldiMLPClassifier
from maldideepkit.augment import SpectrumAugment, apply_cutmix, apply_mixup, to_one_hot


class TestSpectrumAugmentIdentity:
    """All-zero parameters => identity transform (bit-exact)."""

    def test_zero_all(self):
        aug = SpectrumAugment()
        x = torch.randn(4, 16)
        torch.testing.assert_close(aug(x), x)

    def test_repr_includes_params(self):
        aug = SpectrumAugment(noise_std=0.1)
        assert "noise_std=0.1" in repr(aug)


class TestSpectrumAugmentParameters:
    def test_negative_noise_raises(self):
        with pytest.raises(ValueError, match="noise_std"):
            SpectrumAugment(noise_std=-1.0)

    def test_invalid_jitter_raises(self):
        with pytest.raises(ValueError, match="intensity_jitter"):
            SpectrumAugment(intensity_jitter=1.0)

    def test_invalid_dropout_raises(self):
        with pytest.raises(ValueError, match="peak_dropout_rate"):
            SpectrumAugment(peak_dropout_rate=1.0)

    def test_negative_mz_shift_raises(self):
        with pytest.raises(ValueError, match="mz_shift_max_bins"):
            SpectrumAugment(mz_shift_max_bins=-1)

    def test_negative_mz_warp_raises(self):
        with pytest.raises(ValueError, match="mz_warp_max_bins"):
            SpectrumAugment(mz_warp_max_bins=-1)

    def test_negative_mz_warp_knots_raises(self):
        with pytest.raises(ValueError, match="mz_warp_n_knots"):
            SpectrumAugment(mz_warp_n_knots=-1)

    def test_negative_blur_sigma_raises(self):
        with pytest.raises(ValueError, match="blur_sigma"):
            SpectrumAugment(blur_sigma=-0.5)


class TestSpectrumAugmentNoise:
    def test_noise_changes_output(self):
        torch.manual_seed(0)
        aug = SpectrumAugment(noise_std=0.5, random_state=0)
        x = torch.zeros(4, 16)
        y = aug(x)
        assert y.shape == x.shape
        assert not torch.allclose(y, x)
        assert torch.isfinite(y).all()
        assert y.abs().max() < 10.0

    def test_fixed_seed_deterministic(self):
        aug1 = SpectrumAugment(noise_std=0.5, random_state=42)
        aug2 = SpectrumAugment(noise_std=0.5, random_state=42)
        x = torch.zeros(4, 16)
        torch.testing.assert_close(aug1(x), aug2(x))


class TestSpectrumAugmentJitter:
    def test_jitter_scales_all_bins_of_a_sample_equally(self):
        """Intensity jitter multiplies every bin of each sample by the same factor."""
        torch.manual_seed(0)
        aug = SpectrumAugment(intensity_jitter=0.2, random_state=0)
        x = torch.ones(4, 16)
        y = aug(x)
        for i in range(4):
            assert torch.allclose(y[i], y[i, 0].expand(16), atol=1e-6)


class TestSpectrumAugmentDropout:
    def test_dropout_zeros_some_bins(self):
        torch.manual_seed(0)
        aug = SpectrumAugment(peak_dropout_rate=0.5, random_state=0)
        x = torch.ones(16, 64)
        y = aug(x)
        zeros = (y == 0).float().mean().item()
        assert 0.3 < zeros < 0.7


class TestSpectrumAugmentDomainOps:
    """Domain-specific m/z augmentations."""

    def test_domain_ops_all_zero_is_identity(self):
        """mz_shift / mz_warp / blur all default-off => identity."""
        aug = SpectrumAugment()
        x = torch.randn(4, 32)
        torch.testing.assert_close(aug(x), x)

    def test_mz_shift_preserves_shape(self):
        torch.manual_seed(0)
        aug = SpectrumAugment(mz_shift_max_bins=2, random_state=0)
        x = torch.arange(64, dtype=torch.float32).expand(3, 64).clone()
        y = aug(x)
        assert y.shape == x.shape

    def test_mz_shift_is_circular(self):
        """torch.roll is circular: sum across axis is preserved."""
        aug = SpectrumAugment(mz_shift_max_bins=3, random_state=0)
        x = torch.randn(5, 64)
        y = aug(x)
        torch.testing.assert_close(y.sum(dim=1), x.sum(dim=1))

    def test_mz_shift_changes_output_for_nonzero_k(self):
        aug = SpectrumAugment(mz_shift_max_bins=5, random_state=0)
        x = torch.arange(64, dtype=torch.float32).expand(8, 64).clone()
        y = aug(x)
        # At least one sample must be shifted by a non-zero amount.
        assert not torch.allclose(y, x)

    def test_mz_warp_preserves_shape(self):
        aug = SpectrumAugment(mz_warp_max_bins=3, mz_warp_n_knots=8, random_state=0)
        x = torch.randn(4, 64)
        y = aug(x)
        assert y.shape == x.shape
        assert torch.isfinite(y).all()

    def test_mz_warp_deterministic_under_seed(self):
        aug1 = SpectrumAugment(mz_warp_max_bins=3, mz_warp_n_knots=8, random_state=42)
        aug2 = SpectrumAugment(mz_warp_max_bins=3, mz_warp_n_knots=8, random_state=42)
        x = torch.randn(4, 64)
        torch.testing.assert_close(aug1(x), aug2(x))

    def test_blur_preserves_shape(self):
        aug = SpectrumAugment(blur_sigma=1.5)
        x = torch.randn(3, 32)
        y = aug(x)
        assert y.shape == x.shape

    def test_blur_reduces_variance_along_mz_axis(self):
        """A Gaussian blur should reduce per-bin variance in a noisy signal."""
        torch.manual_seed(0)
        aug = SpectrumAugment(blur_sigma=2.0)
        x = torch.randn(4, 256)
        y = aug(x)
        assert y.var(dim=1).mean() < x.var(dim=1).mean()

    def test_blur_preserves_mass_on_uniform_input(self):
        """Convolving a constant signal with a normalized kernel returns the same signal (up to edge effects)."""
        aug = SpectrumAugment(blur_sigma=1.0)
        x = torch.ones(2, 64)
        y = aug(x)
        assert torch.allclose(y[:, 10:-10], x[:, 10:-10], atol=1e-4)


class TestSpectrumAugmentInBaseClassifier:
    def test_default_augment_is_none(self):
        clf = MaldiMLPClassifier()
        assert clf.augment is None

    def test_fit_with_augment_runs(self, synthetic_binary):
        X, y = synthetic_binary
        aug = SpectrumAugment(noise_std=0.05, random_state=0)
        clf = MaldiMLPClassifier(
            hidden_dim=8,
            head_dims=(4,),
            epochs=2,
            batch_size=8,
            augment=aug,
            random_state=0,
        ).fit(X, y)
        assert clf.predict(X).shape == (len(X),)

    def test_augment_not_applied_at_inference(self, synthetic_binary):
        """Two predict_proba calls with augment=SpectrumAugment(noise_std=large)
        should return the SAME output (augmentation is train-only)."""
        X, y = synthetic_binary
        aug = SpectrumAugment(noise_std=10.0)  # would be huge if applied
        clf = MaldiMLPClassifier(
            hidden_dim=8,
            head_dims=(4,),
            epochs=1,
            batch_size=8,
            augment=aug,
            random_state=0,
        ).fit(X, y)
        p1 = clf.predict_proba(X[:4])
        p2 = clf.predict_proba(X[:4])
        np.testing.assert_allclose(p1, p2, atol=1e-5)


class TestToOneHot:
    def test_matches_torch_one_hot(self):
        y = torch.tensor([0, 1, 2, 1])
        out = to_one_hot(y, n_classes=3)
        expected = torch.nn.functional.one_hot(y, 3).to(torch.float32)
        torch.testing.assert_close(out, expected)

    def test_dtype_is_float(self):
        out = to_one_hot(torch.tensor([0, 1]), n_classes=2)
        assert out.dtype == torch.float32


class TestMixup:
    def test_alpha_zero_raises(self):
        with pytest.raises(ValueError, match="mixup alpha"):
            apply_mixup(torch.zeros(2, 4), torch.zeros(2, 2), alpha=0.0)

    def test_preserves_shapes(self):
        x = torch.randn(8, 32)
        y = to_one_hot(torch.randint(0, 3, (8,)), 3)
        x_out, y_out = apply_mixup(x, y, alpha=0.2)
        assert x_out.shape == x.shape
        assert y_out.shape == y.shape

    def test_is_convex_combination(self):
        """For any lam, rows of x_mixed are convex combinations of rows of x."""
        x = torch.arange(16, dtype=torch.float32).expand(4, 16).clone()
        x = x + torch.arange(4).unsqueeze(1).float()
        y = to_one_hot(torch.arange(4) % 2, 2)
        gen = torch.Generator().manual_seed(0)
        x_out, y_out = apply_mixup(x, y, alpha=0.5, generator=gen)

        x_min = x.min(dim=0).values
        x_max = x.max(dim=0).values
        assert (x_out >= x_min - 1e-5).all()
        assert (x_out <= x_max + 1e-5).all()
        torch.testing.assert_close(y_out.sum(dim=1), torch.ones(4))

    def test_seeded_is_deterministic(self):
        x = torch.randn(4, 8)
        y = to_one_hot(torch.tensor([0, 1, 0, 1]), 2)
        gen1 = torch.Generator().manual_seed(7)
        gen2 = torch.Generator().manual_seed(7)
        x1, y1 = apply_mixup(x, y, alpha=0.2, generator=gen1)
        x2, y2 = apply_mixup(x, y, alpha=0.2, generator=gen2)
        torch.testing.assert_close(x1, x2)
        torch.testing.assert_close(y1, y2)


class TestCutmix:
    def test_alpha_zero_raises(self):
        with pytest.raises(ValueError, match="cutmix alpha"):
            apply_cutmix(torch.zeros(2, 4), torch.zeros(2, 2), alpha=0.0)

    def test_preserves_shapes(self):
        x = torch.randn(8, 32)
        y = to_one_hot(torch.randint(0, 3, (8,)), 3)
        x_out, y_out = apply_cutmix(x, y, alpha=1.0)
        assert x_out.shape == x.shape
        assert y_out.shape == y.shape

    def test_splices_contiguous_window(self):
        """Most of x_out matches original x; a contiguous window matches x[perm]."""
        torch.manual_seed(0)
        x = torch.arange(64, dtype=torch.float32).expand(4, 64).clone()
        x = x + torch.arange(4).unsqueeze(1).float() * 1000.0
        y = to_one_hot(torch.arange(4) % 2, 2)
        gen = torch.Generator().manual_seed(0)
        x_out, y_out = apply_cutmix(x, y, alpha=1.0, generator=gen)
        for i in range(4):
            diff = (x_out[i] != x[i]).long()
            transitions = (diff[1:] != diff[:-1]).sum().item()
            assert transitions <= 2
        torch.testing.assert_close(y_out.sum(dim=1), torch.ones(4))

    def test_seeded_is_deterministic(self):
        x = torch.randn(4, 32)
        y = to_one_hot(torch.tensor([0, 1, 0, 1]), 2)
        gen1 = torch.Generator().manual_seed(11)
        gen2 = torch.Generator().manual_seed(11)
        x1, y1 = apply_cutmix(x, y, alpha=1.0, generator=gen1)
        x2, y2 = apply_cutmix(x, y, alpha=1.0, generator=gen2)
        torch.testing.assert_close(x1, x2)
        torch.testing.assert_close(y1, y2)
