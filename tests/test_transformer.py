"""Tests specific to MaldiTransformerClassifier."""

from __future__ import annotations

import pytest
import torch

from maldideepkit import MaldiTransformerClassifier
from maldideepkit.transformer.transformer import (
    MultiHeadSelfAttention,
    SpectralTransformer1D,
    TransformerBlock,
)


class TestMaldiTransformerClassifier:
    def test_defaults(self):
        clf = MaldiTransformerClassifier()
        assert clf.patch_size == 4
        assert clf.embed_dim == 64
        assert clf.depth == 6
        assert clf.num_heads == 4
        assert clf.mlp_ratio == 4
        assert clf.pool == "mean"
        assert clf.layerscale_init == 1e-4
        assert clf.learning_rate == 3e-4
        assert clf.weight_decay == 0.05
        assert clf.grad_clip_norm == 1.0
        assert clf.warmup_epochs == 5
        assert clf.drop_path_rate == 0.1
        assert clf.attention_dropout == 0.0

    def test_fit_predict(self, synthetic_binary):
        X, y = synthetic_binary
        clf = MaldiTransformerClassifier(
            embed_dim=16,
            depth=2,
            num_heads=2,
            patch_size=2,
            head_dim=16,
            epochs=2,
            batch_size=8,
            random_state=0,
        ).fit(X, y)
        assert clf.predict(X).shape == (len(X),)
        assert clf.predict_proba(X).shape == (len(X), 2)

    def test_multiclass(self, synthetic_multiclass):
        X, y = synthetic_multiclass
        clf = MaldiTransformerClassifier(
            embed_dim=16,
            depth=2,
            num_heads=2,
            patch_size=2,
            head_dim=16,
            epochs=2,
            batch_size=8,
            random_state=0,
        ).fit(X, y)
        assert clf.predict_proba(X).shape == (len(X), 3)

    def test_forward_shape(self):
        model = SpectralTransformer1D(
            input_dim=128,
            n_classes=2,
            embed_dim=16,
            depth=2,
            num_heads=2,
            patch_size=2,
        )
        out = model(torch.randn(2, 128))
        assert out.shape == (2, 2)

    def test_handles_non_divisible_length(self):
        # 30 is not cleanly divisible by patch_size
        model = SpectralTransformer1D(
            input_dim=30,
            n_classes=2,
            embed_dim=16,
            depth=2,
            num_heads=2,
            patch_size=4,
        )
        out = model(torch.randn(2, 30))
        assert out.shape == (2, 2)

    def test_bad_head_divisibility(self):
        with pytest.raises(ValueError, match="divisible"):
            MultiHeadSelfAttention(dim=17, num_heads=4)

    def test_bad_pool_raises(self):
        with pytest.raises(ValueError, match="pool"):
            SpectralTransformer1D(
                input_dim=64, embed_dim=16, depth=1, num_heads=2, pool="bogus"
            )

    def test_bad_depth_raises(self):
        with pytest.raises(ValueError, match="depth"):
            SpectralTransformer1D(input_dim=64, embed_dim=16, depth=0, num_heads=2)

    def test_embed_dim_heads_mismatch(self):
        with pytest.raises(ValueError, match="divisible"):
            SpectralTransformer1D(input_dim=64, embed_dim=17, depth=2, num_heads=4)


class TestTransformerPooling:
    def test_cls_pool_creates_cls_token(self):
        model = SpectralTransformer1D(
            input_dim=64,
            embed_dim=16,
            depth=2,
            num_heads=2,
            patch_size=2,
            pool="cls",
        )
        assert model.cls_token is not None
        assert model.cls_token.shape == (1, 1, 16)

    def test_mean_pool_has_no_cls_token(self):
        model = SpectralTransformer1D(
            input_dim=64,
            embed_dim=16,
            depth=2,
            num_heads=2,
            patch_size=2,
            pool="mean",
        )
        assert model.cls_token is None

    def test_both_pools_produce_same_shape(self):
        x = torch.randn(2, 64)
        out_mean = SpectralTransformer1D(
            input_dim=64, embed_dim=16, depth=2, num_heads=2, patch_size=2, pool="mean"
        )(x)
        out_cls = SpectralTransformer1D(
            input_dim=64, embed_dim=16, depth=2, num_heads=2, patch_size=2, pool="cls"
        )(x)
        assert out_mean.shape == out_cls.shape == (2, 2)


class TestLayerScale:
    def test_enabled_by_default(self):
        block = TransformerBlock(dim=16, num_heads=2)
        assert block.use_layerscale is True
        assert hasattr(block, "gamma1")
        assert hasattr(block, "gamma2")

    def test_disabled_when_init_is_none(self):
        block = TransformerBlock(dim=16, num_heads=2, layerscale_init=None)
        assert block.use_layerscale is False
        assert not hasattr(block, "gamma1")

    def test_initial_gammas_match_init(self):
        block = TransformerBlock(dim=8, num_heads=2, layerscale_init=1e-3)
        assert torch.allclose(block.gamma1, torch.full((8,), 1e-3))
        assert torch.allclose(block.gamma2, torch.full((8,), 1e-3))

    def test_layerscale_near_zero_makes_block_near_identity(self):
        """With LayerScale initialised near zero, each block should be ~identity at init."""
        torch.manual_seed(0)
        block = TransformerBlock(dim=16, num_heads=2, layerscale_init=1e-6, dropout=0.0)
        block.eval()
        x = torch.randn(2, 8, 16)
        y = block(x)
        # With γ ≈ 0, residual contributions are negligible and y ≈ x.
        torch.testing.assert_close(y, x, rtol=1e-3, atol=1e-3)


class TestDropPath:
    def test_drop_path_changes_train_output(self):
        """drop_path > 0 should randomize forward outputs during training."""
        torch.manual_seed(0)
        model = SpectralTransformer1D(
            input_dim=32,
            embed_dim=8,
            depth=4,
            num_heads=2,
            patch_size=2,
            drop_path_rate=0.5,
            layerscale_init=None,  # isolate drop_path from LayerScale
        )
        model.train()
        x = torch.randn(4, 32)
        out1 = model(x)
        out2 = model(x)
        assert not torch.allclose(out1, out2)

    def test_drop_path_is_noop_at_eval(self):
        torch.manual_seed(0)
        model = SpectralTransformer1D(
            input_dim=32,
            embed_dim=8,
            depth=4,
            num_heads=2,
            patch_size=2,
            drop_path_rate=0.5,
        )
        model.eval()
        x = torch.randn(4, 32)
        torch.testing.assert_close(model(x), model(x))


class TestMultiHeadSelfAttention:
    def test_preserves_shape(self):
        attn = MultiHeadSelfAttention(dim=16, num_heads=4)
        x = torch.randn(3, 7, 16)
        out = attn(x)
        assert out.shape == x.shape

    def test_is_permutation_equivariant(self):
        """Same tokens in a permuted order produce permuted outputs.

        Holds despite QK-norm because the per-head LayerNorm
        normalises over the feature axis (``head_dim``), not over
        tokens.
        """
        torch.manual_seed(0)
        attn = MultiHeadSelfAttention(dim=16, num_heads=4)
        attn.eval()
        x = torch.randn(1, 5, 16)
        perm = torch.tensor([2, 0, 4, 1, 3])
        out_a = attn(x)[:, perm]
        out_b = attn(x[:, perm])
        torch.testing.assert_close(out_a, out_b, rtol=1e-5, atol=1e-5)

    def test_qk_norm_is_on_by_default(self):
        """QK-norm (``q_norm`` / ``k_norm``) is always on - universal stability knob."""
        attn = MultiHeadSelfAttention(dim=16, num_heads=4)
        assert isinstance(attn.q_norm, torch.nn.LayerNorm)
        assert isinstance(attn.k_norm, torch.nn.LayerNorm)
        # Normalized shape is per-head feature dim.
        assert attn.q_norm.normalized_shape == (4,)
        assert attn.k_norm.normalized_shape == (4,)

    def test_qk_norm_bounds_attention_under_scale(self):
        """Scaling the input by a large factor must not blow up the softmax.

        Without QK-norm, ``(q @ k.T) / sqrt(d)`` grows linearly with
        input norm; softmax saturates and gradients vanish. With QK-
        norm, q and k are unit-variance regardless of input scale, so
        the attention output stays bounded.
        """
        torch.manual_seed(0)
        attn = MultiHeadSelfAttention(dim=16, num_heads=4)
        attn.eval()
        x = torch.randn(2, 8, 16)
        out_small = attn(x)
        out_large = attn(x * 1000.0)
        assert torch.isfinite(out_large).all()
        assert out_large.abs().max() < 1e6  # not blown up
        assert out_small.abs().max() < 1e3
