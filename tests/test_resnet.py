"""Tests specific to MaldiResNetClassifier."""

from __future__ import annotations

import pytest
import torch

from maldideepkit import MaldiResNetClassifier
from maldideepkit.resnet.resnet import BasicBlock1D, SpectralResNet1D


class TestMaldiResNetClassifier:
    def test_defaults(self):
        clf = MaldiResNetClassifier()
        assert clf.stage_channels == (64, 128, 256, 512)
        assert clf.blocks_per_stage == (2, 2, 2, 2)
        assert clf.stem_kernel_size == 7
        assert clf.stem_stride == 1
        assert clf.block_kernel_size == 7
        assert clf.use_stem_pool is False
        assert clf.weight_decay == 1e-4
        assert clf.grad_clip_norm == 1.0
        assert clf.warmup_epochs == 5
        assert clf.input_transform == "log1p"

    def test_literal_resnet18_variant_still_available(self, synthetic_binary):
        X, y = synthetic_binary
        clf = MaldiResNetClassifier(
            stem_channels=8,
            stage_channels=(8, 16),
            blocks_per_stage=(1, 1),
            stem_stride=2,
            block_kernel_size=3,
            use_stem_pool=True,
            epochs=1,
            batch_size=8,
            random_state=0,
        ).fit(X, y)
        assert clf.predict(X).shape == (len(X),)
        assert clf.use_stem_pool is True

    def test_stem_pool_toggles_module_structure(self):
        model_no_pool = SpectralResNet1D(
            input_dim=128,
            stem_channels=8,
            stage_channels=(8,),
            blocks_per_stage=(1,),
            use_stem_pool=False,
        )
        model_with_pool = SpectralResNet1D(
            input_dim=128,
            stem_channels=8,
            stage_channels=(8,),
            blocks_per_stage=(1,),
            use_stem_pool=True,
        )
        no_pool_has_maxpool = any(
            isinstance(m, torch.nn.MaxPool1d) for m in model_no_pool.stem
        )
        with_pool_has_maxpool = any(
            isinstance(m, torch.nn.MaxPool1d) for m in model_with_pool.stem
        )
        assert no_pool_has_maxpool is False
        assert with_pool_has_maxpool is True

    def test_stage_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            SpectralResNet1D(
                input_dim=256,
                stage_channels=(8, 16),
                blocks_per_stage=(1, 1, 1),
            )

    def test_basic_block_shortcut(self):
        block = BasicBlock1D(8, 16, stride=2)
        x = torch.randn(2, 8, 32)
        out = block(x)
        assert out.shape == (2, 16, 16)

    def test_basic_block_identity_shortcut(self):
        block = BasicBlock1D(8, 8, stride=1)
        x = torch.randn(2, 8, 32)
        out = block(x)
        assert out.shape == (2, 8, 32)

    def test_fit_predict(self, synthetic_binary):
        X, y = synthetic_binary
        clf = MaldiResNetClassifier(
            stem_channels=8,
            stage_channels=(8, 16),
            blocks_per_stage=(1, 1),
            epochs=2,
            batch_size=8,
            random_state=0,
        ).fit(X, y)
        assert clf.predict(X).shape == (len(X),)

    def test_forward_shape(self):
        model = SpectralResNet1D(
            input_dim=256,
            n_classes=4,
            stem_channels=8,
            stage_channels=(8, 16),
            blocks_per_stage=(1, 1),
        )
        out = model(torch.randn(2, 256))
        assert out.shape == (2, 4)
