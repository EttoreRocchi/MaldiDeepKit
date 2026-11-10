"""Tests for the ``forward_features`` / ``final_layer`` trunk boundary.

The ``_*_forward_v030`` functions below are the ``forward`` bodies of
the v0.3.0 backbones, copied verbatim (``self`` is the module under
test). They run on the module's own submodules, so a renamed or
restructured attribute makes them fail, and the refactored ``forward``
must match them bit for bit in both eval and train mode.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from maldideepkit import MaldiDualBranchClassifier
from maldideepkit.attention.mlp import SpectralAttentionMLP
from maldideepkit.base._trunk import final_layer_prefix
from maldideepkit.cnn.cnn import SpectralCNN1D
from maldideepkit.resnet.resnet import SpectralResNet1D
from maldideepkit.transformer.transformer import SpectralTransformer1D

INPUT_DIM = 64
N_CLASSES = 3
BATCH_SIZE = 8


def _mlp_forward_v030(self, x: torch.Tensor) -> torch.Tensor:
    """``SpectralAttentionMLP.forward`` of v0.3.0."""
    projected = self.proj(x)
    if self.use_attention:
        weights = self.attn(projected)
        self.last_attention = weights.detach()
        gated = projected * weights
    else:
        self.last_attention = None
        gated = projected
    return self.head(gated)


def _cnn_forward_v030(self, x: torch.Tensor) -> torch.Tensor:
    """``SpectralCNN1D.forward`` of v0.3.0."""
    x = x.unsqueeze(1)
    feat = self.backbone(x)
    return self.head(feat.flatten(1))


def _resnet_forward_v030(self, x: torch.Tensor) -> torch.Tensor:
    """``SpectralResNet1D.forward`` of v0.3.0."""
    x = x.unsqueeze(1)
    x = self.stem(x)
    x = self.stages(x)
    x = self.gap(x).squeeze(-1)
    x = self.dropout(x)
    return self.fc(x)


def _transformer_forward_v030(self, x: torch.Tensor) -> torch.Tensor:
    """``SpectralTransformer1D.forward`` of v0.3.0."""
    x = x.unsqueeze(1)
    pad = (-x.shape[-1]) % self.patch_size
    if pad:
        x = torch.nn.functional.pad(x, (0, pad))
    tokens = self.embed(x)
    if self.cls_token is not None:
        cls = self.cls_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
    tokens = self.pos_drop(tokens + self.pos_embed[:, : tokens.shape[1]])
    for block in self.blocks:
        tokens = block(tokens)
    tokens = self.norm(tokens)
    if self.pool == "cls":
        pooled = tokens[:, 0]
    else:
        start = 1 if self.cls_token is not None else 0
        pooled = tokens[:, start:].mean(dim=1)
    return self.head(pooled)


V030_FORWARD = {
    SpectralAttentionMLP: _mlp_forward_v030,
    SpectralCNN1D: _cnn_forward_v030,
    SpectralResNet1D: _resnet_forward_v030,
    SpectralTransformer1D: _transformer_forward_v030,
}

CONFIGS: dict[str, tuple[type[nn.Module], dict]] = {
    "mlp_default": (SpectralAttentionMLP, {}),
    "mlp_no_head": (SpectralAttentionMLP, {"head_dims": ()}),
    "mlp_no_attention": (SpectralAttentionMLP, {"use_attention": False}),
    "cnn_default": (SpectralCNN1D, {}),
    "resnet_small": (
        SpectralResNet1D,
        {"stem_channels": 8, "stage_channels": (8, 16), "blocks_per_stage": (1, 1)},
    ),
    "transformer_mean": (SpectralTransformer1D, {"pool": "mean"}),
    "transformer_cls": (SpectralTransformer1D, {"pool": "cls"}),
}
CONFIG_NAMES = list(CONFIGS)

EXPECTED_PREFIX = {
    "mlp_default": "head.8",
    "mlp_no_head": "head.0",
    "mlp_no_attention": "head.8",
    "cnn_default": "head.4",
    "resnet_small": "fc",
    "transformer_mean": "head.3",
    "transformer_cls": "head.3",
}


def _build(name: str) -> tuple[nn.Module, torch.Tensor]:
    """Seed-0 module of config ``name`` in eval mode, plus a seed-0 input batch."""
    cls, extra = CONFIGS[name]
    torch.manual_seed(0)
    model = cls(input_dim=INPUT_DIM, n_classes=N_CLASSES, **extra)
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(BATCH_SIZE, INPUT_DIM, generator=gen)
    return model.eval(), x


def _legacy_last_linear(module: nn.Module) -> nn.Linear | None:
    """Last ``nn.Linear`` in registration order (the v0.3.0 dual-branch lookup)."""
    last = None
    for sub in module.modules():
        if isinstance(sub, nn.Linear):
            last = sub
    return last


def test_every_config_has_expected_prefix():
    assert set(CONFIG_NAMES) == set(EXPECTED_PREFIX)


@pytest.mark.parametrize("train", [False, True])
@pytest.mark.parametrize("name", CONFIG_NAMES)
def test_forward_matches_v030_reference(name, train):
    model, x = _build(name)
    model.train(train)
    reference_forward = V030_FORWARD[type(model)]
    with torch.no_grad():
        torch.manual_seed(1)
        expected = reference_forward(model, x)
        torch.manual_seed(1)
        got = model(x)
    assert expected.shape == (BATCH_SIZE, N_CLASSES)
    assert torch.equal(got, expected)


@pytest.mark.parametrize("train", [False, True])
@pytest.mark.parametrize("name", CONFIG_NAMES)
def test_final_layer_of_features_equals_forward(name, train):
    model, x = _build(name)
    model.train(train)
    with torch.no_grad():
        torch.manual_seed(1)
        expected = model(x)
        torch.manual_seed(1)
        got = model.final_layer(model.forward_features(x))
    assert torch.equal(got, expected)


@pytest.mark.parametrize("name", CONFIG_NAMES)
def test_feature_dim_matches_forward_features(name):
    model, x = _build(name)
    with torch.no_grad():
        feats = model.forward_features(x)
    assert feats.shape == (x.shape[0], model.feature_dim)
    assert model.final_layer.in_features == model.feature_dim


@pytest.mark.parametrize("name", CONFIG_NAMES)
def test_final_layer_is_legacy_last_linear(name):
    model, _ = _build(name)
    assert isinstance(model.final_layer, nn.Linear)
    assert model.final_layer is _legacy_last_linear(model)


@pytest.mark.parametrize("name", CONFIG_NAMES)
def test_final_layer_prefix(name):
    model, _ = _build(name)
    prefix = final_layer_prefix(model)
    assert prefix == EXPECTED_PREFIX[name]
    assert model.get_submodule(prefix) is model.final_layer
    final_keys = [k for k in model.state_dict() if k.startswith(prefix + ".")]
    assert final_keys == [f"{prefix}.weight", f"{prefix}.bias"]


def test_final_layer_prefix_requires_final_layer():
    with pytest.raises(AttributeError):
        final_layer_prefix(nn.Sequential(nn.Linear(4, 2)))


@pytest.mark.parametrize("name", ["mlp_default", "mlp_no_head", "mlp_no_attention"])
def test_mlp_forward_features_records_attention(name):
    model, x = _build(name)
    with torch.no_grad():
        model.forward_features(x)
    if CONFIGS[name][1].get("use_attention", True):
        assert model.last_attention is not None
        assert model.last_attention.shape == (x.shape[0], model.hidden_dim)
    else:
        assert model.last_attention is None


@pytest.mark.parametrize("name", CONFIG_NAMES)
def test_dual_branch_substitution_uses_final_layer(name):
    model, x = _build(name)
    with torch.no_grad():
        feats = model.forward_features(x)
    dim = MaldiDualBranchClassifier._find_last_linear_in_features(model)
    assert dim == model.feature_dim
    final = model.final_layer
    spec = MaldiDualBranchClassifier._replace_last_linear_with_identity(model)
    assert isinstance(spec.final_layer, nn.Identity)
    assert all(sub is not final for sub in spec.modules())
    with torch.no_grad():
        assert torch.equal(spec(x), feats)


def test_dual_branch_legacy_lookup_for_custom_modules():
    module = nn.Sequential(nn.Linear(6, 5), nn.ReLU(), nn.Linear(5, 2))
    assert MaldiDualBranchClassifier._find_last_linear_in_features(module) == 5
    MaldiDualBranchClassifier._replace_last_linear_with_identity(module)
    assert isinstance(module[0], nn.Linear)
    assert isinstance(module[2], nn.Identity)


def test_dual_branch_feature_dim_without_final_layer_uses_legacy_lookup():
    module = nn.Sequential(nn.Linear(6, 5), nn.ReLU(), nn.Linear(5, 2))
    module.feature_dim = 99  # unrelated attribute on a custom module
    assert MaldiDualBranchClassifier._find_last_linear_in_features(module) == 5
