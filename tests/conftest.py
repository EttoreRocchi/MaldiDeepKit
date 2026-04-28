"""Shared fixtures and model configurations for MaldiDeepKit tests.

Every classifier gets a ``_tiny`` configuration (small enough to train
in seconds on CPU) used throughout the test suite.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from maldideepkit import (
    MaldiCNNClassifier,
    MaldiMLPClassifier,
    MaldiResNetClassifier,
    MaldiTransformerClassifier,
)


@pytest.fixture(autouse=True)
def _single_thread_torch():
    torch.set_num_threads(2)
    yield


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture
def synthetic_binary(rng):
    n, d = 48, 128
    X = rng.standard_normal((n, d)).astype(np.float32)
    y = rng.integers(0, 2, size=n)
    return X, y


@pytest.fixture
def synthetic_multiclass(rng):
    n, d = 60, 128
    X = rng.standard_normal((n, d)).astype(np.float32)
    y = rng.integers(0, 3, size=n)
    return X, y


def _mlp_tiny(**kwargs):
    defaults = dict(
        hidden_dim=32,
        head_dims=(16,),
        epochs=2,
        batch_size=8,
        early_stopping_patience=3,
        random_state=0,
    )
    defaults.update(kwargs)
    return MaldiMLPClassifier(**defaults)


def _cnn_tiny(**kwargs):
    defaults = dict(
        channels=(8, 16),
        kernel_size=5,
        head_dim=16,
        epochs=2,
        batch_size=8,
        early_stopping_patience=3,
        random_state=0,
    )
    defaults.update(kwargs)
    return MaldiCNNClassifier(**defaults)


def _resnet_tiny(**kwargs):
    defaults = dict(
        stem_channels=8,
        stage_channels=(8, 16),
        blocks_per_stage=(1, 1),
        epochs=2,
        batch_size=8,
        early_stopping_patience=3,
        random_state=0,
    )
    defaults.update(kwargs)
    return MaldiResNetClassifier(**defaults)


def _transformer_tiny(**kwargs):
    defaults = dict(
        embed_dim=16,
        depth=2,
        num_heads=2,
        patch_size=2,
        head_dim=16,
        epochs=2,
        batch_size=8,
        early_stopping_patience=3,
        random_state=0,
    )
    defaults.update(kwargs)
    return MaldiTransformerClassifier(**defaults)


ALL_FACTORIES = {
    "MaldiMLPClassifier": _mlp_tiny,
    "MaldiCNNClassifier": _cnn_tiny,
    "MaldiResNetClassifier": _resnet_tiny,
    "MaldiTransformerClassifier": _transformer_tiny,
}


@pytest.fixture(params=sorted(ALL_FACTORIES.keys()))
def any_tiny_classifier(request):
    """Parametrised fixture producing one tiny classifier per family."""
    return ALL_FACTORIES[request.param]()
