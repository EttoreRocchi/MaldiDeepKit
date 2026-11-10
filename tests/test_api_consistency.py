"""Every architecture exposes the same API.

A parameter, method or feature added to one classifier and missing from
another fails here.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch
from torch import nn

from maldideepkit import (
    BaseSpectralClassifier,
    Callback,
    MaldiCNNClassifier,
    MaldiDualBranchClassifier,
    MaldiMLPClassifier,
    MaldiResNetClassifier,
    MaldiSelfTrainingClassifier,
    MaldiSpectrumEncoder,
    MaldiTransformerClassifier,
    get_backbone_class,
)
from maldideepkit.augment import SpectrumAugment
from maldideepkit.utils import SpectralEnsemble

BINS = 128
SIDE = 8
FAST = {"epochs": 1, "batch_size": 16, "device": "cpu", "random_state": 0}
SINGLE = (
    MaldiMLPClassifier,
    MaldiCNNClassifier,
    MaldiResNetClassifier,
    MaldiTransformerClassifier,
)
# Defaults tuned to the architecture.
TUNED_DEFAULTS = {
    MaldiResNetClassifier: {
        "weight_decay",
        "grad_clip_norm",
        "warmup_epochs",
        "input_transform",
    },
    MaldiTransformerClassifier: {
        "learning_rate",
        "weight_decay",
        "grad_clip_norm",
        "warmup_epochs",
    },
}
# Shared with the classifiers, under the same name and meaning.
ENCODER_SHARED = (
    "learning_rate",
    "weight_decay",
    "batch_size",
    "epochs",
    "warmup_epochs",
    "grad_clip_norm",
    "early_stopping_patience",
    "val_fraction",
    "use_amp",
    "amp_dtype",
    "device",
    "random_state",
    "verbose",
    "torch_compile",
    "optimizer",
    "scheduler",
    "callbacks",
    "augment",
)


class KwargsClassifier(BaseSpectralClassifier):
    """The shortest custom classifier."""

    def __init__(self, hidden_dim=8, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim

    def _build_model(self):
        return nn.Sequential(
            nn.Linear(self.input_dim_, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 2),
        )


class Counter(Callback):
    def __init__(self):
        self.steps = 0

    def on_step_end(self, model, step):
        self.steps += 1


def per_row_loss(logits, target):
    return torch.nn.functional.cross_entropy(logits, target, reduction="none")


def _params(cls) -> dict[str, inspect.Parameter]:
    return dict(list(inspect.signature(cls.__init__).parameters.items())[1:])


def _data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    y = np.tile([0, 1], 16)
    X = rng.random((32, BINS), dtype=np.float32)
    X[y == 1, :8] += 0.5
    return X, np.hstack([X, rng.random((32, SIDE), dtype=np.float32)]), y


ESTIMATORS = {
    "mlp": lambda **k: MaldiMLPClassifier(**{**FAST, **k}),
    "cnn": lambda **k: MaldiCNNClassifier(**{**FAST, **k}),
    "resnet": lambda **k: MaldiResNetClassifier(**{**FAST, **k}),
    "transformer": lambda **k: MaldiTransformerClassifier(
        **{**FAST, "patch_size": 16, "embed_dim": 16, "depth": 1, "num_heads": 2, **k}
    ),
    "custom": lambda **k: KwargsClassifier(**{**FAST, **k}),
    "dual": lambda **k: MaldiDualBranchClassifier(BINS, **{**FAST, **k}),
    "dual_cnn": lambda **k: MaldiDualBranchClassifier(
        BINS, backbone="cnn", **{**FAST, **k}
    ),
}
FEATURES = {
    "focal": {"loss": "focal"},
    "loss_object": {"loss": per_row_loss},
    "class_weight": {"class_weight": "balanced"},
    "label_smoothing": {"label_smoothing": 0.1},
    "mixup": {"mixup_alpha": 0.2},
    "cutmix": {"cutmix_alpha": 0.2},
    "ema": {"ema_decay": 0.9},
    "swa": {"swa_start_epoch": 0},
    "sam": {"use_sam": True},
    "threshold": {"tune_threshold": True, "min_val_auroc_for_threshold_tune": 0.0},
    "calibration": {"calibrate_temperature": True},
    "augment": {"augment": SpectrumAugment()},
    "retry": {"retry_on_val_auroc_below": 1.1, "max_retries": 1},
    "warmup": {"warmup_epochs": 1},
    "grad_clip": {"grad_clip_norm": 1.0},
    "track_metrics": {"track_train_metrics": True},
    "input_transform": {"input_transform": "log1p"},
    "optimizer": {"optimizer": "sgd"},
    "scheduler": {"scheduler": "constant"},
    "freeze": {"freeze_trunk_epochs": 1},
}


@pytest.mark.parametrize(
    "cls", [*SINGLE, MaldiDualBranchClassifier], ids=lambda c: c.__name__
)
def test_every_classifier_has_every_base_parameter(cls):
    base, own = _params(BaseSpectralClassifier), _params(cls)
    assert not set(base) - set(own)
    tuned = TUNED_DEFAULTS.get(cls, set())
    for name, parameter in base.items():
        assert own[name].kind is inspect.Parameter.KEYWORD_ONLY or name in (
            "input_dim",
            "n_classes",
        )
        if name not in tuned:
            assert repr(own[name].default) == repr(parameter.default), name


@pytest.mark.parametrize(
    ("cls", "positional"),
    [
        (BaseSpectralClassifier, ["input_dim", "n_classes"]),
        *[(cls, ["input_dim", "n_classes"]) for cls in SINGLE],
        (MaldiDualBranchClassifier, ["n_spec_bins"]),
        (MaldiSelfTrainingClassifier, ["base_estimator"]),
        (MaldiSpectrumEncoder, ["backbone"]),
        (SpectralEnsemble, ["classifiers"]),
    ],
    ids=lambda v: getattr(v, "__name__", None),
)
def test_only_the_identifying_parameters_are_positional(cls, positional):
    params = _params(cls)
    assert [
        n for n, p in params.items() if p.kind is not inspect.Parameter.KEYWORD_ONLY
    ] == positional


def test_encoder_shares_the_training_parameters():
    encoder, base = _params(MaldiSpectrumEncoder), _params(BaseSpectralClassifier)
    for name in ENCODER_SHARED:
        assert name in encoder and name in base, name


@pytest.mark.parametrize(
    "cls",
    [*SINGLE, KwargsClassifier, MaldiDualBranchClassifier, MaldiSelfTrainingClassifier],
    ids=lambda c: c.__name__,
)
def test_every_classifier_has_the_public_methods(cls):
    for name in (
        "fit",
        "predict",
        "predict_proba",
        "score",
        "save",
        "load",
        "get_embeddings",
        "get_params",
        "set_params",
    ):
        assert callable(getattr(cls, name)), name


@pytest.mark.parametrize(
    "cls",
    [*SINGLE, KwargsClassifier, MaldiDualBranchClassifier],
    ids=lambda c: c.__name__,
)
def test_every_architecture_has_from_spectrum(cls):
    kwargs = {"hidden_dim": 4} if cls is KwargsClassifier else {}
    clf = cls.from_spectrum(3, BINS, **kwargs)
    width = clf.n_spec_bins if cls is MaldiDualBranchClassifier else clf.input_dim
    assert width == BINS


def test_dual_from_spectrum_scales_a_named_backbone():
    scaled = MaldiDualBranchClassifier.from_spectrum(6, BINS, backbone="cnn")
    assert isinstance(scaled.backbone, MaldiCNNClassifier)
    assert (
        scaled.backbone.kernel_size
        == MaldiCNNClassifier.from_spectrum(6, BINS).kernel_size
    )


@pytest.mark.parametrize(
    "cls",
    [
        *SINGLE,
        MaldiDualBranchClassifier,
        MaldiSelfTrainingClassifier,
        SpectralEnsemble,
    ],
    ids=lambda c: c.__name__,
)
def test_fit_takes_sample_weight_keyword(cls):
    parameter = inspect.signature(cls.fit).parameters["sample_weight"]
    assert (
        parameter.kind is inspect.Parameter.KEYWORD_ONLY and parameter.default is None
    )


@pytest.mark.parametrize(
    "cls",
    [
        BaseSpectralClassifier,
        *SINGLE,
        MaldiDualBranchClassifier,
        MaldiSelfTrainingClassifier,
        MaldiSpectrumEncoder,
        SpectralEnsemble,
    ],
    ids=lambda c: c.__name__,
)
def test_load_has_the_same_form(cls):
    parameters = inspect.signature(cls.load).parameters
    assert list(parameters) == ["path", "trusted"]
    assert parameters["trusted"].kind is inspect.Parameter.KEYWORD_ONLY


@pytest.mark.parametrize("name", ["mlp", "cnn", "resnet", "transformer"])
def test_builtin_names_match_the_classes(name):
    assert issubclass(get_backbone_class(name), BaseSpectralClassifier)


@pytest.mark.parametrize("feature", list(FEATURES))
@pytest.mark.parametrize("arch", list(ESTIMATORS))
def test_every_feature_on_every_architecture(arch, feature):
    X, X_dual, y = _data()
    data = X_dual if arch.startswith("dual") else X
    clf = ESTIMATORS[arch](**FEATURES[feature]).fit(data, y)
    assert clf.predict_proba(data).shape == (32, 2)
    assert clf.predict(data).shape == (32,)


@pytest.mark.parametrize("arch", list(ESTIMATORS))
class TestEveryArchitecture:
    @staticmethod
    def _data(arch):
        X, X_dual, y = _data()
        return (X_dual if arch.startswith("dual") else X), y

    def test_sample_weight_and_warm_start(self, arch):
        data, y = self._data(arch)
        clf = ESTIMATORS[arch]().fit(data, y, sample_weight=np.ones(32))
        assert clf.fit(data, y, warm_start=True).predict(data).shape == (32,)

    def test_callbacks_run(self, arch):
        data, y = self._data(arch)
        counter = Counter()
        ESTIMATORS[arch](callbacks=[counter]).fit(data, y)
        assert counter.steps > 0

    def test_save_load(self, arch, tmp_path):
        data, y = self._data(arch)
        clf = ESTIMATORS[arch](optimizer="sgd", callbacks=[Counter()]).fit(data, y)
        clf.save(tmp_path / "m")
        loaded = type(clf).load(tmp_path / "m")
        assert loaded.optimizer == "sgd"
        assert isinstance(loaded.callbacks[0], Counter)
        np.testing.assert_allclose(
            loaded.predict_proba(data), clf.predict_proba(data), atol=1e-5
        )

    def test_embeddings_feed_the_final_layer(self, arch):
        data, y = self._data(arch)
        clf = ESTIMATORS[arch]().fit(data, y)
        embeddings = clf.get_embeddings(data)
        assert embeddings.ndim == 2 and embeddings.shape[0] == 32
        assert embeddings.dtype == np.float32

    def test_init_from(self, arch):
        data, y = self._data(arch)
        if arch.startswith("dual"):
            template = MaldiCNNClassifier()
        else:
            template = ESTIMATORS[arch]()
        encoder = MaldiSpectrumEncoder(backbone=template, **FAST).fit(data[:, :BINS])
        clf = ESTIMATORS[arch](init_from=encoder, freeze_trunk_epochs=1)
        assert clf.fit(data, y).predict_proba(data).shape == (32, 2)
        assert clf.trunk_unfroze_ is False


@pytest.mark.parametrize("name", ["mlp", "cnn", "resnet", "transformer"])
def test_embeddings_are_what_the_final_layer_reads(name):
    X, _, y = _data()
    clf = ESTIMATORS[name]().fit(X, y)
    final = clf.model_.final_layer
    embeddings = torch.from_numpy(clf.get_embeddings(X)).to(final.weight.device)
    with torch.no_grad():
        logits = final(embeddings).cpu().numpy()
    np.testing.assert_allclose(logits, clf._forward_logits(X), atol=1e-4)
