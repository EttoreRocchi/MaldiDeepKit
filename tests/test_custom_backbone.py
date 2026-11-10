"""Custom backbones without ``forward_features`` / ``feature_dim`` / ``final_layer``.

A user-defined classifier whose module is a plain ``nn.Sequential``
follows the dual-branch convention: the trunk is everything except the
``nn.Linear`` producing the output (see ``maldideepkit.base._trunk``).
"""

from __future__ import annotations

import inspect
import json
from collections import OrderedDict

import numpy as np
import pytest
import torch
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV
from torch import nn

from maldideepkit import (
    BaseSpectralClassifier,
    MaldiCNNClassifier,
    MaldiDualBranchClassifier,
    MaldiMLPClassifier,
    MaldiResNetClassifier,
)
from maldideepkit.attention.mlp import SpectralAttentionMLP
from maldideepkit.base._trunk import (
    final_layer_prefix,
    split_trunk,
    trunk_prefix,
    trunk_state,
)
from maldideepkit.pretraining import MaldiSpectrumEncoder, SSLModel

INPUT_DIM = 64
HIDDEN = 12


class SequentialClassifier(BaseSpectralClassifier):
    """Classifier whose module is a plain ``nn.Sequential`` (no trunk interface)."""

    def __init__(
        self,
        hidden_dim: int = HIDDEN,
        nested: bool = False,
        *,
        input_dim=None,
        n_classes=2,
        epochs=0,
        batch_size=16,
        val_fraction=0.25,
        input_transform=None,
        device="cpu",
        random_state=0,
        init_from=None,
        freeze_trunk_epochs=0,
    ):
        super().__init__(
            input_dim=input_dim,
            n_classes=n_classes,
            epochs=epochs,
            batch_size=batch_size,
            val_fraction=val_fraction,
            input_transform=input_transform,
            device=device,
            random_state=random_state,
            init_from=init_from,
            freeze_trunk_epochs=freeze_trunk_epochs,
        )
        self.hidden_dim = hidden_dim
        self.nested = nested

    def _build_model(self) -> nn.Module:
        body = [
            nn.Linear(self.input_dim_, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
        ]
        head = [nn.Dropout(0.2), nn.Linear(self.hidden_dim, self.n_classes_)]
        if self.nested:
            return nn.Sequential(
                OrderedDict(body=nn.Sequential(*body), head=nn.Sequential(*head))
            )
        return nn.Sequential(*body, *head)


FINAL_PREFIX = {False: "4", True: "head.1"}


class _TrainingAware(nn.Module):
    """Root module whose own forward depends on ``self.training``."""

    def __init__(self) -> None:
        super().__init__()
        self.inner = nn.Linear(6, 5)
        self.out = nn.Linear(5, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.inner(x)
        if self.training:
            h = 2.0 * h
        return self.out(h)


def _data(n: int = 48) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    X = np.abs(rng.standard_normal((n, INPUT_DIM))).astype(np.float32)
    X[:, [5, 20, 40]] += 5.0
    return X, np.arange(n) % 2


def _fit_encoder(template, **kwargs) -> MaldiSpectrumEncoder:
    defaults = dict(epochs=2, batch_size=16, warmup_epochs=1, device="cpu")
    defaults.update(kwargs)
    return MaldiSpectrumEncoder(backbone=template, **defaults).fit(_data()[0])


def _split(module: nn.Module) -> tuple[dict, dict]:
    prefix = trunk_prefix(module)
    final = {k: v for k, v in module.state_dict().items() if k.startswith(prefix + ".")}
    return trunk_state(module, prefix), final


def _assert_states_equal(a: dict, b: dict) -> None:
    assert a.keys() == b.keys()
    for key in a:
        assert torch.equal(a[key].cpu(), b[key].cpu()), key


@pytest.fixture(scope="module")
def encoders():
    cache: dict = {}

    def get(nested: bool) -> MaldiSpectrumEncoder:
        if nested not in cache:
            cache[nested] = _fit_encoder(SequentialClassifier(nested=nested))
        return cache[nested]

    return get


class TestSplitTrunk:
    @pytest.mark.parametrize("nested", [False, True])
    def test_last_linear_is_the_final_layer(self, nested):
        clf = SequentialClassifier(nested=nested)
        clf.input_dim_, clf.n_classes_ = INPUT_DIM, 2
        module = clf._build_model().eval()
        keys = list(module.state_dict())
        split = split_trunk(module)
        assert split.final_prefix == trunk_prefix(module) == FINAL_PREFIX[nested]
        assert split.feature_dim == HIDDEN
        x = torch.randn(4, INPUT_DIM)
        trunk = module.body if nested else module[:4]
        with torch.no_grad():
            assert torch.equal(split.features(x), trunk(x))
        # The module itself is untouched.
        assert isinstance(module.get_submodule(FINAL_PREFIX[nested]), nn.Linear)
        assert list(module.state_dict()) == keys

    def test_view_shares_parameters_devices_and_training_flags(self):
        module = _TrainingAware()
        split = split_trunk(module)
        assert split.final_prefix == "out" and split.feature_dim == 5
        x = torch.randn(3, 6)
        split.features(x).sum().backward()
        assert module.inner.weight.grad is not None
        assert module.out.weight.grad is None
        with torch.no_grad():
            assert torch.equal(split.features(x), 2.0 * module.inner(x))
            module.eval()
            assert torch.equal(split.features(x), module.inner(x))
        module.to(torch.float64)
        assert split.features(x.double()).dtype == torch.float64

    def test_builtin_interface_is_used_when_exposed(self):
        mlp = SpectralAttentionMLP(input_dim=INPUT_DIM, hidden_dim=8, head_dims=(4,))
        split = split_trunk(mlp)
        assert split.features == mlp.forward_features
        assert split.feature_dim == mlp.feature_dim == 4
        assert split.final_prefix == final_layer_prefix(mlp)

    def test_module_without_linear_raises(self):
        with pytest.raises(ValueError, match="no nn.Linear"):
            trunk_prefix(nn.Sequential(nn.ReLU()))

    def test_ssl_model_state_dict_layout(self):
        module = nn.Sequential(nn.Linear(INPUT_DIM, 8), nn.ReLU(), nn.Linear(8, 2))
        model = SSLModel(module, nn.Linear(8, INPUT_DIM))
        assert model.feature_dim == 8
        assert list(model.state_dict()) == [
            "backbone.0.weight",
            "backbone.0.bias",
            "backbone.2.weight",
            "backbone.2.bias",
            "pretext.weight",
            "pretext.bias",
        ]
        x = torch.randn(2, INPUT_DIM)
        with torch.no_grad():
            assert torch.equal(model(x), module[1](module[0](x)))


class TestEncoder:
    @pytest.mark.filterwarnings("ignore:pretext='contrastive' with batch_size")
    @pytest.mark.parametrize("pretext", ["mae", "shuffle+mae", "contrastive"])
    @pytest.mark.parametrize("nested", [False, True])
    def test_fit_transform_save_load(self, nested, pretext, tmp_path):
        X, _ = _data()
        enc = _fit_encoder(SequentialClassifier(nested=nested), pretext=pretext)
        assert enc.feature_dim_ == HIDDEN
        assert np.isfinite(enc.history_["val_loss"]).all()
        Z = enc.transform(X)
        assert Z.shape == (len(X), HIDDEN) and Z.dtype == np.float32
        backbone = enc.model_.backbone
        assert isinstance(backbone.get_submodule(FINAL_PREFIX[nested]), nn.Linear)
        with torch.no_grad():
            expected = split_trunk(backbone).features(
                torch.from_numpy(enc._preprocess(X))
            )
        np.testing.assert_allclose(Z, expected.numpy(), atol=1e-6, rtol=0)

        enc.save(tmp_path / "enc")
        meta = json.loads((tmp_path / "enc.json").read_text())
        assert meta["backbone_template"]["__class__"] == (
            "tests.test_custom_backbone.SequentialClassifier"
        )
        loaded = MaldiSpectrumEncoder.load(tmp_path / "enc")
        assert type(loaded.backbone) is SequentialClassifier
        assert loaded.backbone.nested is nested
        assert loaded.feature_dim_ == HIDDEN
        np.testing.assert_allclose(loaded.transform(X), Z, atol=1e-6, rtol=0)
        _assert_states_equal(loaded.model_.state_dict(), enc.model_.state_dict())

    def test_unimportable_template_class_fails_at_save(self, tmp_path):
        class LocalClassifier(SequentialClassifier):
            pass

        enc = _fit_encoder(LocalClassifier(), epochs=0)
        with pytest.raises(ValueError, match="backbone class .* cannot be re-imported"):
            enc.save(tmp_path / "enc")
        assert not list(tmp_path.iterdir())

    def test_unimportable_template_class_fails_at_load(self, tmp_path):
        _fit_encoder(SequentialClassifier(), epochs=0).save(tmp_path / "enc")
        meta = json.loads((tmp_path / "enc.json").read_text())
        for spec in (meta["params"]["backbone"], meta["backbone_template"]):
            spec["__class__"] += "Missing"
        (tmp_path / "enc.json").write_text(json.dumps(meta))
        with pytest.raises(ValueError, match="Cannot import the backbone class"):
            MaldiSpectrumEncoder.load(tmp_path / "enc")

    def test_module_without_linear_cannot_be_pretrained(self):
        class NoLinear(SequentialClassifier):
            def _build_model(self) -> nn.Module:
                return nn.Sequential(nn.ReLU())

        with pytest.raises(ValueError, match="no nn.Linear"):
            _fit_encoder(NoLinear(), epochs=0)


class TestInitFrom:
    @pytest.mark.parametrize("nested", [False, True])
    def test_trunk_transferred_and_last_linear_fresh(self, nested, encoders):
        X, y = _data()
        enc = encoders(nested)
        clf = SequentialClassifier(nested=nested, init_from=enc).fit(X, y)
        trunk, final = _split(clf.model_)
        enc_trunk, enc_final = _split(enc.model_.backbone)
        _assert_states_equal(trunk, enc_trunk)
        assert final.keys() == enc_final.keys()
        assert any(not torch.equal(final[k], enc_final[k]) for k in final)
        scratch = _split(SequentialClassifier(nested=nested).fit(X, y).model_)[0]
        assert any(not torch.equal(scratch[k], enc_trunk[k]) for k in scratch)

    def test_fine_tune_from_path_with_lpft(self, encoders, tmp_path):
        X, y = _data()
        enc = encoders(False)
        enc.save(tmp_path / "enc")
        clf = SequentialClassifier(
            init_from=str(tmp_path / "enc"), epochs=2, freeze_trunk_epochs=5
        ).fit(X, y)
        assert clf.trunk_unfroze_ is False and clf.best_epoch_phase_ == "frozen"
        _assert_states_equal(_split(clf.model_)[0], _split(enc.model_.backbone)[0])
        assert clf.predict_proba(X).shape == (len(X), 2)
        ft = SequentialClassifier(init_from=str(tmp_path / "enc"), epochs=2).fit(X, y)
        assert ft.trunk_unfroze_ is True

    def test_architecture_mismatch_raises(self, encoders):
        X, y = _data()
        clf = SequentialClassifier(hidden_dim=HIDDEN + 1, init_from=encoders(False))
        with pytest.raises(ValueError, match="shape mismatch"):
            clf.fit(X, y)

    def test_dual_branch_spectral_branch(self, encoders):
        X, y = _data()
        side = np.random.default_rng(1).integers(0, 2, (len(X), 8))
        enc = encoders(True)
        dual = MaldiDualBranchClassifier(
            n_spec_bins=INPUT_DIM,
            backbone=SequentialClassifier(nested=True, init_from=enc),
            fp_hidden_dims=(8,),
            epochs=0,
            batch_size=8,
            device="cpu",
        ).fit(np.hstack([X, side.astype(np.float32)]), y)
        spectral = dual.model_.inner.spectral
        assert isinstance(spectral.head[1], nn.Identity)
        _assert_states_equal(spectral.state_dict(), _split(enc.model_.backbone)[0])


class _HeadFirstModule(nn.Module):
    """Final layer registered before a Linear that runs earlier in forward."""

    def __init__(self, input_dim: int, hidden: int) -> None:
        super().__init__()
        self.head = nn.Linear(hidden, 2)
        self.body = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden)
        )

    def forward(self, x):
        return self.head(self.body(x))


class _SqueezedModule(nn.Module):
    """Unused Linear registered last, and a reshape after the final layer."""

    def __init__(self, input_dim: int, hidden: int) -> None:
        super().__init__()
        self.body = nn.Linear(input_dim, hidden)
        self.fc = nn.Linear(hidden, 2)
        self.unused = nn.Linear(hidden, hidden)

    def forward(self, x):
        return self.fc(self.body(x)).reshape(x.shape[0], 2)


class HeadFirstClassifier(SequentialClassifier):
    def _build_model(self) -> nn.Module:
        return _HeadFirstModule(self.input_dim_, self.hidden_dim)


class KwargsClassifier(BaseSpectralClassifier):
    """The shortest custom classifier: everything else goes through ``**kwargs``."""

    def __init__(self, hidden_dim: int = HIDDEN, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim

    def _build_model(self) -> nn.Module:
        return nn.Sequential(
            nn.Linear(self.input_dim_, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 2),
        )


class TestFinalLayerTracing:
    @pytest.mark.parametrize(
        ("module", "expected"),
        [
            (_HeadFirstModule(INPUT_DIM, HIDDEN), "head"),
            (_SqueezedModule(INPUT_DIM, HIDDEN), "fc"),
        ],
    )
    def test_the_layer_producing_the_output_is_found(self, module, expected):
        assert trunk_prefix(module, INPUT_DIM) == expected
        assert split_trunk(module, INPUT_DIM).feature_dim == HIDDEN

    def test_without_input_dim_the_last_registered_linear_is_used(self):
        assert trunk_prefix(_HeadFirstModule(INPUT_DIM, HIDDEN)) == "body.2"

    def test_a_failing_forward_falls_back_to_the_last_registered_linear(self):
        module = _HeadFirstModule(INPUT_DIM, HIDDEN)
        assert trunk_prefix(module, INPUT_DIM + 1) == "body.2"

    def test_tracing_leaves_flags_rng_and_buffers_untouched(self):
        module = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN),
            nn.BatchNorm1d(HIDDEN),
            nn.Dropout(0.5),
            nn.Linear(HIDDEN, 2),
        )
        module.train()
        module[1].eval()
        tracked = module[1].num_batches_tracked.clone()
        rng = torch.get_rng_state()
        assert trunk_prefix(module, INPUT_DIM) == "3"
        assert torch.equal(rng, torch.get_rng_state())
        assert torch.equal(tracked, module[1].num_batches_tracked)
        assert module.training and module[0].training and not module[1].training
        assert not module[0]._forward_hooks

    def test_encoder_and_init_from_use_the_traced_layer(self):
        X = np.random.default_rng(0).random((40, INPUT_DIM), dtype=np.float32)
        y = np.tile([0, 1], 20)
        template = HeadFirstClassifier(device="cpu")
        encoder = MaldiSpectrumEncoder(
            backbone=template, epochs=1, batch_size=16, device="cpu"
        ).fit(X)
        assert encoder.transform(X).shape == (40, HIDDEN)
        clf = HeadFirstClassifier(init_from=encoder, freeze_trunk_epochs=5, epochs=1)
        assert clf.fit(X, y).predict_proba(X).shape == (40, 2)
        assert torch.equal(
            clf.model_.body[2].weight, encoder.model_.backbone.body[2].weight
        )


class TestKwargsTemplate:
    def test_clone_keeps_every_parameter(self):
        clf = KwargsClassifier(
            hidden_dim=8, input_transform="log1p", epochs=7, batch_size=4
        )
        new = clone(clf)
        assert (new.hidden_dim, new.input_transform, new.epochs, new.batch_size) == (
            8,
            "log1p",
            7,
            4,
        )
        assert new.get_params()["input_transform"] == "log1p"

    def test_encoder_keeps_the_template_preprocessing(self):
        X = np.random.default_rng(0).random((40, INPUT_DIM), dtype=np.float32)
        encoder = MaldiSpectrumEncoder(
            backbone=KwargsClassifier(input_transform="log1p"),
            epochs=1,
            batch_size=16,
            device="cpu",
        ).fit(X)
        assert encoder.backbone_template_.input_transform == "log1p"
        assert encoder.input_transform_state_["mode"] == "log1p"

    def test_grid_search_and_save_load(self, tmp_path):
        X = np.random.default_rng(0).random((40, INPUT_DIM), dtype=np.float32)
        y = np.tile([0, 1], 20)
        clf = KwargsClassifier(epochs=1, batch_size=16, device="cpu")
        search = GridSearchCV(clf, {"learning_rate": [1e-3, 1e-2]}, cv=2).fit(X, y)
        assert search.best_estimator_.epochs == 1
        path = tmp_path / "kw"
        search.best_estimator_.save(path)
        loaded = KwargsClassifier.load(path)
        assert loaded.batch_size == 16
        np.testing.assert_allclose(
            loaded.predict_proba(X), search.best_estimator_.predict_proba(X), atol=1e-5
        )

    @pytest.mark.parametrize(
        "cls",
        [MaldiMLPClassifier, MaldiCNNClassifier, MaldiResNetClassifier],
    )
    def test_builtin_parameter_names_follow_the_signature(self, cls):
        names = [p for p in inspect.signature(cls.__init__).parameters if p != "self"]
        assert cls._get_param_names() == sorted(names)
