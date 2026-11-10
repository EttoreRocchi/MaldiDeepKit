"""Checkpoint parameters: codec round trips, atomic writes and class resolution."""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
import uuid

import numpy as np
import pytest
import torch

from maldideepkit import BaseSpectralClassifier, MaldiMLPClassifier
from maldideepkit.augment import SpectrumAugment
from maldideepkit.base._serialization import _import_class, _UntrustedClassError
from maldideepkit.pretraining import MaldiSpectrumEncoder
from maldideepkit.utils import SpectralEnsemble

INPUT_DIM = 32

MARKER_SOURCE = """
import pathlib

pathlib.Path({marker!r}).write_text("imported")

from maldideepkit import MaldiMLPClassifier
from maldideepkit.augment import SpectrumAugment
from maldideepkit.pretraining import MAEPretext


class MarkerAugment(SpectrumAugment):
    pass


class MarkerPretext(MAEPretext):
    pass


class MarkerMLP(MaldiMLPClassifier):
    pass
"""


def _data(n: int = 40) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    X = np.abs(rng.standard_normal((n, INPUT_DIM))).astype(np.float32)
    return X, np.arange(n) % 2


def _mlp(cls=MaldiMLPClassifier, **kwargs):
    defaults = dict(hidden_dim=8, head_dims=(4,), epochs=1, batch_size=8, device="cpu")
    defaults.update(kwargs)
    return cls(**defaults)


def _encoder(**kwargs) -> MaldiSpectrumEncoder:
    defaults = dict(
        backbone=_mlp(), epochs=1, batch_size=8, warmup_epochs=0, device="cpu"
    )
    defaults.update(kwargs)
    return MaldiSpectrumEncoder(**defaults)


@pytest.fixture
def marker_module(tmp_path, monkeypatch):
    """Return ``(module name, marker path)``: importing the module writes the marker."""
    name = f"mdk_marker_{uuid.uuid4().hex}"
    src = tmp_path / "src"
    src.mkdir()
    marker = tmp_path / "imported"
    (src / f"{name}.py").write_text(MARKER_SOURCE.format(marker=str(marker)))
    monkeypatch.syspath_prepend(str(src))
    yield name, marker
    sys.modules.pop(name, None)


def _unimport(name: str, marker) -> None:
    """Drop the marker module from ``sys.modules`` and delete its marker."""
    del sys.modules[name]
    marker.unlink()


class TestParamRoundTrip:
    def test_classifier_augment(self, tmp_path):
        X, y = _data()
        augment = SpectrumAugment(noise_std=0.1, mz_shift_max_bins=2, random_state=3)
        clf = _mlp(augment=augment).fit(X, y)
        clf.save(tmp_path / "clf")
        meta = json.loads((tmp_path / "clf.json").read_text())
        assert meta["hparams"]["augment"]["__augment__"] == (
            "maldideepkit.augment.spectra.SpectrumAugment"
        )
        loaded = MaldiMLPClassifier.load(tmp_path / "clf")
        assert type(loaded.augment) is SpectrumAugment
        assert repr(loaded.augment) == repr(augment)
        assert loaded.augment.random_state == 3
        np.testing.assert_allclose(
            loaded.predict_proba(X), clf.predict_proba(X), atol=1e-6
        )

    def test_encoder_template_augment(self, tmp_path):
        X, _ = _data()
        augment = SpectrumAugment(noise_std=0.1, blur_sigma=1.0)
        enc = _encoder(backbone=_mlp(augment=augment)).fit(X)
        enc.save(tmp_path / "enc")
        loaded = MaldiSpectrumEncoder.load(tmp_path / "enc")
        for template in (loaded.backbone, loaded.backbone_template_):
            assert repr(template.augment) == repr(augment)
        np.testing.assert_allclose(loaded.transform(X), enc.transform(X), atol=1e-6)

    @pytest.mark.parametrize("kind", ["classifier", "encoder"])
    def test_unserialisable_param_writes_nothing(self, tmp_path, kind):
        X, y = _data()
        if kind == "classifier":
            model = _mlp(augment=lambda x: x).fit(X, y)
        else:
            model = _encoder(backbone=_mlp(augment=lambda x: x), epochs=0).fit(X)
        with pytest.raises(ValueError, match="augment"):
            model.save(tmp_path / "ckpt")
        assert not list(tmp_path.iterdir())

    def test_failed_write_keeps_the_previous_checkpoint(self, tmp_path):
        class Unpicklable:
            def __reduce__(self):
                raise RuntimeError("cannot pickle this warper")

        X, y = _data()
        clf = _mlp().fit(X, y)
        clf.save(tmp_path / "clf")
        before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
        clf.warper_ = Unpicklable()
        with pytest.raises(RuntimeError, match="cannot pickle"):
            clf.save(tmp_path / "clf")
        assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


class TestClassResolution:
    def test_classifier_load(self, marker_module, tmp_path):
        name, marker = marker_module
        module = importlib.import_module(name)
        X, y = _data()
        clf = _mlp(augment=module.MarkerAugment(noise_std=0.1)).fit(X, y)
        clf.save(tmp_path / "clf")
        _unimport(name, marker)

        with pytest.raises(ValueError, match="trusted=True"):
            MaldiMLPClassifier.load(tmp_path / "clf")
        assert not marker.exists() and name not in sys.modules

        loaded = BaseSpectralClassifier.load(tmp_path / "clf", trusted=True)
        assert marker.exists()
        assert type(loaded.augment) is sys.modules[name].MarkerAugment
        np.testing.assert_allclose(
            loaded.predict_proba(X), clf.predict_proba(X), atol=1e-6
        )

        # Already imported: resolved without trusted=True (and not re-run).
        marker.unlink()
        loaded = MaldiMLPClassifier.load(tmp_path / "clf")
        assert type(loaded.augment) is sys.modules[name].MarkerAugment
        assert not marker.exists()

    def test_encoder_load(self, marker_module, tmp_path):
        name, marker = marker_module
        module = importlib.import_module(name)
        X, _ = _data()
        enc = _encoder(
            backbone=_mlp(module.MarkerMLP, augment=module.MarkerAugment()),
            pretext=module.MarkerPretext(mask_ratio=0.2),
        ).fit(X)
        enc.save(tmp_path / "enc")
        _unimport(name, marker)

        with pytest.raises(ValueError, match="trusted=True"):
            MaldiSpectrumEncoder.load(tmp_path / "enc")
        assert not marker.exists() and name not in sys.modules

        loaded = MaldiSpectrumEncoder.load(tmp_path / "enc", trusted=True)
        assert marker.exists()
        module = sys.modules[name]
        assert type(loaded.backbone) is module.MarkerMLP
        assert type(loaded.backbone_template_.augment) is module.MarkerAugment
        assert type(loaded.pretext) is module.MarkerPretext
        assert loaded.pretext.mask_ratio == 0.2
        np.testing.assert_allclose(loaded.transform(X), enc.transform(X), atol=1e-6)

        marker.unlink()
        assert type(MaldiSpectrumEncoder.load(tmp_path / "enc").pretext) is (
            module.MarkerPretext
        )
        assert not marker.exists()

    def test_ensemble_load(self, marker_module, tmp_path):
        name, marker = marker_module
        module = importlib.import_module(name)
        X, y = _data()
        ens = SpectralEnsemble([_mlp(augment=module.MarkerAugment())]).fit(X, y)
        ens.save(tmp_path / "ens")
        _unimport(name, marker)
        with pytest.raises(ValueError, match="trusted=True"):
            SpectralEnsemble.load(tmp_path / "ens")
        assert not marker.exists()
        loaded = SpectralEnsemble.load(tmp_path / "ens", trusted=True)
        assert marker.exists()
        assert type(loaded.classifiers[0].augment).__name__ == "MarkerAugment"

    def test_init_from_path_is_not_trusted(self, marker_module, tmp_path):
        name, marker = marker_module
        module = importlib.import_module(name)
        X, y = _data()
        _encoder(pretext=module.MarkerPretext()).fit(X).save(tmp_path / "enc")
        _unimport(name, marker)
        clf = _mlp(init_from=str(tmp_path / "enc"), epochs=0)
        with pytest.raises(
            ValueError, match=r"MaldiSpectrumEncoder\.load\(path, trusted"
        ):
            clf.fit(X, y)
        assert not marker.exists()
        encoder = MaldiSpectrumEncoder.load(tmp_path / "enc", trusted=True)
        assert _mlp(init_from=encoder, epochs=0).fit(X, y).predict(X).shape == (40,)

    def test_submodule_of_an_imported_package_is_not_imported(
        self, tmp_path, monkeypatch
    ):
        package = f"mdk_pkg_{uuid.uuid4().hex}"
        root = tmp_path / "src" / package
        root.mkdir(parents=True)
        (root / "__init__.py").write_text("")
        marker = tmp_path / "imported"
        (root / "sub.py").write_text(MARKER_SOURCE.format(marker=str(marker)))
        monkeypatch.syspath_prepend(str(tmp_path / "src"))
        try:
            importlib.import_module(package)
            path = f"{package}.sub.MarkerAugment"
            with pytest.raises(_UntrustedClassError, match=rf"{package}\.sub"):
                _import_class(path, SpectrumAugment, "augment")
            assert not marker.exists()
            cls = _import_class(path, SpectrumAugment, "augment", trusted=True)
            assert marker.exists() and cls.__name__ == "MarkerAugment"
        finally:
            for module in (f"{package}.sub", package):
                sys.modules.pop(module, None)

    def test_import_errors_inside_a_module_propagate(self, tmp_path, monkeypatch):
        name = f"mdk_broken_{uuid.uuid4().hex}"
        missing = f"mdk_missing_{uuid.uuid4().hex}"
        (tmp_path / f"{name}.py").write_text(f"import {missing}\n")
        monkeypatch.syspath_prepend(str(tmp_path))
        with pytest.raises(ModuleNotFoundError) as info:
            _import_class(f"{name}.Augment", SpectrumAugment, "augment", trusted=True)
        assert info.value.name == missing
        # Without trusted=True the module is not even imported.
        with pytest.raises(_UntrustedClassError):
            _import_class(f"{name}.Augment", SpectrumAugment, "augment")

    @pytest.mark.parametrize(
        ("path", "match"),
        [
            ("maldideepkit.MaldiMLPClassifier", "not a SpectrumAugment subclass"),
            ("os.system", "not a SpectrumAugment subclass"),
            ("maldideepkit.augment.spectra.Missing", "Cannot import the augment"),
            ("maldideepkit.no_such_module.Augment", "Cannot import the augment"),
        ],
    )
    def test_invalid_paths(self, path, match):
        with pytest.raises(ValueError, match=match) as info:
            _import_class(path, SpectrumAugment, "augment")
        assert not isinstance(info.value, _UntrustedClassError)


class TestSaveTimeImportability:
    def test_local_augment_class_fails_before_writing(self, tmp_path):
        class LocalAugment(SpectrumAugment):
            pass

        X, y = _data()
        clf = _mlp(augment=LocalAugment()).fit(X, y)
        with pytest.raises(ValueError, match="augment class .* cannot be re-imported"):
            clf.save(tmp_path / "clf")
        assert not list(tmp_path.iterdir())

    def test_redefined_class_fails_before_writing(self, marker_module, tmp_path):
        name, _ = marker_module
        module = importlib.import_module(name)
        X, _ = _data()
        enc = _encoder(pretext=module.MarkerPretext(), epochs=0).fit(X)
        importlib.reload(module)  # the saved path now names another class
        out = tmp_path / "out"
        with pytest.raises(ValueError, match="pretext class .* cannot be re-imported"):
            enc.save(out / "enc")
        assert not out.exists()


class TestCheckpointsAreData:
    """A modified JSON must not run code under the default ``trusted=False``."""

    @staticmethod
    def _checkpoint(tmp_path, key, spec):
        X, y = _data()
        _mlp().fit(X, y).save(tmp_path / "m")
        meta = json.loads((tmp_path / "m.json").read_text())
        meta["hparams"][key] = spec
        (tmp_path / "m.json").write_text(json.dumps(meta))
        return tmp_path / "m"

    @pytest.mark.parametrize(
        "make_spec",
        [
            lambda marker: {
                "__object__": "subprocess.Popen",
                "params": {"args": ["touch", marker]},
            },
            lambda marker: {"__class_ref__": "subprocess.Popen"},
            lambda marker: {"__function__": "os.system"},
            lambda marker: {"__function__": "maldideepkit.base.classifier.os.system"},
            lambda marker: {"__function__": "subprocess.run"},
            lambda marker: {
                "__partial__": {"__function__": "os.system"},
                "args": [f"touch {marker}"],
                "keywords": {},
            },
            lambda marker: {
                "__partial__": {"__class_ref__": "subprocess.Popen"},
                "args": [["touch", marker]],
                "keywords": {},
            },
        ],
        ids=[
            "object",
            "class_ref",
            "function",
            "function_through_attributes",
            "imported_function",
            "partial_function",
            "partial_class",
        ],
    )
    def test_load_refuses_anything_but_training_components(self, tmp_path, make_spec):
        assert subprocess.Popen is not None
        marker = tmp_path / "ran"
        path = self._checkpoint(tmp_path, "loss", make_spec(str(marker)))
        with pytest.raises(ValueError, match="trusted=True"):
            MaldiMLPClassifier.load(path)
        assert not marker.exists()

    def test_the_encoder_applies_the_same_rule(self, tmp_path):
        X, _ = _data()
        _encoder(epochs=0).fit(X).save(tmp_path / "e")
        meta = json.loads((tmp_path / "e.json").read_text())
        meta["params"]["optimizer"] = {"__function__": "os.system"}
        (tmp_path / "e.json").write_text(json.dumps(meta))
        with pytest.raises(ValueError, match="trusted=True"):
            MaldiSpectrumEncoder.load(tmp_path / "e")

    def test_trusted_load_accepts_a_function_of_an_imported_module(self, tmp_path):
        path = self._checkpoint(tmp_path, "loss", {"__function__": "os.getcwd"})
        assert MaldiMLPClassifier.load(path, trusted=True).loss is os.getcwd

    def test_torch_functions_and_components_are_allowed(self, tmp_path):
        path = self._checkpoint(
            tmp_path,
            "optimizer",
            {
                "__partial__": {"__class_ref__": "torch.optim.sgd.SGD"},
                "args": [],
                "keywords": {"momentum": 0.9},
            },
        )
        loaded = MaldiMLPClassifier.load(path)
        assert loaded.optimizer.func is torch.optim.SGD
        path = self._checkpoint(
            tmp_path, "loss", {"__function__": "torch.nn.functional.cross_entropy"}
        )
        assert MaldiMLPClassifier.load(path).loss is torch.nn.functional.cross_entropy


class TestRobustSave:
    def test_a_class_that_cannot_be_reimported_still_saves(self, tmp_path):
        class Local(MaldiMLPClassifier):
            pass

        X, y = _data()
        clf = Local(epochs=1, batch_size=8, device="cpu").fit(X, y)
        clf.save(tmp_path / "m")
        assert json.loads((tmp_path / "m.json").read_text())["class_path"] is None
        np.testing.assert_allclose(
            Local.load(tmp_path / "m").predict_proba(X), clf.predict_proba(X), atol=1e-5
        )

    def test_encoder_file_under_a_dotted_name_is_not_a_v030_checkpoint(self, tmp_path):
        X, _ = _data()
        _encoder(epochs=0).fit(X).save(tmp_path / "enc")
        with pytest.raises(FileNotFoundError):
            MaldiMLPClassifier.load(tmp_path / "enc.v2")


class KeywordAugment(SpectrumAugment):
    """Subclass whose constructor takes no ``random_state``."""

    def __init__(self, strength=0.1):
        super().__init__(noise_std=strength)
        self.strength = strength


class KwargsAugment(SpectrumAugment):
    """Subclass forwarding the base parameters through ``**kwargs``."""

    def __init__(self, strength=0.1, **kwargs):
        super().__init__(**kwargs)
        self.strength = strength


class TestAugmentSubclasses:
    @pytest.mark.parametrize(
        "augment", [KeywordAugment(), KwargsAugment(noise_std=0.05, random_state=1)]
    )
    def test_contrastive_save_load(self, tmp_path, augment):
        X, _ = _data()
        enc = _encoder(
            pretext="contrastive", augment=augment, epochs=1, batch_size=16
        ).fit(X)
        assert augment._generator is None
        enc.save(tmp_path / "e")
        loaded = MaldiSpectrumEncoder.load(tmp_path / "e")
        assert type(loaded.augment) is type(augment)
        assert loaded.augment.strength == augment.strength
        np.testing.assert_allclose(loaded.transform(X), enc.transform(X), atol=1e-6)


def test_focal_loss_class_weights_round_trip(tmp_path):
    from maldideepkit.utils.loss import FocalLoss

    rng = np.random.default_rng(0)
    X = rng.random((40, INPUT_DIM)).astype("float32")
    y = rng.integers(0, 2, 40)
    weight = torch.tensor([1.0, 5.0])
    MaldiMLPClassifier(
        hidden_dim=8,
        head_dims=(4,),
        epochs=1,
        batch_size=8,
        device="cpu",
        loss=FocalLoss(weight=weight, gamma=1.0),
    ).fit(X, y).save(tmp_path / "clf")
    loaded = MaldiMLPClassifier.load(tmp_path / "clf")
    assert torch.equal(loaded.loss.class_weight, weight)
    assert loaded.loss.gamma == 1.0
