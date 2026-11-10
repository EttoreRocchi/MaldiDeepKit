"""Tests for ``init_from=`` (trunk transfer + preprocessing inheritance)."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
import sklearn
import torch
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, cross_validate

from maldideepkit import (
    BaseDualSpectralClassifier,
    MaldiCNNClassifier,
    MaldiDualBranchClassifier,
    MaldiMLPClassifier,
    MaldiSelfTrainingClassifier,
)
from maldideepkit.base import classifier as classifier_module
from maldideepkit.base._config import same_config
from maldideepkit.base._trunk import final_layer_prefix, trunk_state
from maldideepkit.pretraining import MaldiSpectrumEncoder
from tests.conftest import ALL_FACTORIES

INPUT_DIM = 64
SINGLE = [
    "MaldiMLPClassifier",
    "MaldiCNNClassifier",
    "MaldiResNetClassifier",
    "MaldiTransformerClassifier",
]


def _data(n: int = 48) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    X = np.abs(rng.standard_normal((n, INPUT_DIM))).astype(np.float32)
    X[:, [5, 20, 40]] += 5.0
    y = np.arange(n) % 2
    return X, y


def _clf(name: str, **kwargs):
    defaults = dict(device="cpu", epochs=0, val_fraction=0.25)
    defaults.update(kwargs)
    return ALL_FACTORIES[name](**defaults)


def _fit_encoder(template) -> MaldiSpectrumEncoder:
    X, _ = _data()
    return MaldiSpectrumEncoder(
        backbone=template,
        epochs=1,
        batch_size=16,
        warmup_epochs=0,
        device="cpu",
        random_state=0,
    ).fit(X)


@pytest.fixture(scope="module")
def encoder():
    """Return a cached fitted encoder for ``(name, **template_kwargs)``."""
    cache: dict = {}

    def get(name: str, **template_kwargs) -> MaldiSpectrumEncoder:
        key = (name, tuple(sorted(template_kwargs.items())))
        if key not in cache:
            cache[key] = _fit_encoder(ALL_FACTORIES[name](**template_kwargs))
        return cache[key]

    return get


def _split(module: torch.nn.Module) -> tuple[dict, dict]:
    """Return ``(trunk, final_layer)`` state dicts of a backbone module."""
    prefix = final_layer_prefix(module)
    final = {k: v for k, v in module.state_dict().items() if k.startswith(prefix + ".")}
    return trunk_state(module, prefix), final


def _assert_states_equal(a: dict, b: dict) -> None:
    assert a.keys() == b.keys()
    for key in a:
        assert torch.equal(a[key].cpu(), b[key].cpu()), key


def _encoder_trunk(enc: MaldiSpectrumEncoder) -> dict:
    return _split(enc.model_.backbone)[0]


class TestTrunkTransfer:
    @pytest.mark.parametrize("name", SINGLE)
    def test_trunk_equals_encoder_and_final_layer_differs(self, name, encoder):
        X, y = _data()
        enc = encoder(name)
        clf = _clf(name, init_from=enc).fit(X, y)
        trunk, final = _split(clf.model_)
        _assert_states_equal(trunk, _encoder_trunk(enc))
        enc_final = _split(enc.model_.backbone)[1]
        assert final.keys() == enc_final.keys()
        assert any(not torch.equal(final[k], enc_final[k]) for k in final)

    @pytest.mark.parametrize("name", SINGLE)
    def test_without_init_from_trunk_is_random(self, name, encoder):
        X, y = _data()
        clf = _clf(name).fit(X, y)
        trunk = _split(clf.model_)[0]
        enc_trunk = _encoder_trunk(encoder(name))
        assert any(
            not torch.equal(trunk[k], enc_trunk[k])
            for k in trunk
            if trunk[k].is_floating_point()
        )

    def test_extra_encoder_keys_raise(self, encoder):
        X, y = _data()
        enc = encoder("MaldiMLPClassifier", use_attention=True)
        clf = _clf("MaldiMLPClassifier", use_attention=False, init_from=enc)
        with pytest.raises(ValueError, match="absent from this classifier"):
            clf.fit(X, y)

    def test_missing_encoder_keys_raise(self, encoder):
        X, y = _data()
        enc = encoder("MaldiMLPClassifier", use_attention=False)
        clf = _clf("MaldiMLPClassifier", use_attention=True, init_from=enc)
        with pytest.raises(ValueError, match="missing from the encoder"):
            clf.fit(X, y)

    def test_shape_mismatch_raises_with_arch_params(self, encoder):
        X, y = _data()
        enc = encoder("MaldiMLPClassifier")
        clf = _clf("MaldiMLPClassifier", hidden_dim=24, init_from=enc)
        with pytest.raises(ValueError, match="shape mismatch") as info:
            clf.fit(X, y)
        assert "hidden_dim=32" in str(info.value)
        assert "hidden_dim=24" in str(info.value)

    def test_other_architecture_raises(self, encoder):
        X, y = _data()
        clf = _clf("MaldiMLPClassifier", init_from=encoder("MaldiCNNClassifier"))
        with pytest.raises(ValueError, match="does not match"):
            clf.fit(X, y)

    def test_unfitted_encoder_raises(self):
        X, y = _data()
        clf = _clf("MaldiMLPClassifier", init_from=MaldiSpectrumEncoder())
        with pytest.raises(ValueError, match="unfitted"):
            clf.fit(X, y)

    def test_wrong_type_raises(self):
        X, y = _data()
        with pytest.raises(TypeError, match="init_from must be"):
            _clf("MaldiMLPClassifier", init_from=3).fit(X, y)


class TestPreprocessingInheritance:
    @pytest.mark.parametrize("name", SINGLE)
    def test_fitted_preprocessing_is_the_encoders(self, name, encoder):
        X, y = _data()
        enc = encoder(name, input_transform="log1p+standardize")
        clf = _clf(name, input_transform="log1p+standardize", init_from=enc)
        clf.fit(X, y)
        np.testing.assert_array_equal(clf.feature_mean_, enc.feature_mean_)
        np.testing.assert_array_equal(clf.feature_std_, enc.feature_std_)
        assert clf.input_transform_state_["mode"] == "log1p+standardize"
        assert clf.input_transform_state_ is not enc.input_transform_state_
        scratch = _clf(name, input_transform="log1p+standardize").fit(X, y)
        assert not np.array_equal(scratch.feature_mean_, enc.feature_mean_)
        np.testing.assert_array_equal(clf._preprocess_input(X), enc._preprocess(X))

    @pytest.mark.parametrize("name", SINGLE)
    def test_resolved_mode_mismatch_raises(self, name, encoder):
        X, y = _data()
        enc = encoder(name)
        other = "robust" if enc.input_transform_state_["mode"] != "robust" else "none"
        clf = _clf(name, input_transform=other, init_from=enc)
        with pytest.raises(ValueError, match="input_transform mismatch"):
            clf.fit(X, y)

    def test_standardize_aliases_are_accepted(self, encoder):
        X, y = _data()
        enc = encoder("MaldiMLPClassifier", standardize=True)
        clf = _clf("MaldiMLPClassifier", input_transform="standardize", init_from=enc)
        clf.fit(X, y)
        np.testing.assert_array_equal(clf.feature_mean_, enc.feature_mean_)
        enc2 = encoder("MaldiMLPClassifier", input_transform="standardize")
        _clf("MaldiMLPClassifier", standardize=True, init_from=enc2).fit(X, y)

    def test_input_dim_mismatch_raises(self, encoder):
        X, y = _data()
        clf = _clf("MaldiMLPClassifier", init_from=encoder("MaldiMLPClassifier"))
        with pytest.raises(ValueError, match="input_dim mismatch"):
            clf.fit(X[:, :32], y)


@pytest.fixture(scope="module")
def warped_encoder():
    pytest.importorskip("maldiamrkit.alignment")
    from maldiamrkit.alignment import Warping

    return _fit_encoder(ALL_FACTORIES["MaldiMLPClassifier"](warping=Warping()))


class TestRealWarping:
    def test_refit_and_clone_after_load(self, warped_encoder, tmp_path):
        from maldiamrkit.alignment import Warping

        X, y = _data()
        clf = _clf("MaldiMLPClassifier", warping=Warping(), init_from=warped_encoder)
        clf.fit(X, y).save(tmp_path / "clf")
        loaded = MaldiMLPClassifier.load(tmp_path / "clf")
        assert same_config(loaded.warping, loaded.warper_)
        assert not hasattr(loaded.warping, "ref_")
        loaded.init_from = warped_encoder
        loaded.fit(X, y, warm_start=True)
        clone(loaded).fit(X, y)

    def test_identical_config_accepted(self, warped_encoder):
        from maldiamrkit.alignment import Warping

        X, y = _data()
        clf = _clf("MaldiMLPClassifier", warping=Warping(), init_from=warped_encoder)
        clf.fit(X, y)
        assert clf.warper_ is not warped_encoder.warper_
        np.testing.assert_array_equal(
            clf.warper_.ref_spec_, warped_encoder.warper_.ref_spec_
        )
        _assert_states_equal(_split(clf.model_)[0], _encoder_trunk(warped_encoder))

    def test_different_method_rejected(self, warped_encoder):
        from maldiamrkit.alignment import Warping

        X, y = _data()
        clf = _clf(
            "MaldiMLPClassifier",
            warping=Warping(method="dtw"),
            init_from=warped_encoder,
        )
        with pytest.raises(ValueError, match="warping mismatch"):
            clf.fit(X, y)

    def test_warping_on_one_side_only_rejected(self, warped_encoder, encoder):
        from maldiamrkit.alignment import Warping

        X, y = _data()
        with pytest.raises(ValueError, match="warping mismatch"):
            _clf("MaldiMLPClassifier", init_from=warped_encoder).fit(X, y)
        clf = _clf(
            "MaldiMLPClassifier",
            warping=Warping(),
            init_from=encoder("MaldiMLPClassifier"),
        )
        with pytest.raises(ValueError, match="warping mismatch"):
            clf.fit(X, y)


class TestRetriesAndWarmStart:
    def test_every_retry_reloads_the_trunk(self, encoder, monkeypatch):
        X, y = _data()
        enc = encoder("MaldiMLPClassifier")
        calls = []
        original = classifier_module.BaseSpectralClassifier._build_initial_model

        def counting(self, device, enc_arg):
            calls.append(enc_arg)
            return original(self, device, enc_arg)

        monkeypatch.setattr(
            classifier_module.BaseSpectralClassifier, "_build_initial_model", counting
        )
        clf = _clf(
            "MaldiMLPClassifier",
            init_from=enc,
            retry_on_val_auroc_below=1.01,
            max_retries=2,
        ).fit(X, y)
        assert len(calls) == 3
        assert all(c is enc for c in calls)
        _assert_states_equal(_split(clf.model_)[0], _encoder_trunk(enc))

    def test_warm_start_takes_precedence(self, encoder):
        X, y = _data()
        enc = encoder("MaldiMLPClassifier")
        clf = _clf("MaldiMLPClassifier", init_from=enc, epochs=2).fit(X, y)
        trained = {k: v.clone() for k, v in clf.model_.state_dict().items()}
        assert any(
            not torch.equal(trained[k], v) for k, v in _encoder_trunk(enc).items()
        )
        model = clf.model_
        clf.set_params(epochs=0)
        clf.fit(X, y, warm_start=True)
        assert clf.model_ is model
        _assert_states_equal(clf.model_.state_dict(), trained)

    def test_path_and_instance_fit_identically(self, encoder, tmp_path):
        X, y = _data()
        enc = encoder("MaldiCNNClassifier")
        enc.save(tmp_path / "enc")
        by_path = _clf("MaldiCNNClassifier", init_from=str(tmp_path / "enc"), epochs=1)
        by_instance = _clf("MaldiCNNClassifier", init_from=enc, epochs=1)
        np.testing.assert_array_equal(
            by_path.fit(X, y).predict_proba(X), by_instance.fit(X, y).predict_proba(X)
        )


class TestCloneAndParams:
    def test_clone_keeps_encoder_reference(self, encoder):
        enc = encoder("MaldiMLPClassifier")
        clf = _clf("MaldiMLPClassifier", init_from=enc)
        twin = clone(clf)
        assert twin.init_from is enc
        assert clone(_clf("MaldiMLPClassifier", init_from="enc")).init_from == "enc"

    def test_clone_keeps_metadata_routing(self, encoder):
        pytest.importorskip("sklearn", minversion="1.4")
        X, y = _data()
        enc = encoder("MaldiMLPClassifier")
        with sklearn.config_context(enable_metadata_routing=True):
            clf = _clf("MaldiMLPClassifier", init_from=enc, epochs=1)
            clf.set_fit_request(sample_weight=True)
            twin = clone(clf)
            assert twin.init_from is enc
            assert twin.get_metadata_routing().fit.requests["sample_weight"] is True
            sw = np.ones(len(y), dtype=np.float32)
            out = cross_validate(clf, X, y, cv=2, params={"sample_weight": sw})
            assert len(out["test_score"]) == 2

    def test_set_params_changed_value_raises(self, encoder):
        enc = encoder("MaldiMLPClassifier")
        clf = _clf("MaldiMLPClassifier", init_from=enc)
        with pytest.raises(ValueError, match="read-only"):
            clf.set_params(init_from__epochs=7)
        assert enc.epochs == 1

    def test_set_params_noop_round_trip_accepted(self, encoder):
        enc = encoder("MaldiMLPClassifier")
        before = enc.get_params(deep=False)
        clf = _clf("MaldiMLPClassifier", init_from=enc)
        clf.set_params(**clf.get_params(deep=True))
        assert clf.init_from is enc
        assert enc.get_params(deep=False) == before

    def test_set_params_on_unfitted_encoder_is_forwarded(self):
        clf = _clf("MaldiMLPClassifier", init_from=MaldiSpectrumEncoder())
        clf.set_params(init_from__epochs=7)
        assert clf.init_from.epochs == 7

    def test_grid_search_with_encoder_instance(self, encoder):
        X, y = _data()
        enc = encoder("MaldiMLPClassifier")
        search = GridSearchCV(
            _clf("MaldiMLPClassifier", init_from=enc, epochs=1),
            {"learning_rate": [1e-3, 1e-2]},
            cv=2,
        ).fit(X, y)
        assert search.best_estimator_.init_from is enc
        assert search.best_estimator_.predict(X).shape == (len(X),)

    def test_grid_search_over_checkpoint_paths(self, encoder, tmp_path):
        X, y = _data()
        enc_a = encoder("MaldiMLPClassifier")
        enc_b = clone(enc_a).set_params(random_state=1).fit(X)
        paths = [str(tmp_path / "enc0.05"), str(tmp_path / "enc0.10")]
        enc_a.save(paths[0])
        enc_b.save(paths[1])
        search = GridSearchCV(
            _clf("MaldiMLPClassifier", epochs=1),
            {"init_from": paths},
            cv=2,
            error_score="raise",
        ).fit(X, y)
        assert [p["init_from"] for p in search.cv_results_["params"]] == paths
        assert np.isfinite(search.cv_results_["mean_test_score"]).all()
        assert search.best_estimator_.init_from in paths

    def test_grid_search_over_encoder_instances_raises(self, encoder):
        X, y = _data()
        enc_a = encoder("MaldiMLPClassifier")
        enc_b = clone(enc_a).set_params(random_state=1).fit(X)
        search = GridSearchCV(
            _clf("MaldiMLPClassifier", epochs=1),
            {"init_from": [enc_a, enc_b]},
            cv=2,
            error_score="raise",
        )
        with pytest.raises(ValueError, match="search over checkpoint paths"):
            search.fit(X, y)

    def test_self_training_smoke(self, encoder):
        X, y = _data()
        enc = encoder("MaldiMLPClassifier")
        y_partial = y.copy()
        y_partial[24:] = -1
        clf = MaldiSelfTrainingClassifier(
            base_estimator=_clf("MaldiMLPClassifier", init_from=enc, epochs=1),
            n_iter=2,
            confidence_threshold=0.5,
        ).fit(X, y_partial)
        assert clf.estimator_.init_from is enc
        assert clf.predict(X).shape == (len(X),)

    @pytest.mark.parametrize("factory_name", sorted(ALL_FACTORIES))
    def test_every_factory_exposes_and_forwards(self, factory_name):
        params = ALL_FACTORIES[factory_name]().get_params()
        assert params["init_from"] is None
        assert params["freeze_trunk_epochs"] == 0
        clf = ALL_FACTORIES[factory_name](init_from="enc", freeze_trunk_epochs=3)
        params = clone(clf).get_params()
        assert params["init_from"] == "enc"
        assert params["freeze_trunk_epochs"] == 3


class TestDualBranch:
    @staticmethod
    def _X_full() -> tuple[np.ndarray, np.ndarray]:
        X, y = _data()
        side = np.random.default_rng(1).integers(0, 2, (len(X), 8))
        return np.hstack([X, side.astype(np.float32)]), y

    @pytest.mark.parametrize("name", SINGLE)
    def test_template_init_from_loads_spectral_branch(self, name, encoder):
        X_full, y = self._X_full()
        enc = encoder(name)
        dual = MaldiDualBranchClassifier(
            n_spec_bins=INPUT_DIM,
            backbone=ALL_FACTORIES[name](init_from=enc),
            fp_hidden_dims=(8,),
            input_transform=enc.input_transform_state_["mode"],
            epochs=0,
            batch_size=8,
            device="cpu",
        ).fit(X_full, y)
        _assert_states_equal(
            dual.model_.inner.spectral.state_dict(), _encoder_trunk(enc)
        )
        assert dual.input_transform_state_ == enc.input_transform_state_

    def test_template_preprocessing_inherited(self, encoder):
        X_full, y = self._X_full()
        enc = encoder("MaldiMLPClassifier", input_transform="standardize")
        dual = MaldiDualBranchClassifier(
            n_spec_bins=INPUT_DIM,
            backbone=ALL_FACTORIES["MaldiMLPClassifier"](init_from=enc),
            fp_hidden_dims=(8,),
            input_transform="standardize",
            epochs=0,
            batch_size=8,
            device="cpu",
        ).fit(X_full, y)
        np.testing.assert_array_equal(dual.feature_mean_, enc.feature_mean_)
        assert dual.feature_mean_.shape == (INPUT_DIM,)

    def test_outer_preprocessing_mismatch_raises(self, encoder):
        X_full, y = self._X_full()
        enc = encoder("MaldiMLPClassifier")
        dual = MaldiDualBranchClassifier(
            n_spec_bins=INPUT_DIM,
            backbone=ALL_FACTORIES["MaldiMLPClassifier"](
                init_from=enc, input_transform="log1p"
            ),
            fp_hidden_dims=(8,),
            input_transform="log1p",
            epochs=0,
            device="cpu",
        )
        with pytest.raises(ValueError, match="input_transform mismatch"):
            dual.fit(X_full, y)

    def test_n_spec_bins_mismatch_raises(self, encoder):
        X_full, y = self._X_full()
        dual = MaldiDualBranchClassifier(
            n_spec_bins=INPUT_DIM - 8,
            backbone=ALL_FACTORIES["MaldiMLPClassifier"](
                init_from=encoder("MaldiMLPClassifier")
            ),
            fp_hidden_dims=(8,),
            epochs=0,
            device="cpu",
        )
        with pytest.raises(ValueError, match="input_dim mismatch"):
            dual.fit(X_full, y)

    @pytest.mark.parametrize(
        "kwargs", [{"init_from": "enc"}, {"freeze_trunk_epochs": 2}]
    )
    def test_custom_dual_classifier_raises_at_fit(self, kwargs):
        class CustomDual(BaseDualSpectralClassifier):
            def _build_dual_model(self, *, input_dim_spec, input_dim_side, n_classes):
                raise AssertionError("not reached")

        X_full, y = self._X_full()
        dual = CustomDual(n_spec_bins=INPUT_DIM, epochs=0, device="cpu", **kwargs)
        with pytest.raises(ValueError, match="does not define which module"):
            dual.fit(X_full, y)

    def test_init_from_on_the_classifier_takes_the_encoder_architecture(
        self, encoder, tmp_path
    ):
        enc = encoder("MaldiCNNClassifier")
        X_full, y = self._X_full()
        dual = MaldiDualBranchClassifier(
            n_spec_bins=INPUT_DIM, init_from=enc, epochs=0, device="cpu"
        ).fit(X_full, y)
        _assert_states_equal(
            dict(dual.model_.inner.spectral.state_dict()), _encoder_trunk(enc)
        )
        dual.save(tmp_path / "dual")
        loaded = MaldiDualBranchClassifier.load(tmp_path / "dual")
        assert isinstance(loaded.backbone, MaldiCNNClassifier)
        np.testing.assert_allclose(
            loaded.predict_proba(X_full), dual.predict_proba(X_full), atol=1e-5
        )

    def test_init_from_on_both_raises(self, encoder):
        enc = encoder("MaldiMLPClassifier")
        X_full, y = self._X_full()
        dual = MaldiDualBranchClassifier(
            n_spec_bins=INPUT_DIM,
            backbone=ALL_FACTORIES["MaldiMLPClassifier"](init_from=enc),
            init_from=enc,
            epochs=0,
            device="cpu",
        )
        with pytest.raises(ValueError, match="both"):
            dual.fit(X_full, y)

    @pytest.mark.parametrize("where", ["classifier", "template"])
    def test_freeze_keeps_the_spectral_branch_and_trains_the_rest(self, encoder, where):
        enc = encoder("MaldiMLPClassifier")
        X_full, y = self._X_full()
        template_kwargs = {"freeze_trunk_epochs": 3} if where == "template" else {}
        outer_kwargs = {"freeze_trunk_epochs": 3} if where == "classifier" else {}
        dual = MaldiDualBranchClassifier(
            n_spec_bins=INPUT_DIM,
            backbone=ALL_FACTORIES["MaldiMLPClassifier"](
                init_from=enc, **template_kwargs
            ),
            epochs=2,
            batch_size=16,
            device="cpu",
            **outer_kwargs,
        ).fit(X_full, y)
        assert dual.trunk_unfroze_ is False
        _assert_states_equal(
            dict(dual.model_.inner.spectral.state_dict()), _encoder_trunk(enc)
        )

    def test_clone_keeps_template_encoder(self, encoder):
        enc = encoder("MaldiMLPClassifier")
        dual = MaldiDualBranchClassifier(
            n_spec_bins=INPUT_DIM, backbone=MaldiMLPClassifier(init_from=enc)
        )
        assert clone(dual).backbone.init_from is enc


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TestSerialisation:
    def test_save_load_with_path(self, encoder, tmp_path):
        X, y = _data()
        enc = encoder("MaldiResNetClassifier")
        enc.save(tmp_path / "enc")
        clf = _clf(
            "MaldiResNetClassifier",
            init_from=str(tmp_path / "enc.pt"),
            epochs=1,
            freeze_trunk_epochs=1,
        ).fit(X, y)
        clf.save(tmp_path / "clf")
        meta = json.loads((tmp_path / "clf.json").read_text())
        assert meta["hparams"]["init_from"] == str((tmp_path / "enc").resolve())
        assert meta["hparams"]["freeze_trunk_epochs"] == 1
        assert meta["init_from_provenance"] == {
            "path": str((tmp_path / "enc").resolve()),
            "sha256": _sha256(tmp_path / "enc.pt"),
        }
        for suffix in (".pt", ".json"):
            (tmp_path / "enc").with_suffix(suffix).unlink()
        loaded = type(clf).load(tmp_path / "clf")
        assert loaded.freeze_trunk_epochs == 1
        assert loaded.init_from == str((tmp_path / "enc").resolve())
        np.testing.assert_allclose(loaded.predict_proba(X), clf.predict_proba(X))
        loaded.save(tmp_path / "again")
        again = json.loads((tmp_path / "again.json").read_text())
        assert again["init_from_provenance"] == meta["init_from_provenance"]

    def test_save_load_with_instance(self, encoder, tmp_path):
        X, y = _data()
        enc = encoder("MaldiMLPClassifier")
        clf = _clf("MaldiMLPClassifier", init_from=enc, epochs=1).fit(X, y)
        clf.save(tmp_path / "clf")
        meta = json.loads((tmp_path / "clf.json").read_text())
        assert meta["hparams"]["init_from"] is None
        assert meta["init_from_provenance"] == {"path": None, "sha256": None}
        loaded = MaldiMLPClassifier.load(tmp_path / "clf")
        assert loaded.init_from is None
        np.testing.assert_allclose(loaded.predict_proba(X), clf.predict_proba(X))

    def test_relative_init_from_is_stored_as_the_fit_time_path(
        self, encoder, tmp_path, monkeypatch
    ):
        X, y = _data()
        X_full, _ = TestDualBranch._X_full()
        (tmp_path / "fit").mkdir()
        (tmp_path / "elsewhere").mkdir()
        encoder("MaldiMLPClassifier").save(tmp_path / "fit" / "enc")
        expected = str((tmp_path / "fit" / "enc").resolve())
        monkeypatch.chdir(tmp_path / "fit")
        clf = _clf("MaldiMLPClassifier", init_from="enc", epochs=1).fit(X, y)
        dual = MaldiDualBranchClassifier(
            n_spec_bins=INPUT_DIM,
            backbone=MaldiMLPClassifier(
                hidden_dim=32, head_dims=(16,), init_from="enc"
            ),
            fp_hidden_dims=(8,),
            epochs=1,
            batch_size=8,
            device="cpu",
        ).fit(X_full, y)
        monkeypatch.chdir(tmp_path / "elsewhere")
        clf.save(tmp_path / "clf")
        dual.save(tmp_path / "dual")
        meta = json.loads((tmp_path / "clf.json").read_text())
        assert meta["hparams"]["init_from"] == expected
        assert meta["init_from_provenance"]["path"] == expected
        dual_meta = json.loads((tmp_path / "dual.json").read_text())
        assert dual_meta["hparams"]["backbone"]["params"]["init_from"] == expected

    def test_dotted_encoder_paths_stay_distinct(self, encoder, tmp_path):
        X, y = _data()
        enc_a = encoder("MaldiMLPClassifier")
        enc_b = clone(enc_a).set_params(random_state=1).fit(X)
        enc_a.save(tmp_path / "enc_mask0.05")
        enc_b.save(tmp_path / "enc_mask0.10")
        assert not (tmp_path / "enc_mask0.pt").exists()
        for name, enc in (("enc_mask0.05", enc_a), ("enc_mask0.10", enc_b)):
            clf = _clf("MaldiMLPClassifier", init_from=str(tmp_path / name)).fit(X, y)
            _assert_states_equal(_split(clf.model_)[0], _encoder_trunk(enc))
            assert clf._init_from_provenance == {
                "path": str((tmp_path / name).resolve()),
                "sha256": _sha256(tmp_path / f"{name}.pt"),
            }

    def test_lpft_phase_attributes_round_trip(self, encoder, tmp_path):
        X, y = _data()
        clf = _clf(
            "MaldiMLPClassifier",
            init_from=encoder("MaldiMLPClassifier"),
            epochs=2,
            freeze_trunk_epochs=5,
        ).fit(X, y)
        assert clf.trunk_unfroze_ is False and clf.best_epoch_phase_ == "frozen"
        clf.save(tmp_path / "clf")
        loaded = MaldiMLPClassifier.load(tmp_path / "clf")
        assert loaded.trunk_unfroze_ is False
        assert loaded.best_epoch_phase_ == "frozen"
        # Checkpoints written before v0.4.0 have neither key.
        meta = json.loads((tmp_path / "clf.json").read_text())
        del meta["fitted"]["trunk_unfroze_"], meta["fitted"]["best_epoch_phase_"]
        (tmp_path / "clf.json").write_text(json.dumps(meta))
        old = MaldiMLPClassifier.load(tmp_path / "clf")
        assert not hasattr(old, "trunk_unfroze_")
        assert not hasattr(old, "best_epoch_phase_")

    def test_no_init_from_has_no_provenance(self, tmp_path):
        X, y = _data()
        _clf("MaldiMLPClassifier").fit(X, y).save(tmp_path / "clf")
        meta = json.loads((tmp_path / "clf.json").read_text())
        assert meta["init_from_provenance"] is None
        assert meta["hparams"]["init_from"] is None

    def test_dual_template_path_round_trip(self, encoder, tmp_path):
        X_full, y = TestDualBranch._X_full()
        enc = encoder("MaldiMLPClassifier")
        enc.save(tmp_path / "enc")
        dual = MaldiDualBranchClassifier(
            n_spec_bins=INPUT_DIM,
            backbone=MaldiMLPClassifier(
                hidden_dim=32, head_dims=(16,), init_from=str(tmp_path / "enc")
            ),
            fp_hidden_dims=(8,),
            epochs=1,
            batch_size=8,
            device="cpu",
        ).fit(X_full, y)
        dual.save(tmp_path / "dual")
        meta = json.loads((tmp_path / "dual.json").read_text())
        assert meta["hparams"]["backbone"]["params"]["init_from"] == str(
            (tmp_path / "enc").resolve()
        )
        assert meta["init_from_provenance"]["sha256"] == _sha256(tmp_path / "enc.pt")
        loaded = MaldiDualBranchClassifier.load(tmp_path / "dual")
        np.testing.assert_allclose(
            loaded.predict_proba(X_full), dual.predict_proba(X_full)
        )
