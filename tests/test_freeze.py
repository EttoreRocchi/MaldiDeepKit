"""Tests for ``freeze_trunk_epochs=`` (LP-FT trunk freezing)."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from maldideepkit import Callback
from maldideepkit.base import classifier as classifier_module
from maldideepkit.base._trunk import TrunkFreezer, final_layer_prefix, trunk_state
from maldideepkit.utils.training import EarlyStopping, train_loop
from tests.conftest import ALL_FACTORIES
from tests.test_init_from import SINGLE, _clf, _data, _encoder_trunk, _fit_encoder


@pytest.fixture(scope="module")
def encoders() -> dict:
    return {name: _fit_encoder(ALL_FACTORIES[name]()) for name in SINGLE}


class _RecordingFreezer(TrunkFreezer):
    """TrunkFreezer that records the model state at every epoch start."""

    instances: list[_RecordingFreezer] = []

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.records: list[dict] = []
        _RecordingFreezer.instances.append(self)

    def __call__(self, epoch: int, model: nn.Module) -> None:
        super().__call__(epoch, model)
        prefix = final_layer_prefix(model)
        self.records.append(
            {
                "epoch": epoch,
                "state": {k: v.clone() for k, v in model.state_dict().items()},
                "trunk_eval": all(
                    not m.training
                    for name, m in model.named_modules()
                    if not (name == prefix or name.startswith(prefix + "."))
                ),
                "final_train": model.final_layer.training,
                "trunk_grad": [p.requires_grad for p in self.trunk_params],
            }
        )


@pytest.fixture
def recording(monkeypatch):
    _RecordingFreezer.instances = []
    monkeypatch.setattr(classifier_module, "TrunkFreezer", _RecordingFreezer)
    return _RecordingFreezer.instances


def _trunk_and_final(model: nn.Module, state: dict | None = None) -> tuple[dict, dict]:
    prefix = final_layer_prefix(model)
    state = model.state_dict() if state is None else state
    trunk = {
        k: v
        for k, v in state.items()
        if not (k == prefix or k.startswith(prefix + "."))
    }
    final = {k: v for k, v in state.items() if k.startswith(prefix + ".")}
    return trunk, final


def _equal(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


class TestFrozenPhase:
    @pytest.mark.parametrize("name", SINGLE)
    def test_trunk_bitwise_frozen_then_unfrozen(self, name, encoders, recording):
        X, y = _data()
        enc = encoders[name]
        clf = _clf(
            name,
            init_from=enc,
            epochs=4,
            freeze_trunk_epochs=3,
            early_stopping_patience=10,
        ).fit(X, y)
        rec = recording[-1]
        model = clf.model_
        initial_trunk = rec.initial_trunk
        assert _equal(initial_trunk, _encoder_trunk(enc))
        assert [r["epoch"] for r in rec.records] == [0, 1, 2, 3]
        initial_final = _trunk_and_final(model, rec.records[0]["state"])[1]
        for r in rec.records[:3]:
            assert r["trunk_eval"] and r["final_train"]
            assert not any(r["trunk_grad"])
        for r in rec.records[1:]:
            # Start of epoch e = after e frozen epochs (BN buffers included).
            trunk, _ = _trunk_and_final(model, r["state"])
            assert _equal(trunk, initial_trunk)
        final_after = _trunk_and_final(model, rec.records[3]["state"])[1]
        assert not _equal(final_after, initial_final)
        assert not rec.records[3]["trunk_eval"]
        assert all(rec.records[3]["trunk_grad"])
        assert clf.trunk_unfroze_

    @pytest.mark.parametrize("name", SINGLE)
    def test_after_unfreeze_trunk_changes_or_best_is_frozen(self, name, encoders):
        X, y = _data()
        enc = encoders[name]
        clf = _clf(
            name,
            init_from=enc,
            epochs=4,
            freeze_trunk_epochs=1,
            early_stopping_patience=10,
        ).fit(X, y)
        assert clf.trunk_unfroze_
        changed = not _equal(_trunk_and_final(clf.model_)[0], _encoder_trunk(enc))
        assert changed or clf.best_epoch_phase_ == "frozen"
        assert clf.best_epoch_phase_ in {"frozen", "unfrozen"}

    def test_freeze_without_init_from(self, recording):
        X, y = _data()
        clf = _clf("MaldiCNNClassifier", epochs=2, freeze_trunk_epochs=2).fit(X, y)
        assert not clf.trunk_unfroze_
        assert clf.best_epoch_phase_ == "frozen"
        trunk = _trunk_and_final(clf.model_)[0]
        assert _equal(trunk, recording[-1].initial_trunk)

    def test_defaults_without_freezing(self):
        X, y = _data()
        clf = _clf("MaldiMLPClassifier", epochs=1).fit(X, y)
        assert clf.trunk_unfroze_ is True
        assert clf.best_epoch_phase_ == "unfrozen"

    def test_negative_value_raises(self):
        X, y = _data()
        with pytest.raises(ValueError, match="freeze_trunk_epochs"):
            _clf("MaldiMLPClassifier", freeze_trunk_epochs=-1).fit(X, y)


class TestNeverUnfroze:
    @pytest.mark.parametrize("name", SINGLE)
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"epochs": 2, "ema_decay": 0.9},
            {"epochs": 3, "swa_start_epoch": 0},
            {
                "epochs": 20,
                "learning_rate": 0.0,
                "early_stopping_patience": 1,
                "swa_start_epoch": 0,
            },
        ],
        ids=["ema", "swa", "early_stop"],
    )
    def test_final_trunk_equals_encoder(self, name, kwargs, encoders, recording):
        X, y = _data()
        enc = encoders[name]
        clf = _clf(name, init_from=enc, freeze_trunk_epochs=10, **kwargs).fit(X, y)
        assert recording[-1].last_epoch < 10
        assert not clf.trunk_unfroze_
        assert clf.best_epoch_phase_ == "frozen"
        assert _equal(_trunk_and_final(clf.model_)[0], _encoder_trunk(enc))

    def test_early_stop_inside_frozen_phase_stops_early(self, encoders, recording):
        X, y = _data()
        _clf(
            "MaldiResNetClassifier",
            init_from=encoders["MaldiResNetClassifier"],
            epochs=20,
            learning_rate=0.0,
            early_stopping_patience=1,
            freeze_trunk_epochs=10,
        ).fit(X, y)
        assert recording[-1].last_epoch == 1


class TestRequiresGradFlags:
    def test_flags_restored_including_user_frozen_param(self):
        X, y = _data()
        clf = _clf("MaldiResNetClassifier", epochs=1).fit(X, y)
        model = clf.model_
        trunk_names = [
            n
            for n, _ in model.named_parameters()
            if not n.startswith(final_layer_prefix(model) + ".")
        ]
        user_frozen = dict(model.named_parameters())[trunk_names[0]]
        user_frozen.requires_grad_(False)
        frozen_value = user_frozen.detach().clone()
        other = dict(model.named_parameters())[trunk_names[-1]]
        other_value = other.detach().clone()
        before = {n: p.requires_grad for n, p in model.named_parameters()}
        # SWA over the unfrozen epochs deploys weights trained after unfreezing.
        clf.set_params(
            epochs=3,
            freeze_trunk_epochs=1,
            swa_start_epoch=1,
            early_stopping_patience=10,
        )
        clf.fit(X, y, warm_start=True)
        assert clf.model_ is model
        assert {n: p.requires_grad for n, p in model.named_parameters()} == before
        assert torch.equal(user_frozen.detach(), frozen_value)
        assert not torch.equal(other.detach(), other_value)

    @pytest.mark.parametrize("name", SINGLE)
    def test_flags_all_true_after_fresh_fit(self, name):
        X, y = _data()
        clf = _clf(name, epochs=1, freeze_trunk_epochs=3).fit(X, y)
        assert all(p.requires_grad for p in clf.model_.parameters())

    def test_flags_restored_when_training_raises(self, monkeypatch):
        X, y = _data()
        clf = _clf("MaldiMLPClassifier", epochs=1).fit(X, y)

        def failing_train_loop(model, *args, on_epoch_start=None, **kwargs):
            on_epoch_start(0, model)
            assert not all(p.requires_grad for p in model.parameters())
            raise RuntimeError("boom")

        monkeypatch.setattr(classifier_module, "train_loop", failing_train_loop)
        clf.set_params(freeze_trunk_epochs=2)
        with pytest.raises(RuntimeError, match="boom"):
            clf.fit(X, y, warm_start=True)
        assert all(p.requires_grad for p in clf.model_.parameters())


class _FreezeProjection(Callback):
    """Freeze a layer in ``on_train_start``, recording the flag at every epoch."""

    def __init__(self):
        self.flags = []

    def on_train_start(self, model, estimator):
        model.proj.requires_grad_(False)

    def on_epoch_start(self, model, epoch):
        self.flags.append(next(model.proj.parameters()).requires_grad)


def test_parameter_frozen_in_on_train_start_stays_frozen():
    X, y = _data()
    hook = _FreezeProjection()
    clf = _clf(
        "MaldiMLPClassifier", freeze_trunk_epochs=1, epochs=3, callbacks=[hook]
    ).fit(X, y)
    assert hook.flags == [False] * len(hook.flags)
    assert not any(p.requires_grad for p in clf.model_.proj.parameters())


class TestSam:
    @pytest.mark.parametrize("name", ["MaldiMLPClassifier", "MaldiResNetClassifier"])
    def test_sam_keeps_frozen_trunk(self, name, encoders):
        X, y = _data()
        enc = encoders[name]
        clf = _clf(
            name, init_from=enc, epochs=2, freeze_trunk_epochs=5, use_sam=True
        ).fit(X, y)
        assert _equal(_trunk_and_final(clf.model_)[0], _encoder_trunk(enc))
        clf = _clf(
            name,
            init_from=enc,
            epochs=3,
            freeze_trunk_epochs=1,
            use_sam=True,
            early_stopping_patience=10,
        ).fit(X, y)
        assert clf.trunk_unfroze_


class TestEarlyStoppingAndHook:
    def test_reset_patience_keeps_best(self):
        model = nn.Linear(2, 2)
        early = EarlyStopping(patience=2)
        early.step(1.0, model)
        early.step(2.0, model)
        early.step(2.0, model)
        assert early.should_stop and early.best_epoch == 0
        early.reset_patience()
        assert not early.should_stop and early.best_loss == 1.0
        assert early.best_epoch == 0 and early.best_state is not None
        early.step(0.5, model)
        assert early.best_epoch == 3

    def test_on_epoch_start_runs_after_train(self):
        model = nn.Sequential(nn.Linear(4, 4), nn.Dropout(0.1), nn.Linear(4, 2))
        X = torch.randn(16, 4)
        y = torch.randint(0, 2, (16,))
        loader = DataLoader(TensorDataset(X, y), batch_size=8)
        seen = []

        def hook(epoch, m):
            seen.append((epoch, m.training))

        train_loop(
            model,
            loader,
            (X, y),
            nn.CrossEntropyLoss(),
            torch.optim.SGD(model.parameters(), lr=0.1),
            None,
            torch.device("cpu"),
            3,
            EarlyStopping(patience=10),
            on_epoch_start=hook,
        )
        assert seen == [(0, True), (1, True), (2, True)]

    def test_v030_positional_order_preserved(self):
        v030 = [
            "model",
            "train_loader",
            "val_tensors",
            "criterion",
            "optimizer",
            "scheduler",
            "device",
            "epochs",
            "early_stopping",
            "verbose",
            "on_epoch_end",
            "warmup_epochs",
            "grad_clip_norm",
            "use_amp",
            "swa_start_epoch",
            "use_sam",
            "metrics_recorder",
            "augment",
            "mixup_alpha",
            "cutmix_alpha",
            "n_classes",
            "mix_generator",
            "ema_decay",
            "class_weight_t",
            "n_augment_cols",
        ]
        params = list(inspect.signature(train_loop).parameters)
        assert params[: len(v030)] == v030
        assert params[len(v030) :] == [
            "on_epoch_start",
            "compiled_model",
            "amp_dtype",
            "callbacks",
        ]

    def test_freezer_unit(self):
        model = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4), nn.Linear(4, 2))
        model[0].bias.requires_grad_(False)
        early = EarlyStopping(patience=3)
        early._stale = 2
        freezer = TrunkFreezer(model, "2", 2, early)
        assert set(freezer.initial_trunk) == set(trunk_state(model, "2"))
        model.train()
        freezer(0, model)
        assert not model[0].weight.requires_grad and not model[1].training
        assert model[2].training and model[2].weight.requires_grad
        model.train()
        freezer(2, model)
        assert freezer.unfroze and early._stale == 0
        assert model[0].weight.requires_grad and not model[0].bias.requires_grad


class TestBestEpochPhase:
    @staticmethod
    def _phase(freezer, best_epoch, **params):
        clf = _clf("MaldiMLPClassifier", **params)
        return clf._best_epoch_phase(freezer, SimpleNamespace(best_epoch=best_epoch))

    def test_cases(self):
        def fz(unfroze, last_epoch=5):
            return SimpleNamespace(unfroze=unfroze, n_frozen=2, last_epoch=last_epoch)

        assert self._phase(None, 0) == "unfrozen"
        assert self._phase(fz(False), 0) == "frozen"
        assert self._phase(fz(True), 1) == "frozen"
        assert self._phase(fz(True), 2) == "unfrozen"
        assert self._phase(fz(True), None) == "unfrozen"
        assert self._phase(fz(True), 1, ema_decay=0.9) == "unfrozen"
        assert self._phase(fz(True), 1, swa_start_epoch=3) == "unfrozen"
        assert self._phase(fz(True, last_epoch=2), 1, swa_start_epoch=3) == "frozen"
