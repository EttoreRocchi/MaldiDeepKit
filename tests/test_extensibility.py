"""User choices of the training framework: optimizer, scheduler, loss, callbacks."""

from __future__ import annotations

import copy
import functools

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import GridSearchCV
from torch import nn

from maldideepkit import (
    BaseSpectralClassifier,
    Callback,
    MaldiCNNClassifier,
    MaldiDualBranchClassifier,
    MaldiMLPClassifier,
    MaldiSelfTrainingClassifier,
    MaldiSpectrumEncoder,
    get_backbone_class,
    register_backbone,
)
from maldideepkit.base import _registry
from maldideepkit.pretraining import Pretext
from maldideepkit.utils import SAMOptimizer, SpectralEnsemble, find_lr
from maldideepkit.utils.optim import build_optimizer, build_scheduler

INPUT_DIM = 64
COMMON = {"epochs": 3, "batch_size": 16, "device": "cpu", "random_state": 0}


def _data(n: int = 48) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    y = np.tile([0, 1], n // 2)
    X = rng.random((n, INPUT_DIM), dtype=np.float32)
    X[y == 1, :8] += 0.5
    return X, y


def _clf(**kwargs) -> MaldiMLPClassifier:
    return MaldiMLPClassifier(**{**COMMON, **kwargs})


def _enc(**kwargs) -> MaldiSpectrumEncoder:
    return MaldiSpectrumEncoder(**{**COMMON, **kwargs})


def _weights(estimator) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in estimator.model_.state_dict().items()}


def _assert_same_weights(a, b) -> None:
    wa, wb = _weights(a), _weights(b)
    assert set(wa) == set(wb)
    for key, value in wa.items():
        assert torch.equal(value, wb[key]), key


def _differ(a, b) -> bool:
    wa, wb = _weights(a), _weights(b)
    return any(not torch.equal(wa[k], wb[k]) for k in wa)


def _fit(estimator, X, y):
    if isinstance(estimator, MaldiSpectrumEncoder):
        return estimator.fit(X)
    return estimator.fit(X, y)


class RecordingCallback(Callback):
    def __init__(self, tag="cb"):
        self.tag = tag
        self.events: list[str] = []
        self.logs: list[dict[str, float]] = []
        self.steps: list[int] = []

    def on_train_start(self, model, estimator):
        self.events.append("train_start")
        self.estimator = estimator

    def on_epoch_start(self, model, epoch):
        self.events.append(f"epoch_start{epoch}")
        assert model.training

    def on_step_end(self, model, step):
        self.steps.append(step)

    def on_epoch_end(self, model, epoch, logs):
        self.events.append(f"epoch_end{epoch}")
        self.logs.append(dict(logs))

    def on_train_end(self, model):
        self.events.append("train_end")


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, target):
        return F.cross_entropy(logits, target, reduction="none") * self.gamma


class ScalarLoss(nn.Module):
    def forward(self, logits, target):
        return F.cross_entropy(logits, target)


def per_row_loss(logits, target):
    return F.cross_entropy(logits, target, reduction="none")


class DenoisePretext(Pretext):
    def __init__(self, noise_std=0.1):
        self.noise_std = noise_std

    def build_head(self, feature_dim, input_dim):
        return nn.Linear(feature_dim, input_dim)

    def loss(self, model, batch, generator):
        noise = torch.randn(batch.x.shape, generator=generator).to(batch.x.device)
        recon = model.pretext(model.forward_features(batch.x + self.noise_std * noise))
        return F.mse_loss(recon.float(), batch.x.float())


class SitePretext(Pretext):
    uses_target = True

    def __init__(self, n_sites=3):
        self.n_sites = n_sites

    def build_head(self, feature_dim, input_dim):
        return nn.Linear(feature_dim, self.n_sites)

    def loss(self, model, batch, generator):
        SitePretext.seen = (batch.target.dtype, bool(batch.labeled.any()))
        logits = model.pretext(model.forward_features(batch.x))[batch.labeled]
        if logits.shape[0] == 0:
            return logits.sum()
        return F.cross_entropy(logits, batch.target[batch.labeled])


class EMAPretext(DenoisePretext):
    """Reconstruction against a momentum teacher, to exercise the hooks."""

    def __init__(self, noise_std=0.1, momentum=0.9):
        super().__init__(noise_std)
        self.momentum = momentum

    def on_train_start(self, model, encoder):
        self.teacher_ = copy.deepcopy(model).requires_grad_(False)
        self.initial_ = copy.deepcopy(model.backbone.state_dict())
        self.encoder_seen_ = encoder
        self.updates_ = 0

    def on_step_end(self, model, step):
        self.updates_ += 1
        pairs = zip(
            self.teacher_.backbone.parameters(),
            model.backbone.parameters(),
            strict=True,
        )
        for teacher, student in pairs:
            teacher.data.mul_(self.momentum).add_(student.data, alpha=1 - self.momentum)

    def on_train_end(self, model):
        self.teacher_final_ = copy.deepcopy(self.teacher_.backbone.state_dict())
        del self.teacher_


class KwargsClassifier(BaseSpectralClassifier):
    def __init__(self, hidden_dim=8, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim

    def _build_model(self):
        return nn.Sequential(
            nn.Linear(self.input_dim_, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 2),
        )


class TestBuildOptimizer:
    @staticmethod
    def _params():
        return [nn.Parameter(torch.zeros(2))]

    @pytest.mark.parametrize(
        ("weight_decay", "cls"),
        [(0.0, torch.optim.Adam), (0.01, torch.optim.AdamW)],
    )
    def test_default(self, weight_decay, cls):
        opt = build_optimizer(
            None, self._params(), learning_rate=0.1, weight_decay=weight_decay
        )
        assert type(opt) is cls
        assert opt.param_groups[0]["lr"] == 0.1
        assert opt.param_groups[0]["weight_decay"] == weight_decay

    @pytest.mark.parametrize("name", ["sgd", "SGD", "Sgd"])
    def test_names_are_case_insensitive(self, name):
        opt = build_optimizer(name, self._params(), learning_rate=0.1, weight_decay=0)
        assert type(opt) is torch.optim.SGD

    def test_unknown_name(self):
        with pytest.raises(ValueError, match="Unknown optimizer"):
            build_optimizer("nope", self._params(), learning_rate=0.1, weight_decay=0)

    def test_invalid_type(self):
        with pytest.raises(TypeError, match="optimizer must be"):
            build_optimizer(3, self._params(), learning_rate=0.1, weight_decay=0)

    def test_partial_keeps_its_own_keywords(self):
        factory = functools.partial(torch.optim.SGD, lr=0.5, momentum=0.9)
        opt = build_optimizer(
            factory, self._params(), learning_rate=0.1, weight_decay=0.01
        )
        group = opt.param_groups[0]
        assert (group["lr"], group["momentum"], group["weight_decay"]) == (
            0.5,
            0.9,
            0.01,
        )

    def test_factory_receives_only_what_it_accepts(self):
        opt = build_optimizer(
            lambda params: torch.optim.SGD(params, lr=0.25),
            self._params(),
            learning_rate=0.1,
            weight_decay=0.01,
        )
        assert opt.param_groups[0]["lr"] == 0.25

    def test_optimizer_without_weight_decay_argument(self):
        opt = build_optimizer(
            "rprop", self._params(), learning_rate=0.1, weight_decay=0.01
        )
        assert type(opt) is torch.optim.Rprop

    def test_sam_wraps_the_factory(self):
        opt = build_optimizer(
            functools.partial(torch.optim.SGD, momentum=0.9),
            self._params(),
            learning_rate=0.1,
            weight_decay=0,
            use_sam=True,
        )
        assert isinstance(opt, SAMOptimizer)
        assert type(opt.base_optimizer) is torch.optim.SGD


class TestBuildScheduler:
    @staticmethod
    def _optimizer():
        return torch.optim.SGD([nn.Parameter(torch.zeros(2))], lr=0.1)

    def test_default_and_cosine(self):
        for spec in (None, "cosine"):
            sched = build_scheduler(spec, self._optimizer(), epochs=10, warmup_epochs=2)
            assert isinstance(sched, torch.optim.lr_scheduler.CosineAnnealingLR)
            assert sched.T_max == 8 and sched.eta_min == 1e-6

    def test_constant(self):
        assert build_scheduler("constant", self._optimizer(), epochs=10) is None

    def test_factory(self):
        factory = functools.partial(torch.optim.lr_scheduler.StepLR, step_size=2)
        sched = build_scheduler(factory, self._optimizer(), epochs=10)
        assert isinstance(sched, torch.optim.lr_scheduler.StepLR)

    @pytest.mark.parametrize(("spec", "error"), [("nope", ValueError), (3, TypeError)])
    def test_invalid(self, spec, error):
        with pytest.raises(error, match="scheduler"):
            build_scheduler(spec, self._optimizer(), epochs=10)


@pytest.mark.parametrize("make", [_clf, _enc], ids=["classifier", "encoder"])
class TestChoicesOnEstimators:
    def test_explicit_default_trains_identically(self, make):
        X, y = _data()
        base = _fit(make(), X, y)
        for kwargs in (
            {"optimizer": "adam"},
            {"optimizer": torch.optim.Adam},
            {"scheduler": "cosine"},
        ):
            _assert_same_weights(base, _fit(make(**kwargs), X, y))

    def test_other_optimizer_changes_the_run(self, make):
        X, y = _data()
        assert _differ(_fit(make(), X, y), _fit(make(optimizer="sgd"), X, y))

    def test_constant_schedule(self, make):
        X, y = _data()
        cb = RecordingCallback()
        _fit(make(scheduler="constant", callbacks=[cb], warmup_epochs=0), X, y)
        assert [log["lr"] for log in cb.logs] == [1e-3] * len(cb.logs)

    def test_step_schedule(self, make):
        X, y = _data()
        cb = RecordingCallback()
        factory = functools.partial(
            torch.optim.lr_scheduler.StepLR, step_size=1, gamma=0.5
        )
        _fit(make(scheduler=factory, callbacks=[cb], warmup_epochs=0), X, y)
        assert [log["lr"] for log in cb.logs] == pytest.approx([1e-3, 5e-4, 2.5e-4])

    def test_reduce_on_plateau_runs(self, make):
        X, y = _data()
        factory = functools.partial(
            torch.optim.lr_scheduler.ReduceLROnPlateau, patience=0
        )
        assert _fit(make(scheduler=factory), X, y)

    def test_invalid_values_fail_before_training(self, make):
        X, y = _data()
        cb = RecordingCallback()
        with pytest.raises(ValueError, match="optimizer"):
            _fit(make(optimizer="nope", callbacks=[cb]), X, y)
        with pytest.raises(ValueError, match="scheduler"):
            _fit(make(scheduler="nope", callbacks=[cb]), X, y)
        with pytest.raises(TypeError, match="callbacks"):
            _fit(make(callbacks=[object()]), X, y)
        assert not cb.events

    def test_hooks_order_and_logs(self, make):
        X, y = _data()
        cb = RecordingCallback()
        estimator = _fit(make(callbacks=[cb]), X, y)
        assert cb.events[0] == "train_start" and cb.events[-1] == "train_end"
        assert cb.events[1:3] == ["epoch_start0", "epoch_end0"]
        assert len(cb.logs) == 3
        assert set(cb.logs[0]) == {"train_loss", "val_loss", "lr"}
        assert np.isfinite(cb.logs[0]["train_loss"])
        assert cb.steps == list(range(len(cb.steps))) and cb.steps
        assert cb.estimator is estimator

    def test_callbacks_do_not_change_the_run(self, make):
        X, y = _data()
        _assert_same_weights(
            _fit(make(), X, y), _fit(make(callbacks=[RecordingCallback()]), X, y)
        )

    def test_a_single_callback_and_a_duck_typed_one(self, make):
        X, y = _data()

        class Duck:
            count = 0

            def on_step_end(self, model, step):
                Duck.count += 1

        _fit(make(callbacks=Duck()), X, y)
        assert Duck.count > 0

    def test_no_hook_runs_without_epochs(self, make):
        X, y = _data()
        cb = RecordingCallback()
        _fit(make(epochs=0, callbacks=[cb]), X, y)
        assert not cb.events

    def test_clones_share_the_callback(self, make):
        cb = RecordingCallback()
        assert clone(make(callbacks=[cb])).callbacks[0] is cb

    def test_save_load_round_trip(self, make, tmp_path):
        X, y = _data()
        estimator = _fit(
            make(
                optimizer=functools.partial(torch.optim.SGD, momentum=0.9),
                scheduler=functools.partial(
                    torch.optim.lr_scheduler.StepLR, step_size=2
                ),
                callbacks=[RecordingCallback(tag="keep")],
            ),
            X,
            y,
        )
        estimator.save(tmp_path / "m")
        loaded = type(estimator).load(tmp_path / "m")
        assert loaded.optimizer.func is torch.optim.SGD
        assert loaded.optimizer.keywords == {"momentum": 0.9}
        assert loaded.scheduler.func is torch.optim.lr_scheduler.StepLR
        assert loaded.callbacks[0].tag == "keep"

    def test_named_optimizer_and_class_round_trip(self, make, tmp_path):
        X, y = _data()
        estimator = _fit(make(optimizer="sgd", scheduler="constant"), X, y)
        estimator.save(tmp_path / "a")
        loaded = type(estimator).load(tmp_path / "a")
        assert (loaded.optimizer, loaded.scheduler) == ("sgd", "constant")
        estimator.set_params(optimizer=torch.optim.SGD).save(tmp_path / "b")
        assert type(estimator).load(tmp_path / "b").optimizer is torch.optim.SGD

    def test_unsavable_object_writes_nothing_and_can_be_cleared(self, make, tmp_path):
        X, y = _data()
        estimator = _fit(make(optimizer=lambda p: torch.optim.SGD(p, lr=0.1)), X, y)
        with pytest.raises(ValueError, match="optimizer"):
            estimator.save(tmp_path / "m")
        assert not list(tmp_path.iterdir())
        estimator.set_params(optimizer=None).save(tmp_path / "m")
        assert (tmp_path / "m.pt").exists()


class TestLoss:
    def test_module_function_and_class(self):
        X, y = _data()
        for loss in (
            AsymmetricLoss(gamma=1.0),
            per_row_loss,
            AsymmetricLoss,
            ScalarLoss(),
        ):
            assert _clf(loss=loss).fit(X, y).predict_proba(X).shape == (48, 2)

    def test_the_loss_drives_the_run(self):
        X, y = _data()
        assert _differ(_clf().fit(X, y), _clf(loss=AsymmetricLoss(gamma=5.0)).fit(X, y))

    def test_equivalent_object_trains_as_the_builtin(self):
        X, y = _data()
        _assert_same_weights(_clf().fit(X, y), _clf(loss=per_row_loss).fit(X, y))

    def test_the_given_module_is_not_modified(self):
        X, y = _data()
        loss = AsymmetricLoss()
        clf = _clf(loss=loss).fit(X, y)
        assert clf.loss is loss

    def test_soft_targets_with_mixup(self):
        X, y = _data()
        seen = []

        def soft_loss(logits, target):
            seen.append(target.ndim)
            return F.cross_entropy(logits, target, reduction="none")

        _clf(loss=soft_loss, mixup_alpha=0.2).fit(X, y)
        assert 2 in seen and 1 in seen

    def test_sample_weight(self):
        X, y = _data()
        weights = np.linspace(0.5, 2.0, 48)
        assert _clf(loss=per_row_loss).fit(X, y, sample_weight=weights)
        with pytest.raises(ValueError):
            _clf(loss=ScalarLoss()).fit(X, y, sample_weight=weights)

    def test_builtin_only_parameters_warn(self):
        X, y = _data()
        with pytest.warns(UserWarning, match="only apply to the built-in"):
            _clf(loss=per_row_loss, class_weight="balanced").fit(X, y)
        with pytest.warns(UserWarning, match="only apply to the built-in"):
            _clf(loss=per_row_loss, label_smoothing=0.1).fit(X, y)

    def test_invalid_value(self):
        X, y = _data()
        with pytest.raises(TypeError, match="loss must be"):
            _clf(loss=3).fit(X, y)
        with pytest.raises(ValueError, match="Unknown loss"):
            _clf(loss="nope").fit(X, y)

    @pytest.mark.parametrize("loss", [AsymmetricLoss(gamma=3.0), per_row_loss])
    def test_save_load(self, tmp_path, loss):
        X, y = _data()
        clf = _clf(loss=loss).fit(X, y)
        clf.save(tmp_path / "m")
        if isinstance(loss, nn.Module):
            loaded = MaldiMLPClassifier.load(tmp_path / "m")
            assert isinstance(loaded.loss, AsymmetricLoss) and loaded.loss.gamma == 3.0
        else:
            with pytest.raises(ValueError, match="trusted=True"):
                MaldiMLPClassifier.load(tmp_path / "m")
            loaded = MaldiMLPClassifier.load(tmp_path / "m", trusted=True)
            assert loaded.loss is per_row_loss
        np.testing.assert_allclose(
            loaded.predict_proba(X), clf.predict_proba(X), atol=1e-5
        )

    def test_torch_loss_with_tensor_argument_round_trips(self, tmp_path):
        X, y = _data()
        loss = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0]), reduction="none")
        clf = _clf(loss=loss).fit(X, y)
        clf.save(tmp_path / "m")
        loaded = MaldiMLPClassifier.load(tmp_path / "m").loss
        assert torch.equal(loaded.weight, torch.tensor([1.0, 2.0]))
        assert loaded.reduction == "none"

    def test_find_lr_uses_the_choices(self):
        X, y = _data()
        out = find_lr(_clf(loss=per_row_loss, optimizer="sgd"), X, y, num_iter=3)
        assert len(out["lrs"]) > 0


def test_retry_runs_the_start_and_end_hooks_per_attempt():
    X, y = _data()
    cb = RecordingCallback()
    _clf(retry_on_val_auroc_below=1.1, max_retries=1, callbacks=[cb]).fit(X, y)
    assert cb.events.count("train_start") == 2
    assert cb.events.count("train_end") == 2


def test_grid_search_over_the_choices():
    X, y = _data()
    grid = {"optimizer": ["adam", "sgd"], "scheduler": ["cosine", "constant"]}
    search = GridSearchCV(_clf(epochs=1), grid, cv=2).fit(X, y)
    assert search.best_params_["optimizer"] in ("adam", "sgd")


class TestPretextHooks:
    def test_momentum_teacher(self, tmp_path):
        X, y = _data()
        enc = _enc(pretext=EMAPretext(), early_stopping_patience=50).fit(X)
        pretext = enc.pretext_
        assert pretext.encoder_seen_ is enc
        assert pretext.updates_ > 0
        assert not hasattr(pretext, "teacher_")
        assert any(
            not torch.equal(pretext.teacher_final_[k], pretext.initial_[k])
            for k in pretext.initial_
        )
        assert not any("teacher" in key for key in enc.model_.state_dict())
        enc.save(tmp_path / "enc")
        loaded = MaldiSpectrumEncoder.load(tmp_path / "enc")
        np.testing.assert_allclose(loaded.transform(X), enc.transform(X), atol=1e-6)

    def test_the_given_instance_is_not_touched(self):
        X, y = _data()
        pretext = EMAPretext()
        _enc(pretext=pretext).fit(X)
        assert not hasattr(pretext, "updates_")

    def test_builtin_pretexts_have_noop_hooks(self):
        X, y = _data()
        enc = _enc().fit(X)
        assert enc.pretext_.uses_target is False

    def test_pretext_hooks_run_before_the_callbacks(self):
        X, y = _data()
        order = []

        class First(DenoisePretext):
            def on_train_start(self, model, encoder):
                order.append("pretext")

        class Second(Callback):
            def on_train_start(self, model, estimator):
                order.append("callback")

        _enc(pretext=First(), callbacks=[Second()]).fit(X)
        assert order == ["pretext", "callback"]


class TestUsesTarget:
    def test_the_pretext_reads_y_without_an_auxiliary_head(self, tmp_path):
        X, _ = _data()
        site = np.array(["A", "B", "C"] * 16)
        enc = _enc(pretext=SitePretext()).fit(X, site)
        assert SitePretext.seen == (torch.int64, True)
        assert enc.model_.aux is None
        assert list(enc.aux_classes_) == ["A", "B", "C"]
        enc.save(tmp_path / "enc")
        loaded = MaldiSpectrumEncoder.load(tmp_path / "enc")
        assert list(loaded.aux_classes_) == ["A", "B", "C"]

    def test_unlabeled_rows_are_masked(self):
        X, _ = _data()
        site = np.array(["A", "B", "C"] * 16, dtype=object)
        site[:12] = None
        assert _enc(pretext=SitePretext()).fit(X, site)

    def test_y_is_required(self):
        X, _ = _data()
        with pytest.raises(ValueError, match="uses_target"):
            _enc(pretext=SitePretext()).fit(X)

    def test_no_labeled_row(self):
        X, _ = _data()
        with pytest.raises(ValueError, match="no labeled row"):
            _enc(pretext=SitePretext()).fit(X, [None] * 48)

    def test_other_pretexts_still_ignore_y(self):
        X, y = _data()
        _assert_same_weights(_enc().fit(X), _enc().fit(X, y))


@pytest.fixture
def clean_backbones():
    saved = dict(_registry._REGISTERED)
    yield
    _registry._REGISTERED.clear()
    _registry._REGISTERED.update(saved)


@pytest.mark.usefixtures("clean_backbones")
class TestBackboneNames:
    def test_register_and_get(self):
        assert register_backbone("mine", KwargsClassifier) is KwargsClassifier
        assert get_backbone_class("mine") is KwargsClassifier
        assert get_backbone_class("cnn") is MaldiCNNClassifier

    @pytest.mark.parametrize("name", ["mlp", "peakformer", ""])
    def test_builtin_and_reserved_names_are_refused(self, name):
        with pytest.raises(ValueError):
            register_backbone(name, KwargsClassifier)

    @pytest.mark.parametrize(
        "cls", [int, BaseSpectralClassifier, MaldiDualBranchClassifier]
    )
    def test_invalid_classes_are_refused(self, cls):
        with pytest.raises(TypeError):
            register_backbone("mine", cls)

    def test_encoder(self, tmp_path):
        X, _ = _data()
        register_backbone("mine", KwargsClassifier)
        enc = _enc(backbone="mine").fit(X)
        assert isinstance(enc.backbone_template_, KwargsClassifier)
        enc.save(tmp_path / "enc")
        _registry._REGISTERED.clear()
        loaded = MaldiSpectrumEncoder.load(tmp_path / "enc")
        np.testing.assert_allclose(loaded.transform(X), enc.transform(X), atol=1e-6)

    def test_dual_branch(self, tmp_path):
        X, y = _data()
        side = np.random.default_rng(1).random((48, 8), dtype=np.float32)
        X_full = np.hstack([X, side])
        dual = MaldiDualBranchClassifier(INPUT_DIM, backbone="cnn", **COMMON)
        assert dual.fit(X_full, y).predict_proba(X_full).shape == (48, 2)
        dual.save(tmp_path / "dual")
        loaded = MaldiDualBranchClassifier.load(tmp_path / "dual")
        assert loaded.backbone == "cnn"
        np.testing.assert_allclose(
            loaded.predict_proba(X_full), dual.predict_proba(X_full), atol=1e-5
        )

    def test_dual_branch_with_a_registered_name(self):
        X, y = _data()
        X_full = np.hstack([X, np.ones((48, 4), dtype=np.float32)])
        register_backbone("mine", KwargsClassifier)
        dual = MaldiDualBranchClassifier(INPUT_DIM, backbone="mine", **COMMON)
        assert dual.fit(X_full, y).predict_proba(X_full).shape == (48, 2)

    def test_self_training(self, tmp_path):
        X, y = _data()
        y = y.copy()
        y[:8] = -1
        clf = MaldiSelfTrainingClassifier("mlp", n_iter=1).fit(X, y)
        assert isinstance(clf.estimator_, MaldiMLPClassifier)
        clf.save(tmp_path / "st")
        assert MaldiSelfTrainingClassifier.load(tmp_path / "st").base_estimator == "mlp"

    def test_ensemble(self, tmp_path):
        X, y = _data()
        ens = SpectralEnsemble(["mlp", _clf()]).fit(X, y)
        assert isinstance(ens.classifiers_[0], MaldiMLPClassifier)
        assert ens.classifiers[0] == "mlp"
        ens.save(tmp_path / "ens")
        np.testing.assert_allclose(
            SpectralEnsemble.load(tmp_path / "ens").predict_proba(X),
            ens.predict_proba(X),
            atol=1e-5,
        )


class TestSelfTraining:
    @staticmethod
    def _partial():
        X, y = _data()
        y = y.copy()
        y[:8] = -1
        return X, y

    def test_sample_weight_reaches_every_fit(self):
        X, y = self._partial()
        fits = []

        class Spy(KwargsClassifier):
            def fit(self, X, y, *, sample_weight=None, warm_start=False):
                fits.append(
                    (len(y), None if sample_weight is None else len(sample_weight))
                )
                return super().fit(X, y, sample_weight=sample_weight)

        clf = MaldiSelfTrainingClassifier(
            Spy(**COMMON), n_iter=1, confidence_threshold=0.0
        )
        clf.fit(X, y, sample_weight=np.ones(48))
        assert fits[0] == (40, 40) and fits[-1] == (48, 48)

    def test_sample_weight_length_is_checked(self):
        X, y = self._partial()
        with pytest.raises(ValueError, match="sample_weight"):
            MaldiSelfTrainingClassifier(_clf()).fit(X, y, sample_weight=np.ones(3))

    def test_load_round_trip(self, tmp_path):
        X, y = self._partial()
        clf = MaldiSelfTrainingClassifier(
            _clf(), n_iter=1, confidence_threshold=0.5, random_state=3
        ).fit(X, y)
        clf.save(tmp_path / "st")
        loaded = MaldiSelfTrainingClassifier.load(tmp_path / "st")
        assert loaded.confidence_threshold == 0.5 and loaded.random_state == 3
        assert loaded.history_ == clf.history_
        assert isinstance(loaded.base_estimator, MaldiMLPClassifier)
        assert not hasattr(loaded.base_estimator, "model_")
        np.testing.assert_allclose(
            loaded.predict_proba(X), clf.predict_proba(X), atol=1e-5
        )

    def test_v030_checkpoint_without_sidecar_loads(self, tmp_path):
        X, y = self._partial()
        clf = MaldiSelfTrainingClassifier(_clf(), n_iter=1).fit(X, y)
        clf.save(tmp_path / "st")
        (tmp_path / "st.self_training.json").unlink()
        loaded = MaldiSelfTrainingClassifier.load(tmp_path / "st")
        assert loaded.history_["n_pseudolabels"] == []
        np.testing.assert_allclose(
            loaded.predict_proba(X), clf.predict_proba(X), atol=1e-5
        )

    def test_embeddings(self):
        X, y = self._partial()
        clf = MaldiSelfTrainingClassifier(_clf(), n_iter=1).fit(X, y)
        np.testing.assert_array_equal(
            clf.get_embeddings(X), clf.estimator_.get_embeddings(X)
        )


class TestEnsemble:
    def test_clone_and_grid_search(self):
        X, y = _data()
        ens = SpectralEnsemble([_clf(), _clf(random_state=1)])
        cloned = clone(ens)
        assert cloned.classifiers[0] is not ens.classifiers[0]
        assert not hasattr(cloned.classifiers[0], "model_")
        grid = {"classifiers": [[_clf()], [_clf(), _clf(random_state=1)]]}
        search = GridSearchCV(ens, grid, cv=2).fit(X, y)
        assert search.best_estimator_.classifiers_

    def test_sample_weight_reaches_the_members(self):
        X, y = _data()
        weights = np.linspace(0.2, 3.0, 48)
        ens = SpectralEnsemble([_clf()]).fit(X, y, sample_weight=weights)
        _assert_same_weights(
            ens.classifiers_[0], _clf().fit(X, y, sample_weight=weights)
        )

    def test_custom_member_round_trip(self, tmp_path):
        X, y = _data()
        ens = SpectralEnsemble([KwargsClassifier(**COMMON)]).fit(X, y)
        ens.save(tmp_path / "ens")
        loaded = SpectralEnsemble.load(tmp_path / "ens")
        assert isinstance(loaded.classifiers_[0], KwargsClassifier)
        np.testing.assert_allclose(
            loaded.predict_proba(X), ens.predict_proba(X), atol=1e-5
        )


class TestEnsembleOfFittedMembers:
    def test_predicts_without_fit(self):
        X, y = _data()
        members = [_clf().fit(X, y), _clf(random_state=1).fit(X, y)]
        ens = SpectralEnsemble(members)
        expected = np.mean([m.predict_proba(X) for m in members], axis=0)
        np.testing.assert_allclose(ens.predict_proba(X), expected)
        assert ens.predict(X).shape == (48,)
        assert ens.score(X, y) >= 0.0

    def test_unfitted_members_raise(self):
        X, y = _data()
        with pytest.raises(NotFittedError):
            SpectralEnsemble([_clf()]).predict_proba(X)
        with pytest.raises(NotFittedError):
            SpectralEnsemble(["mlp"]).predict_proba(X)

    def test_save_load(self, tmp_path):
        X, y = _data()
        ens = SpectralEnsemble([_clf().fit(X, y)])
        ens.save(tmp_path / "ens")
        np.testing.assert_allclose(
            SpectralEnsemble.load(tmp_path / "ens").predict_proba(X),
            ens.predict_proba(X),
            atol=1e-5,
        )


def test_reconstruction_ignores_the_scale_of_a_bin_without_variance():
    rng = np.random.default_rng(0)
    X = rng.random((48, INPUT_DIM), dtype=np.float32) * 1e-3
    constant = np.arange(INPUT_DIM) == 3
    X[:, constant] = 0.0
    rows = rng.integers(0, 48, 3)
    X[rows, 3] = 5e-4
    enc = _enc(
        backbone=MaldiMLPClassifier(input_transform="log1p"),
        pretext="mae",
        epochs=2,
        warmup_epochs=0,
        val_fraction=0.25,
    ).fit(X)
    assert max(enc.history_["val_loss"]) < 1e3
