"""Tests for :mod:`maldideepkit.pretraining`."""

from __future__ import annotations

import json
import platform
import tracemalloc
import warnings

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from torch import nn
from torch.utils.data import DataLoader

from maldideepkit import (
    MaldiCNNClassifier,
    MaldiDualBranchClassifier,
    MaldiMLPClassifier,
    MaldiResNetClassifier,
    MaldiTransformerClassifier,
    make_loaders,
)
from maldideepkit.attention.mlp import SpectralAttentionMLP
from maldideepkit.augment import SpectrumAugment
from maldideepkit.base._config import same_config
from maldideepkit.base.data import (
    INPUT_TRANSFORMS,
    _warp_numpy,
    apply_fitted_preprocessing,
    apply_input_transform,
    fit_input_transform,
)
from maldideepkit.pretraining import (
    ContrastivePretext,
    MAEPretext,
    MaldiSpectrumEncoder,
    Pretext,
    ShuffleMAEPretext,
    ShufflePretext,
    get_pretext_class,
    make_pretraining_loaders,
    register_pretext,
)
from maldideepkit.pretraining import _pretext as pretext_module
from maldideepkit.pretraining import data as pretraining_data
from maldideepkit.pretraining._pretext import (
    CONTRASTIVE_AUGMENT,
    PRETEXTS,
    MAEHead,
    SSLModel,
    _mae_batch_loss,
    contrastive_loss,
    contrastive_views,
    mae_loss,
    nt_xent_loss,
    random_bin_mask,
    shuffle_corruption,
    shuffle_loss,
)
from maldideepkit.pretraining._training import (
    SSLBatch,
    _FittedTransform,
    _loss_sum_dtype,
    prepare_batch,
    ssl_train_loop,
)
from maldideepkit.pretraining.data import stack_collate
from maldideepkit.utils.training import EarlyStopping

INPUT_DIM = 64
# round(0.15 * 64) bins sampled / masked per spectrum by default.
K_DEFAULT = 10
IGNORE_SMALL_BATCH = pytest.mark.filterwarnings(
    "ignore:pretext='contrastive' with batch_size:UserWarning"
)

TEMPLATES = {
    "mlp": lambda **kw: MaldiMLPClassifier(hidden_dim=16, head_dims=(8,), **kw),
    "cnn": lambda **kw: MaldiCNNClassifier(
        channels=(4, 8), kernel_size=5, head_dim=8, **kw
    ),
    "resnet": lambda **kw: MaldiResNetClassifier(
        stem_channels=4, stage_channels=(4, 8), blocks_per_stage=(1, 1), **kw
    ),
    "transformer": lambda **kw: MaldiTransformerClassifier(
        embed_dim=8, depth=1, num_heads=2, patch_size=4, head_dim=8, **kw
    ),
}


class _MeanShiftWarper(BaseEstimator, TransformerMixin):
    """Deterministic warper: ``fit`` stores the column mean, ``transform`` subtracts it."""

    def fit(self, X, y=None):
        self.ref_ = np.asarray(X, dtype=np.float32).mean(axis=0)
        return self

    def transform(self, X):
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(np.asarray(X))
        return pd.DataFrame(df.to_numpy() - self.ref_, index=df.index)


def _encoder(backbone, **kwargs) -> MaldiSpectrumEncoder:
    defaults = dict(
        epochs=2, batch_size=16, warmup_epochs=1, device="cpu", random_state=0
    )
    defaults.update(kwargs)
    return MaldiSpectrumEncoder(backbone=backbone, **defaults)


@pytest.fixture
def pool(rng) -> np.ndarray:
    X = np.abs(rng.standard_normal((40, INPUT_DIM))).astype(np.float32)
    X[:, [5, 20, 40]] += 5.0
    return X


class TestPretrainingLoaders:
    def test_split_yields_raw_rows_and_train_fitted_state(self, pool):
        tr, val, stats = make_pretraining_loaders(
            pool,
            batch_size=8,
            val_size=0.25,
            random_state=0,
            transform_mode="log1p+standardize",
        )
        idx_tr, idx_val = stats["train_indices"], stats["val_indices"]
        assert len(idx_tr) == 30 and len(idx_val) == 10
        assert np.array_equal(np.sort(np.concatenate([idx_tr, idx_val])), np.arange(40))
        expected_state = fit_input_transform(pool[idx_tr], "log1p+standardize")
        np.testing.assert_array_equal(stats["mean"], expected_state["mean"])
        np.testing.assert_array_equal(stats["std"], expected_state["std"])
        assert stats["sample_weight_val"] is None
        assert stats["n_labeled"] == 0
        x_raw, target, labeled = val.dataset[0]
        assert torch.equal(x_raw, torch.from_numpy(pool[idx_val[0]]))
        assert not bool(labeled)
        xb, tb, mb = next(iter(tr))
        assert xb.shape == (8, INPUT_DIM) and tb.shape == (8,) and mb.shape == (8,)

    def test_warper_fit_on_train_split_and_pool_warped_once(self, pool):
        tr, val, stats = make_pretraining_loaders(
            pool, batch_size=8, val_size=0.25, warper=_MeanShiftWarper()
        )
        ref = stats["warper"].ref_
        np.testing.assert_allclose(
            ref, pool[stats["train_indices"]].mean(axis=0), rtol=1e-5
        )
        assert not np.allclose(ref, pool.mean(axis=0))
        np.testing.assert_array_equal(tr.dataset.X, pool - ref)
        assert tr.dataset.X is val.dataset.X

    def test_memmap_pool_is_not_copied(self, pool, tmp_path):
        np.save(tmp_path / "pool.npy", pool)
        mm = np.load(tmp_path / "pool.npy", mmap_mode="r")
        tr, _, _ = make_pretraining_loaders(mm, batch_size=8)
        assert np.shares_memory(tr.dataset.X, mm)
        xb, _, _ = next(iter(tr))
        assert xb.dtype == torch.float32

    @pytest.mark.parametrize("mode", INPUT_TRANSFORMS)
    def test_transform_state_equals_fit_on_train_rows(self, pool, mode):
        _, _, stats = make_pretraining_loaders(pool, batch_size=8, transform_mode=mode)
        expected = fit_input_transform(pool[stats["train_indices"]], mode)
        state = stats["input_transform_state"]
        assert state.keys() == expected.keys()
        for key, value in expected.items():
            np.testing.assert_array_equal(state[key], value)

    @pytest.mark.parametrize("mode", ["none", "log1p"])
    def test_stateless_mode_does_not_copy_train_rows(self, tmp_path, monkeypatch, mode):
        big = np.ones((2000, 512), dtype=np.float32)
        np.save(tmp_path / "pool.npy", big)
        mm = np.load(tmp_path / "pool.npy", mmap_mode="r")
        # The reconstruction statistics read the training rows in chunks
        # of _RECON_CHUNK_BYTES (16 MB, larger than this pool): shrink it.
        monkeypatch.setattr(pretraining_data, "_RECON_CHUNK_BYTES", 16 * 512 * 4)
        tracemalloc.start()
        try:
            make_pretraining_loaders(mm, batch_size=8, transform_mode=mode)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        # Copying the training rows would allocate ~90% of the 4 MB pool.
        assert peak < big.nbytes // 8

    def test_partial_int_targets(self, pool):
        y = np.array([0, 1, 2, -1] * 10)
        tr, _, stats = make_pretraining_loaders(
            pool, y, batch_size=8, aux_target="classification"
        )
        np.testing.assert_array_equal(stats["aux_classes"], [0, 1, 2])
        assert stats["n_labeled"] == 30
        ds = tr.dataset
        np.testing.assert_array_equal(ds.labeled.numpy(), y != -1)
        np.testing.assert_array_equal(ds.targets.numpy(), y)
        _, target, labeled = ds[0]
        j = ds.indices[0]
        assert target.dtype == torch.int64
        assert int(target) == y[j] and bool(labeled) == (y[j] != -1)

    def test_partial_string_targets(self, pool):
        y = np.array(["S", "R", "?", None] * 10, dtype=object)
        tr, _, stats = make_pretraining_loaders(
            pool, y, aux_target="classification", unlabeled_marker="?"
        )
        np.testing.assert_array_equal(stats["aux_classes"], ["R", "S"])
        ds = tr.dataset
        np.testing.assert_array_equal(
            ds.labeled.numpy(), [True, True, False, False] * 10
        )
        np.testing.assert_array_equal(ds.targets.numpy(), [1, 0, -1, -1] * 10)

    def test_partial_float_targets_with_nan_marker(self, pool):
        y = np.array([0.25, -1.0, np.nan, 3.5] * 10)
        tr, _, stats = make_pretraining_loaders(
            pool, y, aux_target="regression", unlabeled_marker=np.nan
        )
        assert stats["aux_classes"] is None
        ds = tr.dataset
        assert ds.targets.dtype == torch.float32
        np.testing.assert_array_equal(
            ds.labeled.numpy(), [True, True, False, True] * 10
        )
        labeled = ds.labeled.numpy()
        np.testing.assert_array_equal(
            ds.targets.numpy()[labeled], y[labeled].astype(np.float32)
        )
        assert np.isnan(ds.targets.numpy()[~labeled]).all()

    def test_regression_with_numeric_marker_warns(self, pool):
        y = np.array([0.25, -1.0, 2.0, 3.5] * 10)
        with pytest.warns(UserWarning, match="unlabeled_marker"):
            tr, _, _ = make_pretraining_loaders(pool, y, aux_target="regression")
        np.testing.assert_array_equal(
            tr.dataset.labeled.numpy(), [True, False, True, True] * 10
        )

    def test_y_without_aux_target_raises(self, pool):
        with pytest.raises(ValueError, match="aux_target"):
            make_pretraining_loaders(pool, np.zeros(40))

    def test_unknown_aux_target_raises(self, pool):
        with pytest.raises(ValueError, match="aux_target"):
            make_pretraining_loaders(pool, np.zeros(40), aux_target="ranking")


class TestSSLBatches:
    def test_per_batch_transform_equals_make_loaders_values(self, pool):
        """Per-batch transform == array-level ``make_loaders`` values, bit for bit."""
        y = np.repeat([0, 1], 20)
        tr, val, stats = make_pretraining_loaders(
            pool,
            batch_size=7,
            val_size=0.25,
            random_state=0,
            warper=_MeanShiftWarper(),
            transform_mode="log1p+standardize",
        )
        state = stats["input_transform_state"]
        ml_tr, ml_val, _ = make_loaders(
            pool,
            y,
            batch_size=7,
            val_size=0.25,
            random_state=0,
            fitted_preprocessing={
                "warper": stats["warper"],
                "input_transform_state": state,
            },
        )
        idx_tr, idx_val = train_test_split(
            np.arange(40), test_size=0.25, random_state=0, stratify=y
        )
        array_level = np.empty_like(pool)
        array_level[idx_tr] = ml_tr.dataset.X.numpy()
        array_level[idx_val] = ml_val.dataset.X.numpy()

        for ds in (tr.dataset, val.dataset):
            loader = DataLoader(
                ds, batch_size=7, shuffle=False, collate_fn=stack_collate
            )
            per_batch = torch.cat(
                [prepare_batch(b, state, torch.device("cpu")).x for b in loader]
            )
            assert torch.equal(per_batch, torch.from_numpy(array_level[ds.indices]))

    def test_custom_collate_fn_is_still_transformed(self, pool):
        calls = {"n": 0}

        def custom_collate(items):
            calls["n"] += 1
            xs, ts, ms = zip(*items, strict=True)
            return [torch.stack(xs), torch.stack(ts), torch.stack(ms)]

        tr, val, stats = make_pretraining_loaders(
            pool, batch_size=8, transform_mode="log1p", collate_fn=custom_collate
        )
        seen: list[tuple[torch.Tensor, torch.Tensor]] = []

        def spy_loss(model, batch, generator):
            seen.append((batch.x.clone(), batch.x_raw.clone()))
            return model(batch.x).pow(2).mean()

        torch.manual_seed(0)
        backbone = SpectralAttentionMLP(
            input_dim=INPUT_DIM, n_classes=2, hidden_dim=8, head_dims=(4,)
        )
        model = SSLModel(backbone, MAEHead(4, INPUT_DIM))
        ssl_train_loop(
            model,
            tr,
            val,
            spy_loss,
            torch.optim.Adam(model.parameters()),
            torch.device("cpu"),
            1,
            EarlyStopping(patience=3),
            input_transform_state=stats["input_transform_state"],
        )
        assert calls["n"] > 0 and len(seen) == calls["n"]
        for x, x_raw in seen:
            expected = apply_input_transform(x_raw.numpy(), {"mode": "log1p"})
            assert torch.equal(x, torch.from_numpy(expected))
            assert not torch.equal(x, x_raw)

    def test_random_bin_mask_is_seeded_and_exact(self):
        def draw(seed):
            gen = torch.Generator().manual_seed(seed)
            return random_bin_mask(5, INPUT_DIM, 0.25, generator=gen, device="cpu")

        mask = draw(0)
        assert mask.dtype == torch.bool and mask.shape == (5, INPUT_DIM)
        assert (mask.sum(dim=1) == 16).all()
        assert torch.equal(mask, draw(0))
        assert not torch.equal(mask, draw(1))


class TestEncoderFitTransform:
    @IGNORE_SMALL_BATCH
    @pytest.mark.parametrize("pretext", PRETEXTS)
    @pytest.mark.parametrize("name", list(TEMPLATES))
    def test_fit_transform_save_load(self, pool, name, pretext, tmp_path):
        enc = _encoder(TEMPLATES[name](), pretext=pretext).fit(pool)
        Z = enc.transform(pool)
        assert Z.shape == (len(pool), enc.feature_dim_)
        assert Z.dtype == np.float32
        assert enc.feature_dim_ == enc.model_.backbone.feature_dim
        assert enc.input_dim_ == enc.n_features_in_ == INPUT_DIM
        assert len(enc.history_["val_loss"]) == 2
        assert np.isfinite(enc.history_["train_loss"]).all()
        assert np.isfinite(enc.history_["val_loss"]).all()
        assert enc.__sklearn_is_fitted__()
        assert enc.model_.aux is None

        enc.save(tmp_path / "enc.pt")
        assert (tmp_path / "enc.pt").exists() and (tmp_path / "enc.json").exists()
        assert not (tmp_path / "enc.warper.pkl").exists()
        state = torch.load(tmp_path / "enc.pt", weights_only=True)
        assert {k.split(".")[0] for k in state} == {"backbone", "pretext"}
        meta = json.loads((tmp_path / "enc.json").read_text())
        assert meta["kind"] == "encoder" and meta["version"] == 1
        assert meta["params"]["pretext"] == pretext
        assert meta["backbone_template"]["__estimator__"] == type(enc.backbone).__name__
        assert meta["fitted"]["feature_dim_"] == enc.feature_dim_

        loaded = MaldiSpectrumEncoder.load(tmp_path / "enc")
        np.testing.assert_allclose(loaded.transform(pool), Z, atol=1e-6, rtol=0)
        assert loaded.pretext == pretext
        assert loaded.input_transform_state_ == enc.input_transform_state_
        assert type(loaded.backbone) is type(enc.backbone)
        assert loaded.feature_dim_ == enc.feature_dim_
        loaded_state = loaded.model_.state_dict()
        assert loaded_state.keys() == enc.model_.state_dict().keys()
        for k, v in enc.model_.state_dict().items():
            assert torch.equal(v.cpu(), loaded_state[k].cpu())

    @IGNORE_SMALL_BATCH
    @pytest.mark.parametrize("pretext", PRETEXTS)
    def test_loss_depends_only_on_the_generator(self, pool, pretext):
        """Same seed -> same loss whatever the global RNG (validation is re-seeded)."""
        enc = _encoder(TEMPLATES["cnn"](), pretext=pretext, epochs=1).fit(pool)
        _, val, _ = make_pretraining_loaders(pool, batch_size=16)
        batch = prepare_batch(
            next(iter(val)), enc.input_transform_state_, torch.device("cpu")
        )

        def loss(seed, global_seed):
            torch.manual_seed(global_seed)
            with torch.no_grad():
                gen = torch.Generator().manual_seed(seed)
                return enc._ssl_loss(enc.model_, batch, gen).item()

        assert loss(1, 0) == loss(1, 123)
        assert loss(1, 0) != loss(2, 0)

    @IGNORE_SMALL_BATCH
    @pytest.mark.parametrize("pretext", ["shuffle", "shuffle+mae", "contrastive"])
    def test_validation_batch_of_one(self, pool, pretext):
        # 4 validation rows with batch_size=3: the last validation batch has 1 row.
        enc = _encoder(TEMPLATES["mlp"](), pretext=pretext, batch_size=3).fit(pool)
        assert np.isfinite(enc.history_["val_loss"]).all()

    def test_pretext_heads(self, pool):
        def heads(pretext, **kw):
            return _encoder(TEMPLATES["mlp"](), pretext=pretext, epochs=0, **kw).fit(
                pool
            )

        enc = heads("shuffle")
        assert enc.model_.pretext.discriminator.out_features == INPUT_DIM
        enc = heads("shuffle+mae")
        assert set(enc.model_.pretext.keys()) == {"shuffle", "mae"}
        assert {k.split(".")[1] for k in enc.model_.state_dict() if "pretext" in k} == {
            "shuffle",
            "mae",
        }
        with pytest.warns(UserWarning, match="batch_size"):
            enc = heads("contrastive", projection_dim=5)
        assert enc.model_.pretext.mlp[-1].out_features == 5

    def test_epochs_zero_fits_preprocessing_only(self, pool):
        enc_a = _encoder(TEMPLATES["mlp"](input_transform="standardize"), epochs=0)
        enc_a.fit(pool)
        enc_b = _encoder(TEMPLATES["mlp"](input_transform="standardize"), epochs=0)
        enc_b.fit(pool)
        trained = _encoder(TEMPLATES["mlp"](input_transform="standardize")).fit(pool)
        assert enc_a.history_["train_loss"] == []
        assert enc_a.input_transform_state_["mode"] == "standardize"
        np.testing.assert_array_equal(enc_a.feature_mean_, trained.feature_mean_)
        state_a = enc_a.model_.state_dict()
        assert all(
            torch.equal(v, enc_b.model_.state_dict()[k]) for k, v in state_a.items()
        )
        assert not all(
            torch.equal(v, trained.model_.state_dict()[k]) for k, v in state_a.items()
        )
        assert enc_a.transform(pool).shape == (len(pool), enc_a.feature_dim_)

    def test_y_is_ignored_without_aux_head(self, pool):
        y = np.repeat([0, 1], 20)
        Z_y = _encoder(TEMPLATES["mlp"]()).fit(pool, y).transform(pool)
        Z = _encoder(TEMPLATES["mlp"]()).fit(pool).transform(pool)
        np.testing.assert_array_equal(Z_y, Z)

    def test_pipeline_and_clone(self, pool):
        y = np.repeat([0, 1], 20)
        pipe = Pipeline(
            [("enc", _encoder(TEMPLATES["cnn"]())), ("lr", LogisticRegression())]
        ).fit(pool, y)
        assert pipe.predict(pool).shape == (40,)
        enc = pipe.named_steps["enc"]
        fresh = clone(enc)
        assert not hasattr(fresh, "model_")
        assert fresh.get_params()["backbone__channels"] == (4, 8)


class TestEncoderPreprocessing:
    def test_string_backbone_uses_class_defaults(self, pool):
        enc = _encoder("resnet", epochs=0).fit(pool)
        assert isinstance(enc.backbone_template_, MaldiResNetClassifier)
        assert enc.backbone_template_.input_transform == "log1p"
        assert enc.input_transform_state_ == {"mode": "log1p"}
        assert enc.feature_mean_ is None and enc.warper_ is None
        assert enc.backbone == "resnet"

    def test_template_preprocessing_is_inherited(self, pool):
        template = TEMPLATES["mlp"](
            input_transform="log1p+standardize", warping=_MeanShiftWarper()
        )
        enc = _encoder(template).fit(pool)
        assert not hasattr(template.warping, "ref_")
        assert enc.input_transform_state_["mode"] == "log1p+standardize"
        _, _, stats = make_pretraining_loaders(
            pool,
            batch_size=16,
            val_size=0.1,
            random_state=0,
            warper=_MeanShiftWarper(),
            transform_mode="log1p+standardize",
        )
        np.testing.assert_array_equal(enc.warper_.ref_, stats["warper"].ref_)
        np.testing.assert_array_equal(enc.feature_mean_, stats["mean"])
        np.testing.assert_array_equal(enc.feature_std_, stats["std"])

        X_proc = apply_input_transform(
            _warp_numpy(enc.warper_, pool), enc.input_transform_state_
        )
        np.testing.assert_array_equal(
            X_proc,
            apply_fitted_preprocessing(
                pool,
                warper=enc.warper_,
                input_transform_state=enc.input_transform_state_,
            ),
        )
        with torch.no_grad():
            expected = enc.model_.backbone.forward_features(torch.from_numpy(X_proc))
        np.testing.assert_allclose(enc.transform(pool), expected.numpy(), atol=1e-6)

    def test_legacy_standardize_flag_resolves(self, pool):
        enc = _encoder(TEMPLATES["mlp"](standardize=True), epochs=0).fit(pool)
        assert enc.input_transform_state_["mode"] == "standardize"
        assert enc.feature_mean_ is not None

    def test_save_load_with_warper(self, pool, tmp_path):
        template = TEMPLATES["cnn"](input_transform="log1p", warping=_MeanShiftWarper())
        enc = _encoder(template).fit(pool)
        enc.save(tmp_path / "enc")
        assert (tmp_path / "enc.warper.pkl").exists()
        loaded = MaldiSpectrumEncoder.load(tmp_path / "enc.json")
        np.testing.assert_array_equal(loaded.warper_.ref_, enc.warper_.ref_)
        np.testing.assert_allclose(
            loaded.transform(pool), enc.transform(pool), atol=1e-6, rtol=0
        )
        # Re-saving without a warper removes the stale pickle.
        _encoder(TEMPLATES["cnn"]()).fit(pool).save(tmp_path / "enc")
        assert not (tmp_path / "enc.warper.pkl").exists()


class TestEncoderPersistenceAndOutput:
    def test_dotted_base_names_stay_distinct(self, pool, tmp_path):
        encs = {
            name: _encoder(TEMPLATES["mlp"](), mask_ratio=ratio, epochs=1).fit(pool)
            for name, ratio in (("enc_mask0.05", 0.05), ("enc_mask0.10", 0.10))
        }
        for name, enc in encs.items():
            enc.save(tmp_path / name)
        assert not (tmp_path / "enc_mask0.pt").exists()
        for name, enc in encs.items():
            for path in (name, f"{name}.pt", f"{name}.json"):
                loaded = MaldiSpectrumEncoder.load(tmp_path / path)
                assert loaded.mask_ratio == enc.mask_ratio
                np.testing.assert_allclose(
                    loaded.transform(pool), enc.transform(pool), atol=1e-6, rtol=0
                )

    def test_set_output_pandas(self, pool):
        enc = _encoder(TEMPLATES["mlp"](), epochs=1)
        with pytest.raises(NotFittedError):
            enc.get_feature_names_out()
        enc.set_output(transform="pandas").fit(pool)
        index = [f"s{i}" for i in range(len(pool))]
        Z = enc.transform(pd.DataFrame(pool, index=index))
        names = [f"maldispectrumencoder{i}" for i in range(enc.feature_dim_)]
        assert isinstance(Z, pd.DataFrame)
        assert list(Z.columns) == names
        assert list(Z.index) == index
        assert list(enc.get_feature_names_out()) == names
        np.testing.assert_allclose(
            Z.to_numpy(), enc.set_output(transform="default").transform(pool)
        )

    @pytest.mark.parametrize(
        "kwargs", [{"init_from": "enc"}, {"freeze_trunk_epochs": 2}]
    )
    def test_template_transfer_params_warn(self, pool, kwargs):
        with pytest.warns(UserWarning, match="MaldiSpectrumEncoder ignores"):
            _encoder(TEMPLATES["mlp"](**kwargs), epochs=0).fit(pool)


class TestEncoderErrors:
    def test_peakformer_is_reserved(self, pool):
        with pytest.raises(NotImplementedError, match="v0.5.0"):
            _encoder("peakformer").fit(pool)

    def test_unknown_backbone_string(self, pool):
        with pytest.raises(ValueError, match="Unknown backbone"):
            _encoder("lstm").fit(pool)

    def test_dual_template_rejected(self, pool):
        with pytest.raises(TypeError, match="single-input"):
            _encoder(MaldiDualBranchClassifier(n_spec_bins=32)).fit(pool)

    def test_non_estimator_backbone_rejected(self, pool):
        with pytest.raises(TypeError, match="backbone"):
            _encoder(object()).fit(pool)

    def test_unknown_pretext(self, pool):
        with pytest.raises(ValueError, match="pretext"):
            _encoder("mlp", pretext="jigsaw").fit(pool)

    @pytest.mark.parametrize(
        ("param", "value"),
        [
            ("mask_ratio", 0.0),
            ("shuffle_fraction", 1.5),
            ("projection_dim", 0),
            ("temperature", 0.0),
            ("aux_lambda", -1.0),
            ("aux_target", "ranking"),
        ],
    )
    def test_invalid_ssl_params(self, pool, param, value):
        with pytest.raises(ValueError, match=param):
            _encoder("mlp", epochs=0, **{param: value}).fit(pool)

    def test_template_input_dim_mismatch(self, pool):
        with pytest.raises(ValueError, match="input_dim"):
            _encoder(TEMPLATES["mlp"](input_dim=32)).fit(pool)

    def test_transform_checks(self, pool):
        enc = _encoder(TEMPLATES["mlp"]())
        with pytest.raises(NotFittedError):
            enc.transform(pool)
        enc.fit(pool)
        with pytest.raises(ValueError, match="features"):
            enc.transform(pool[:, :32])

    def test_load_rejects_classifier_checkpoint(self, pool, tmp_path):
        clf = MaldiMLPClassifier(hidden_dim=8, head_dims=(4,), epochs=1, batch_size=8)
        clf.fit(pool, np.repeat([0, 1], 20)).save(tmp_path / "clf")
        with pytest.raises(ValueError, match="not a MaldiSpectrumEncoder"):
            MaldiSpectrumEncoder.load(tmp_path / "clf")


def _shuffle(x, x_raw=None, *, mask_ratio=0.15, shuffle_fraction=0.5, seed=0):
    gen = torch.Generator().manual_seed(seed)
    return shuffle_corruption(
        x,
        x if x_raw is None else x_raw,
        mask_ratio,
        shuffle_fraction,
        generator=gen,
    )


class TestLossSums:
    def test_epoch_losses_are_python_float_sums(self, pool, monkeypatch):
        """Train and validation losses are summed in ``float64`` on CPU / CUDA."""
        records = []
        ssl_loss = MaldiSpectrumEncoder._ssl_loss

        def spy(self, model, batch, generator):
            loss = ssl_loss(self, model, batch, generator)
            records.append((model.training, float(loss), int(batch.x.shape[0])))
            return loss

        monkeypatch.setattr(MaldiSpectrumEncoder, "_ssl_loss", spy)
        enc = _encoder("mlp", pretext="shuffle+mae", epochs=1).fit(pool)
        train = [loss for training, loss, _ in records if training]
        val = [(loss, n) for training, loss, n in records if not training]
        assert len(train) > 1
        assert enc.history_["train_loss"] == [sum(train) / len(train)]
        assert enc.history_["val_loss"] == [
            sum(loss * n for loss, n in val) / sum(n for _, n in val)
        ]

    @pytest.mark.parametrize(
        ("device", "dtype"),
        [("cpu", torch.float64), ("cuda", torch.float64), ("mps", torch.float32)],
    )
    def test_sum_dtype(self, device, dtype):
        """MPS has no ``float64``: its sums stay in ``float32``."""
        assert _loss_sum_dtype(torch.device(device)) is dtype


class TestShuffleCorruption:
    def test_sampling_frequency_follows_intensity(self, rng):
        intensity = rng.exponential(1.0, INPUT_DIM).astype(np.float32)
        x = torch.from_numpy(np.tile(intensity, (4000, 1)))
        _, sampled, _ = _shuffle(x)
        assert (sampled.sum(dim=1) == K_DEFAULT).all()
        freq = sampled.float().mean(dim=0).numpy()
        assert spearmanr(freq, intensity)[0] > 0.9

    def test_noise_iff_value_changed(self, rng):
        x = torch.from_numpy(
            np.abs(rng.standard_normal((8, INPUT_DIM)), dtype=np.float32)
        )
        x_corrupt, sampled, noise = _shuffle(x)
        changed = x_corrupt != x
        assert torch.equal(noise, changed)
        assert not (changed & ~sampled).any()
        n_swap = round(0.5 * K_DEFAULT)
        assert (changed.sum(dim=1) == n_swap).all()
        # Each spectrum receives the same bins of one other spectrum.
        for i in range(x.shape[0]):
            cols = changed[i].nonzero().squeeze(1)
            donors = [
                d
                for d in range(x.shape[0])
                if d != i and torch.equal(x_corrupt[i, cols], x[d, cols])
            ]
            assert donors
        assert x_corrupt.data_ptr() != x.data_ptr()

    def test_zero_for_zero_and_equal_swaps_are_real(self):
        zeros = torch.zeros((6, INPUT_DIM))
        x_corrupt, sampled, noise = _shuffle(zeros, shuffle_fraction=1.0)
        assert (sampled.sum(dim=1) == K_DEFAULT).all()
        assert torch.equal(x_corrupt, zeros) and not noise.any()

        same = torch.zeros((6, INPUT_DIM))
        same[:, [3, 7, 11]] = torch.tensor([1.0, 2.0, 3.0])
        x_corrupt, sampled, noise = _shuffle(same, shuffle_fraction=1.0)
        assert sampled[:, [3, 7, 11]].all()
        assert torch.equal(x_corrupt, same) and not noise.any()

    def test_negative_inputs_and_batch_of_one(self, rng):
        x_raw = torch.from_numpy(rng.standard_normal((8, INPUT_DIM)).astype(np.float32))
        x_corrupt, sampled, noise = _shuffle(x_raw)
        assert (sampled.sum(dim=1) == K_DEFAULT).all()
        # Non-positive bins only get the weight floor: never sampled here.
        assert (x_raw[sampled] > 0).all()
        assert noise.any()

        one = x_raw[:1].abs()
        x_corrupt, sampled, noise = _shuffle(one)
        assert torch.equal(x_corrupt, one)
        assert sampled.sum() == K_DEFAULT and not noise.any()

    @pytest.mark.parametrize("mode", INPUT_TRANSFORMS)
    def test_swapping_transformed_values_equals_transforming_swapped_raw(
        self, rng, mode
    ):
        raw = (3.0 * rng.standard_normal((16, INPUT_DIM))).astype(np.float32)
        raw[rng.random(raw.shape) < 0.5] = 0.0
        state = fit_input_transform(raw, mode)
        x = torch.from_numpy(apply_input_transform(raw, state).copy())
        x_raw = torch.from_numpy(raw)
        # Same seed and weights -> same sampled bins and donors.
        corrupt_x, _, _ = _shuffle(x, x_raw, shuffle_fraction=1.0)
        corrupt_raw, _, _ = _shuffle(x_raw, x_raw, shuffle_fraction=1.0)
        np.testing.assert_allclose(
            corrupt_x.numpy(),
            apply_input_transform(corrupt_raw.numpy(), state),
            rtol=1e-6,
            atol=0,
        )


class TestContrastive:
    def test_small_batch_warns(self, pool):
        with pytest.warns(UserWarning, match="batch_size=16"):
            _encoder("mlp", pretext="contrastive", epochs=0).fit(pool)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            _encoder("mlp", pretext="contrastive", epochs=0, batch_size=64).fit(pool)
            _encoder("mlp", pretext="mae", epochs=0).fit(pool)

    def test_views_are_seeded_augmented_and_masked(self, pool):
        x_raw = torch.from_numpy(pool[:8])

        def views(seed):
            gen = torch.Generator().manual_seed(seed)
            return contrastive_views(
                x_raw,
                lambda raw: torch.log1p(raw),
                0.25,
                generator=gen,
                device=torch.device("cpu"),
            )

        a, b = views(0)
        assert a.shape == b.shape == x_raw.shape
        assert not torch.equal(a, b)
        assert ((a == 0).sum(dim=1) >= 16).all()
        a2, b2 = views(0)
        assert torch.equal(a, a2) and torch.equal(b, b2)
        assert not torch.equal(a, views(1)[0])

    def test_nt_xent(self):
        p = torch.eye(4)
        aligned = nt_xent_loss(p, p, temperature=0.1)
        shuffled = nt_xent_loss(p, p[[1, 2, 3, 0]], temperature=0.1)
        assert aligned < shuffled
        single = nt_xent_loss(torch.ones(1, 4), torch.ones(1, 4), temperature=0.1)
        assert single.item() == 0.0


class TestAuxHead:
    def test_classification_with_string_labels_and_marker(self, pool, tmp_path):
        y = np.array(["S", "R", "?", None] * 10, dtype=object)
        enc = _encoder(TEMPLATES["mlp"](), aux_lambda=1.0, unlabeled_marker="?").fit(
            pool, y
        )
        np.testing.assert_array_equal(enc.aux_classes_, ["R", "S"])
        assert enc.model_.aux.out_features == 2
        assert np.isfinite(enc.history_["val_loss"]).all()

        enc.save(tmp_path / "enc")
        state = torch.load(tmp_path / "enc.pt", weights_only=True)
        assert {k.split(".")[0] for k in state} == {"backbone", "pretext", "aux"}
        loaded = MaldiSpectrumEncoder.load(tmp_path / "enc")
        np.testing.assert_array_equal(loaded.aux_classes_, enc.aux_classes_)
        for k, v in enc.model_.aux.state_dict().items():
            assert torch.equal(v, loaded.model_.aux.state_dict()[k])
        np.testing.assert_allclose(
            loaded.transform(pool), enc.transform(pool), atol=1e-6, rtol=0
        )

    def test_regression_uses_fractional_targets(self, pool):
        y = np.linspace(-2.0, 3.0, 40) + 0.37
        y[::4] = np.nan
        fit = dict(aux_lambda=1.0, aux_target="regression", unlabeled_marker=np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            enc = _encoder(TEMPLATES["mlp"](), **fit).fit(pool, y)
        assert enc.aux_classes_ is None
        assert enc.model_.aux.out_features == 1
        assert np.isfinite(enc.history_["val_loss"]).all()

        floored = _encoder(TEMPLATES["mlp"](), **fit).fit(pool, np.floor(y))
        assert not np.allclose(enc.transform(pool), floored.transform(pool))

        x = torch.from_numpy(pool[:8])
        labeled = torch.tensor([True, False] * 4)
        with torch.no_grad():
            pred = enc.model_.aux(enc.model_.backbone.forward_features(x)).squeeze(1)

        def aux_loss(target):
            target = torch.as_tensor(target, dtype=torch.float32)
            batch = SSLBatch(x=x, x_raw=x, target=target, labeled=labeled)
            with torch.no_grad():
                return enc._aux_loss(enc.model_, batch).item()

        target = torch.linspace(0.25, 2.5, 8)
        target[~labeled] = float("nan")
        expected = (pred - target)[labeled].pow(2).mean().item()
        assert aux_loss(target) == pytest.approx(expected, rel=1e-5)
        assert aux_loss(target) != pytest.approx(aux_loss(torch.floor(target)))

    def test_numeric_marker_warns_for_regression(self, pool):
        y = np.array([0.25, -1.0, 2.0, 3.5] * 10)
        with pytest.warns(UserWarning, match="unlabeled_marker"):
            _encoder(TEMPLATES["mlp"](), aux_lambda=0.5, aux_target="regression").fit(
                pool, y
            )

    @pytest.mark.parametrize("aux_target", ["classification", "regression"])
    def test_batch_without_labeled_rows_contributes_zero(self, pool, aux_target):
        y = np.repeat([0.0, 1.0], 20)
        enc = _encoder(
            TEMPLATES["mlp"](),
            aux_lambda=2.0,
            aux_target=aux_target,
            unlabeled_marker=np.nan,
        ).fit(pool, y)
        x = torch.from_numpy(pool[:8])
        if aux_target == "classification":
            target = torch.full((8,), -1, dtype=torch.int64)
        else:
            target = torch.full((8,), float("nan"))
        batch = SSLBatch(
            x=x, x_raw=x, target=target, labeled=torch.zeros(8, dtype=bool)
        )
        with torch.no_grad():
            assert enc._aux_loss(enc.model_, batch).item() == 0.0
            total = enc._ssl_loss(enc.model_, batch, torch.Generator().manual_seed(0))
            pretext = enc._pretext_loss(
                enc.model_, batch, torch.Generator().manual_seed(0)
            )
        assert torch.isfinite(total) and total.item() == pretext.item()

    def test_no_labeled_rows_warns_and_disables_classification_head(self, pool):
        with pytest.warns(UserWarning, match="no labeled row"):
            enc = _encoder(TEMPLATES["mlp"](), aux_lambda=1.0).fit(pool)
        assert enc.model_.aux is None
        assert len(enc.aux_classes_) == 0


class DenoisePretext(Pretext):
    """Custom pre-text task: reconstruct the model input from a noisy copy."""

    def __init__(self, noise_std=0.1, mask_ratio=0.15):
        self.noise_std = noise_std
        self.mask_ratio = mask_ratio

    def check_params(self, encoder=None):
        if float(self.noise_std) <= 0:
            raise ValueError("noise_std must be > 0.")

    def build_head(self, feature_dim, input_dim):
        return nn.Linear(feature_dim, input_dim)

    def loss(self, model, batch, generator):
        noise = torch.randn(batch.x.shape, generator=generator).to(batch.x.device)
        recon = model.pretext(model.forward_features(batch.x + self.noise_std * noise))
        return torch.nn.functional.mse_loss(recon.float(), batch.x.float())


# Pre-text losses recorded with the if/elif dispatch of ``encoder.py``
# before the Pretext classes replaced it (torch 2.7.1+cu128, x86_64,
# ``torch.set_num_threads(2)``): per case ``(train_loss epoch 0, 1,
# val_loss epoch 0, 1, loss of the first validation batch with a
# generator seeded 7)``. ``custom`` sets every encoder-level pre-text
# parameter away from its default. The train losses were re-recorded
# when their per-epoch sum moved from ``float32`` to ``float64`` (same
# per-batch losses; the validation losses are unchanged).
REFERENCE_PLATFORM = ("2.7.1+cu128", "x86_64")
REFERENCE_PARAMS = {
    "default": {},
    "custom": {
        "mask_ratio": 0.3,
        "shuffle_fraction": 0.75,
        "projection_dim": 16,
        "temperature": 0.5,
    },
}
REFERENCE_LOSSES = {
    "mlp|default|mae": (2.9221932093302407, 2.70312770207723, 2.792356014251709, 2.783137798309326, 4.241656303405762),
    "mlp|default|shuffle": (0.7086911996205648, 0.7373911539713541, 0.710363507270813, 0.7110941410064697, 0.6963560581207275),
    "mlp|default|shuffle+mae": (3.586195786794027, 3.052543799082438, 3.0902135372161865, 3.1168456077575684, 1.8531041145324707),
    "mlp|default|contrastive": (3.187251567840576, 2.9918153285980225, 1.971604585647583, 1.9709935188293457, 1.9324147701263428),
    "mlp|custom|mae": (3.354564984639486, 2.2753304640452066, 3.6838009357452393, 3.683271646499634, 4.0935869216918945),
    "mlp|custom|shuffle": (0.7219944397608439, 0.7036747535069784, 0.6943145394325256, 0.6945565342903137, 0.6842656135559082),
    "mlp|custom|shuffle+mae": (3.6421054204305015, 3.9737114906311035, 3.5605196952819824, 3.6128852367401123, 4.0464653968811035),
    "mlp|custom|contrastive": (3.0837841828664145, 2.988962729771932, 1.9484832286834717, 1.9464670419692993, 1.94083833694458),
    "cnn|default|mae": (1.176398475964864, 1.3782188892364502, 1.0986566543579102, 1.0944385528564453, 1.128063440322876),
    "cnn|default|shuffle": (0.7256627082824707, 0.7047284245491028, 0.7260544896125793, 0.7238208055496216, 0.6956521272659302),
    "cnn|default|shuffle+mae": (2.090381463368734, 1.9033466180165608, 2.718674421310425, 2.6433217525482178, 1.9658641815185547),
    "cnn|default|contrastive": (3.5056901772816977, 3.1798541943232217, 1.8327864408493042, 1.860303521156311, 1.74770188331604),
    "cnn|custom|mae": (1.2418821255366008, 1.296053369839986, 0.9911344647407532, 0.9736555218696594, 1.2990702390670776),
    "cnn|custom|shuffle": (0.7131132483482361, 0.7046976288159689, 0.7079946994781494, 0.6975609660148621, 0.6860175728797913),
    "cnn|custom|shuffle+mae": (2.0620981454849243, 2.0210872888565063, 1.6251142024993896, 1.603177547454834, 1.8751463890075684),
    "cnn|custom|contrastive": (2.972382148106893, 2.9625981648763022, 1.9290817975997925, 1.9314651489257812, 1.9258726835250854),
}  # fmt: skip
REFERENCE_AUX = (4.0405698617299395, 3.720048189163208, 3.455843687057495, 3.4872000217437744)  # fmt: skip
PRETEXT_CLASSES = dict(
    zip(
        PRETEXTS,
        (MAEPretext, ShufflePretext, ShuffleMAEPretext, ContrastivePretext),
        strict=True,
    )
)


def _reference_template(name):
    if name == "cnn":
        return TEMPLATES["cnn"](input_transform="log1p+standardize")
    return TEMPLATES["mlp"]()


def _losses(enc):
    return enc.history_["train_loss"] + enc.history_["val_loss"]


def _first_val_batch(enc, pool):
    _, val, _ = make_pretraining_loaders(pool, batch_size=16)
    return prepare_batch(
        next(iter(val)), enc.input_transform_state_, torch.device("cpu")
    )


def _assert_reference(actual, expected):
    if (str(torch.__version__), platform.machine()) != REFERENCE_PLATFORM:
        pytest.skip(f"reference losses recorded on {REFERENCE_PLATFORM}")
    assert tuple(actual) == tuple(expected)


def _legacy_pretext_loss(enc, model, batch, gen):
    """The if/elif pre-text dispatch of ``encoder.py`` before the Pretext classes."""

    def mae(head):
        mask = random_bin_mask(
            *batch.x.shape, float(enc.mask_ratio), generator=gen, device=batch.x.device
        )
        return mae_loss(model.backbone, head, batch.x, mask)

    def shuffle(head):
        corrupt = shuffle_corruption(
            batch.x,
            batch.x_raw,
            float(enc.mask_ratio),
            float(enc.shuffle_fraction),
            generator=gen,
        )
        return shuffle_loss(model.backbone, head, *corrupt)

    def transform_raw(raw):
        x = apply_input_transform(raw.numpy(), enc.input_transform_state_)
        return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))

    if enc.pretext == "mae":
        return mae(model.pretext)
    if enc.pretext == "shuffle":
        return shuffle(model.pretext)
    if enc.pretext == "shuffle+mae":
        return shuffle(model.pretext["shuffle"]) + mae(model.pretext["mae"])
    view_a, view_b = contrastive_views(
        batch.x_raw,
        transform_raw,
        float(enc.mask_ratio),
        generator=gen,
        device=batch.x.device,
    )
    return contrastive_loss(
        model.backbone, model.pretext, view_a, view_b, float(enc.temperature)
    )


@pytest.fixture
def clean_registry():
    saved = dict(pretext_module._REGISTRY)
    yield
    pretext_module._REGISTRY.clear()
    pretext_module._REGISTRY.update(saved)


class TestPretextRefactorIsBitwise:
    @pytest.fixture(autouse=True)
    def _raw_scale_mae_target(self, monkeypatch):
        # The references predate the standardized reconstruction target of
        # the raw-scale modes (the "mlp" template resolves to "none"): with
        # it switched off every loss must still match bitwise.
        monkeypatch.setattr(pretraining_data, "_RECON_STATS_MODES", ())

    @IGNORE_SMALL_BATCH
    @pytest.mark.parametrize("pretext", PRETEXTS)
    @pytest.mark.parametrize("params", list(REFERENCE_PARAMS))
    @pytest.mark.parametrize("backbone", ["mlp", "cnn"])
    def test_losses_match_pre_refactor_reference(self, pool, backbone, params, pretext):
        enc = _encoder(
            _reference_template(backbone), pretext=pretext, **REFERENCE_PARAMS[params]
        ).fit(pool)
        batch = _first_val_batch(enc, pool)
        with torch.no_grad():
            gen = torch.Generator().manual_seed(7)
            batch_loss = enc._ssl_loss(enc.model_, batch, gen).item()
        _assert_reference(
            [*_losses(enc), batch_loss],
            REFERENCE_LOSSES[f"{backbone}|{params}|{pretext}"],
        )

    def test_aux_losses_match_pre_refactor_reference(self, pool):
        y = np.array([0, 1, -1, -1] * 10)
        enc = _encoder(TEMPLATES["mlp"](), pretext="shuffle+mae", aux_lambda=0.5).fit(
            pool, y
        )
        _assert_reference(_losses(enc), REFERENCE_AUX)

    @IGNORE_SMALL_BATCH
    @pytest.mark.parametrize("pretext", PRETEXTS)
    @pytest.mark.parametrize("params", list(REFERENCE_PARAMS))
    def test_delegation_equals_legacy_dispatch(self, pool, params, pretext):
        enc = _encoder(
            _reference_template("cnn"),
            pretext=pretext,
            epochs=1,
            **REFERENCE_PARAMS[params],
        ).fit(pool)
        batch = _first_val_batch(enc, pool)
        with torch.no_grad():
            new = enc._pretext_loss(enc.model_, batch, torch.Generator().manual_seed(3))
            old = _legacy_pretext_loss(
                enc, enc.model_, batch, torch.Generator().manual_seed(3)
            )
        assert new.item() == old.item()

    @IGNORE_SMALL_BATCH
    @pytest.mark.parametrize("pretext", PRETEXTS)
    def test_name_and_instance_train_identically(self, pool, pretext):
        params = REFERENCE_PARAMS["custom"]
        cls = PRETEXT_CLASSES[pretext]
        own = {k: v for k, v in params.items() if k in cls._get_param_names()}
        by_name = _encoder(TEMPLATES["mlp"](), pretext=pretext, **params).fit(pool)
        by_instance = _encoder(TEMPLATES["mlp"](), pretext=cls(**own)).fit(pool)
        assert _losses(by_name) == _losses(by_instance)
        assert same_config(by_name.pretext_, by_instance.pretext_)
        state = by_instance.model_.state_dict()
        for key, value in by_name.model_.state_dict().items():
            assert torch.equal(value, state[key]), key


class TestPluggablePretexts:
    def test_name_resolves_to_class_with_encoder_params(self, pool):
        enc = _encoder("mlp", pretext="shuffle", mask_ratio=0.3, epochs=0).fit(pool)
        assert type(enc.pretext_) is ShufflePretext
        assert enc.pretext_.get_params() == {"mask_ratio": 0.3, "shuffle_fraction": 0.5}
        assert enc.pretext == "shuffle"
        assert get_pretext_class("contrastive") is ContrastivePretext

    def test_instance_is_cloned_and_params_are_nested(self, pool):
        pretext = MAEPretext(mask_ratio=0.3)
        enc = _encoder(TEMPLATES["mlp"](), pretext=pretext)
        params = enc.get_params()
        assert params["pretext__mask_ratio"] == 0.3
        enc.set_params(pretext__mask_ratio=0.25)
        assert pretext.mask_ratio == 0.25
        twin = clone(enc)
        assert twin.pretext is not pretext and twin.pretext.mask_ratio == 0.25
        enc.fit(pool)
        assert enc.pretext_ is not pretext and enc.pretext_.mask_ratio == 0.25

    def test_grid_search_over_nested_pretext_param(self, pool):
        y = np.repeat([0, 1], 20)
        pipe = Pipeline(
            [
                (
                    "enc",
                    _encoder(TEMPLATES["mlp"](), pretext=ShufflePretext(), epochs=1),
                ),
                ("lr", LogisticRegression()),
            ]
        )
        search = GridSearchCV(
            pipe, {"enc__pretext__shuffle_fraction": [0.25, 1.0]}, cv=2
        ).fit(pool, y)
        assert [
            p["enc__pretext__shuffle_fraction"] for p in search.cv_results_["params"]
        ] == [0.25, 1.0]
        best = search.best_estimator_.named_steps["enc"]
        assert best.pretext_.shuffle_fraction in (0.25, 1.0)
        assert pipe.named_steps["enc"].pretext.shuffle_fraction == 0.5

    def test_instance_warns_about_changed_encoder_params(self, pool):
        with pytest.warns(UserWarning, match="mask_ratio, temperature are ignored"):
            enc = _encoder(
                "mlp",
                pretext=MAEPretext(mask_ratio=0.3),
                mask_ratio=0.5,
                temperature=0.2,
                epochs=0,
            ).fit(pool)
        assert enc.pretext_.mask_ratio == 0.3
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            _encoder("mlp", pretext=MAEPretext(), epochs=0).fit(pool)

    @pytest.mark.parametrize(
        ("pretext", "error", "match"),
        [
            (MAEPretext(mask_ratio=0.0), ValueError, "mask_ratio"),
            (ShufflePretext(shuffle_fraction=2.0), ValueError, "shuffle_fraction"),
            (ContrastivePretext(projection_dim=0), ValueError, "projection_dim"),
            (ContrastivePretext(temperature=-1.0), ValueError, "temperature"),
            (ContrastivePretext(augment="jitter"), TypeError, "SpectrumAugment"),
            (DenoisePretext(noise_std=0.0), ValueError, "noise_std"),
            (3, TypeError, "Pretext instance"),
        ],
    )
    def test_invalid_pretext_raises(self, pool, pretext, error, match):
        with pytest.raises(error, match=match):
            _encoder("mlp", pretext=pretext, batch_size=64, epochs=0).fit(pool)

    def test_contrastive_instance_warns_on_small_batch(self, pool):
        with pytest.warns(UserWarning, match="batch_size=16"):
            _encoder("mlp", pretext=ContrastivePretext(), epochs=0).fit(pool)

    def test_abstract_methods_are_required(self):
        class Incomplete(Pretext):
            def build_head(self, feature_dim, input_dim):
                return nn.Identity()

        with pytest.raises(TypeError, match="abstract"):
            Incomplete()

    def test_custom_pretext_fit_save_load(self, pool, tmp_path):
        enc = _encoder(TEMPLATES["cnn"](), pretext=DenoisePretext(noise_std=0.3)).fit(
            pool
        )
        assert np.isfinite(enc.history_["val_loss"]).all()
        assert enc.model_.pretext.out_features == INPUT_DIM
        enc.save(tmp_path / "enc")
        meta = json.loads((tmp_path / "enc.json").read_text())
        assert meta["params"]["pretext"] == {
            "__pretext__": "tests.test_pretraining.DenoisePretext",
            "params": {"mask_ratio": 0.15, "noise_std": 0.3},
        }
        loaded = MaldiSpectrumEncoder.load(tmp_path / "enc")
        assert type(loaded.pretext) is DenoisePretext
        assert loaded.pretext.noise_std == 0.3
        assert same_config(loaded.pretext_, enc.pretext_)
        np.testing.assert_allclose(
            loaded.transform(pool), enc.transform(pool), atol=1e-6, rtol=0
        )
        state = loaded.model_.state_dict()
        for key, value in enc.model_.state_dict().items():
            assert torch.equal(value, state[key]), key

    @IGNORE_SMALL_BATCH
    def test_builtin_instance_round_trips(self, pool, tmp_path):
        pretext = ContrastivePretext(
            temperature=0.5, augment=SpectrumAugment(noise_std=0.1, blur_sigma=1.0)
        )
        enc = _encoder(TEMPLATES["mlp"](), pretext=pretext, epochs=1).fit(pool)
        enc.save(tmp_path / "enc")
        loaded = MaldiSpectrumEncoder.load(tmp_path / "enc")
        assert type(loaded.pretext) is ContrastivePretext
        assert loaded.pretext.temperature == 0.5
        assert repr(loaded.pretext.augment) == repr(pretext.augment)
        batch = _first_val_batch(enc, pool)
        with torch.no_grad():
            losses = [
                e._ssl_loss(e.model_, batch, torch.Generator().manual_seed(0)).item()
                for e in (enc, loaded)
            ]
        assert losses[0] == losses[1]

    def test_unimportable_custom_pretext_fails_at_save(self, pool, tmp_path):
        class LocalPretext(DenoisePretext):
            pass

        enc = _encoder("mlp", pretext=LocalPretext(), epochs=0).fit(pool)
        with pytest.raises(ValueError, match="pretext class .* cannot be re-imported"):
            enc.save(tmp_path / "enc")
        assert not list(tmp_path.iterdir())

    def test_unimportable_custom_pretext_fails_at_load(self, pool, tmp_path):
        enc = _encoder("mlp", pretext=DenoisePretext(), epochs=0).fit(pool)
        enc.save(tmp_path / "enc")
        meta = json.loads((tmp_path / "enc.json").read_text())
        meta["params"]["pretext"]["__pretext__"] += "Missing"
        (tmp_path / "enc.json").write_text(json.dumps(meta))
        with pytest.raises(ValueError, match="Cannot import the pretext class"):
            MaldiSpectrumEncoder.load(tmp_path / "enc")

    def test_unserialisable_pretext_param_fails_before_writing(self, pool, tmp_path):
        enc = _encoder("mlp", pretext=DenoisePretext(noise_std=object()), epochs=0)
        enc.pretext_ = enc.pretext
        enc.model_ = object()
        with pytest.raises(TypeError, match="pretext__noise_std"):
            enc.save(tmp_path / "enc")
        assert not list(tmp_path.iterdir())

    @pytest.mark.usefixtures("clean_registry")
    def test_registered_name(self, pool, tmp_path):
        assert register_pretext("denoise", DenoisePretext) is DenoisePretext
        enc = _encoder(
            "mlp", pretext="denoise", mask_ratio=0.3, temperature=0.2, epochs=1
        ).fit(pool)
        # Only the encoder-level parameters the class accepts are passed.
        assert enc.pretext_.get_params() == {"mask_ratio": 0.3, "noise_std": 0.1}
        enc.save(tmp_path / "enc")
        assert json.loads((tmp_path / "enc.json").read_text())["params"]["pretext"] == (
            "denoise"
        )
        loaded = MaldiSpectrumEncoder.load(tmp_path / "enc")
        np.testing.assert_allclose(
            loaded.transform(pool), enc.transform(pool), atol=1e-6, rtol=0
        )
        del pretext_module._REGISTRY["denoise"]
        with pytest.raises(ValueError, match="Unknown pretext='denoise'"):
            MaldiSpectrumEncoder.load(tmp_path / "enc")

    @pytest.mark.usefixtures("clean_registry")
    def test_register_pretext_validation(self):
        with pytest.raises(ValueError, match="built-in"):
            register_pretext("mae", DenoisePretext)
        with pytest.raises(TypeError, match="Pretext subclass"):
            register_pretext("denoise", MAEHead)
        with pytest.raises(ValueError, match="non-empty"):
            register_pretext("", DenoisePretext)
        assert get_pretext_class("mae") is MAEPretext


class TestContrastiveAugment:
    @IGNORE_SMALL_BATCH
    def test_none_is_the_contrastive_default(self, pool):
        default = _encoder(TEMPLATES["mlp"](), pretext="contrastive").fit(pool)
        explicit = _encoder(
            TEMPLATES["mlp"](),
            pretext="contrastive",
            augment=SpectrumAugment(**CONTRASTIVE_AUGMENT),
        ).fit(pool)
        assert _losses(default) == _losses(explicit)

    @IGNORE_SMALL_BATCH
    def test_user_augment_is_used_and_never_modified(self, pool, tmp_path):
        augment = SpectrumAugment(noise_std=0.3, peak_dropout_rate=0.1)
        enc = _encoder(TEMPLATES["mlp"](), pretext="contrastive", augment=augment)
        enc.fit(pool)
        default = _encoder(TEMPLATES["mlp"](), pretext="contrastive").fit(pool)
        assert _losses(enc) != _losses(default)
        assert enc.pretext_.augment is augment
        assert augment.random_state is None
        assert augment._generator is None and augment._np_rng is None

        enc.save(tmp_path / "enc")
        loaded = MaldiSpectrumEncoder.load(tmp_path / "enc")
        assert repr(loaded.augment) == repr(augment)
        assert repr(loaded.pretext_.augment) == repr(augment)

    @IGNORE_SMALL_BATCH
    def test_user_augment_loss_depends_only_on_the_generator(self, pool):
        augment = SpectrumAugment(
            noise_std=0.1,
            peak_dropout_rate=0.1,
            mz_warp_max_bins=2,
            mz_warp_n_knots=3,
            random_state=5,
        )
        enc = _encoder(
            TEMPLATES["cnn"](),
            pretext=ContrastivePretext(augment=augment),
            epochs=1,
        ).fit(pool)
        batch = _first_val_batch(enc, pool)

        def loss(seed, global_seed):
            torch.manual_seed(global_seed)
            np.random.seed(global_seed)
            with torch.no_grad():
                gen = torch.Generator().manual_seed(seed)
                return enc._ssl_loss(enc.model_, batch, gen).item()

        assert loss(1, 0) == loss(1, 123) == loss(1, 0)
        assert loss(1, 0) != loss(2, 0)

    def test_validation_loss_is_identical_across_epochs(self, pool):
        """Re-seeded copies: same views every validation pass (weights never updated)."""
        torch.manual_seed(0)
        backbone = nn.Sequential(nn.Linear(INPUT_DIM, 8), nn.ReLU(), nn.Linear(8, 2))
        pretext = ContrastivePretext(
            projection_dim=4,
            augment=SpectrumAugment(noise_std=0.2, peak_dropout_rate=0.1),
        )
        model = SSLModel(backbone, pretext.build_head(8, INPUT_DIM))
        tr, val, stats = make_pretraining_loaders(pool, batch_size=8)
        _, history = ssl_train_loop(
            model,
            tr,
            val,
            pretext.loss,
            # The optimizer holds an unrelated parameter: the model never moves.
            torch.optim.SGD([nn.Parameter(torch.zeros(1))], lr=0.0),
            torch.device("cpu"),
            3,
            EarlyStopping(patience=5),
            input_transform_state=stats["input_transform_state"],
        )
        assert len(history["val_loss"]) == 3
        assert len(set(history["val_loss"])) == 1
        assert len(set(history["train_loss"])) == 3

    def test_augment_with_other_pretext_warns(self, pool):
        with pytest.warns(UserWarning, match="augment is only used by"):
            _encoder("mlp", augment=SpectrumAugment(noise_std=0.1), epochs=0).fit(pool)
        with pytest.warns(UserWarning, match="augment is ignored"):
            _encoder(
                "mlp",
                pretext=ContrastivePretext(),
                augment=SpectrumAugment(noise_std=0.1),
                batch_size=64,
                epochs=0,
            ).fit(pool)

    def test_invalid_augment_raises(self, pool):
        with pytest.raises(TypeError, match="SpectrumAugment"):
            _encoder(
                "mlp", pretext="contrastive", augment="jitter", batch_size=64, epochs=0
            ).fit(pool)

    def test_contrastive_needs_the_batch_transform(self, pool):
        enc = _encoder("mlp", pretext="contrastive", batch_size=64, epochs=0).fit(pool)
        x = torch.from_numpy(pool[:8])
        batch = SSLBatch(
            x=x, x_raw=x, target=torch.zeros(8), labeled=torch.zeros(8, dtype=bool)
        )
        with pytest.raises(ValueError, match="batch.transform"):
            enc._pretext_loss(enc.model_, batch, torch.Generator().manual_seed(0))


def _tic_pool(n_rows=48, n_bins=INPUT_DIM, seed=0):
    """Total-ion-current normalised spectra (bins of order 1 / n_bins, as DRIAMS)."""
    rng = np.random.default_rng(seed)
    X = rng.gamma(0.5, 1.0, (n_rows, n_bins)).astype(np.float32)
    X[:, [5, 20, 40]] += 20.0
    return X / X.sum(axis=1, keepdims=True)


class TestReconstructionTarget:
    @pytest.mark.parametrize("mode", INPUT_TRANSFORMS)
    def test_loader_stats_of_transformed_train_rows(self, pool, mode):
        _, _, stats = make_pretraining_loaders(pool, batch_size=8, transform_mode=mode)
        recon = stats["recon_stats"]
        if mode not in ("log1p", "none"):
            assert recon is None
            return
        x = apply_input_transform(
            pool[stats["train_indices"]], stats["input_transform_state"]
        ).astype(np.float64)
        np.testing.assert_allclose(recon["mean"], x.mean(axis=0), rtol=1e-6)
        np.testing.assert_allclose(recon["std"], x.std(axis=0), rtol=1e-5)
        assert recon["mean"].dtype == recon["std"].dtype == np.float32

    def test_chunked_stats_equal_one_pass(self, pool, monkeypatch):
        _, _, whole = make_pretraining_loaders(
            pool, batch_size=8, transform_mode="log1p"
        )
        monkeypatch.setattr(pretraining_data, "_RECON_CHUNK_BYTES", 3 * 4 * INPUT_DIM)
        _, _, chunked = make_pretraining_loaders(
            pool, batch_size=8, transform_mode="log1p"
        )
        for key in ("mean", "std"):
            np.testing.assert_allclose(
                chunked["recon_stats"][key], whole["recon_stats"][key], rtol=1e-6
            )

    def test_batch_recon_target(self):
        x = torch.tensor([[1.0, 2.0], [3.0, 6.0]])
        batch = SSLBatch(
            x=x, x_raw=x, target=torch.zeros(2), labeled=torch.zeros(2, dtype=bool)
        )
        assert batch.recon_target() is x
        stats = (torch.tensor([2.0, 4.0]), torch.tensor([1.0, 2.0]))
        torch.testing.assert_close(
            batch._replace(recon_stats=stats).recon_target(),
            torch.tensor([[-1.0, -1.0], [1.0, 1.0]]),
        )

    def test_fitted_transform_hands_floored_stats_once(self):
        recon = {
            "mean": np.zeros(2, dtype=np.float32),
            "std": np.array([0.0, 2.0], dtype=np.float32),
        }
        transform = _FittedTransform({"mode": "none"}, recon)
        mean, std = transform.recon_stats_on(torch.device("cpu"))
        assert std.tolist() == pytest.approx([1.0, 2.0])
        assert transform.recon_stats_on(torch.device("cpu"))[1] is std
        assert _FittedTransform({"mode": "none"}).recon_stats_on("cpu") is None
        raw = torch.ones(3, 2)
        batch = prepare_batch(
            (raw, torch.zeros(3), torch.zeros(3, dtype=bool)), transform, "cpu"
        )
        assert batch.recon_stats[1] is std

    def test_mae_batch_loss_reconstructs_recon_target(self, pool):
        enc = _encoder(TEMPLATES["mlp"](), epochs=0).fit(pool)
        head = MAEHead(enc.feature_dim_, INPUT_DIM)
        x = torch.from_numpy(pool[:8])
        stats = (torch.full((INPUT_DIM,), 0.5), torch.full((INPUT_DIM,), 2.0))
        batch = SSLBatch(
            x=x,
            x_raw=x,
            target=torch.zeros(8),
            labeled=torch.zeros(8, dtype=bool),
            recon_stats=stats,
        )
        with torch.no_grad():
            got = _mae_batch_loss(
                enc.model_, head, batch, 0.15, torch.Generator().manual_seed(1)
            )
            mask = random_bin_mask(
                8,
                INPUT_DIM,
                0.15,
                generator=torch.Generator().manual_seed(1),
                device=torch.device("cpu"),
            )
            want = mae_loss(enc.model_, head, x, mask, (x - 0.5) / 2.0)
        assert got.item() == want.item()

    @pytest.mark.parametrize("pretext", ["mae", "shuffle+mae"])
    def test_raw_scale_mae_loss_is_standardized(self, monkeypatch, pretext):
        X = _tic_pool()
        enc = _encoder(TEMPLATES["resnet"](), pretext=pretext).fit(X)
        assert enc.input_transform_state_["mode"] == "log1p"
        assert enc.recon_mean_.shape == enc.recon_std_.shape == (INPUT_DIM,)
        # Standardized target: a loss of order one (about 1.5 here). On the
        # raw scale it stays at the decoder's initial noise (about 0.05 here;
        # on 6000-bin spectra the target itself is of order 1e-4).
        assert min(enc.history_["val_loss"]) > 0.5
        if pretext == "mae":
            monkeypatch.setattr(pretraining_data, "_RECON_STATS_MODES", ())
            raw = _encoder(TEMPLATES["resnet"](), pretext=pretext).fit(X)
            assert raw.recon_mean_ is None
            assert max(raw.history_["val_loss"]) < 0.2

    def test_standardizing_mode_has_no_recon_stats(self, pool):
        enc = _encoder(TEMPLATES["cnn"](input_transform="standardize"), epochs=1).fit(
            pool
        )
        assert enc.recon_mean_ is None and enc.recon_std_ is None

    def test_save_load_keeps_recon_stats(self, tmp_path):
        enc = _encoder(TEMPLATES["resnet"](), epochs=1).fit(_tic_pool())
        enc.save(tmp_path / "enc")
        loaded = MaldiSpectrumEncoder.load(tmp_path / "enc")
        np.testing.assert_array_equal(loaded.recon_mean_, enc.recon_mean_)
        np.testing.assert_array_equal(loaded.recon_std_, enc.recon_std_)
        meta = json.loads((tmp_path / "enc.json").read_text())
        del meta["fitted"]["recon_mean_"], meta["fitted"]["recon_std_"]
        (tmp_path / "enc.json").write_text(json.dumps(meta))
        older = MaldiSpectrumEncoder.load(tmp_path / "enc")
        assert older.recon_mean_ is None and older.recon_std_ is None
