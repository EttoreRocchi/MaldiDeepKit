"""Tests for :class:`MaldiDualBranchClassifier` and the dual base."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from maldideepkit import (
    BaseDualSpectralClassifier,
    MaldiCNNClassifier,
    MaldiDualBranchClassifier,
    MaldiMLPClassifier,
    smiles_to_fingerprint,
)


@pytest.fixture
def dual_data(rng):
    n = 60
    n_spec = 24
    n_side = 16
    X_spec = rng.standard_normal((n, n_spec)).astype("float32")
    X_side = rng.standard_normal((n, n_side)).astype("float32")
    X = np.hstack([X_spec, X_side])
    y = rng.integers(0, 2, size=n).astype(np.int64)
    return {"X": X, "y": y, "n_spec": n_spec, "n_side": n_side}


def test_base_dual_rejects_invalid_n_spec_bins():
    with pytest.raises(ValueError, match="n_spec_bins"):

        class _M(BaseDualSpectralClassifier):
            def _build_dual_model(self, **kw):
                raise NotImplementedError

        _M(n_spec_bins=0)


def test_base_dual_is_abstract():
    with pytest.raises(TypeError, match="abstract"):
        BaseDualSpectralClassifier(n_spec_bins=4)  # type: ignore[abstract]


def test_base_dual_splitter_calls_dual_module():
    """The internal _DualInputSplitter must split X at n_spec_bins
    and forward to the user's dual-branch module."""

    class _ToyDualModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.recorded_spec_shape = None
            self.recorded_side_shape = None
            self.head = nn.Linear(8, 2)

        def forward(self, x_spec, x_side):
            self.recorded_spec_shape = tuple(x_spec.shape)
            self.recorded_side_shape = tuple(x_side.shape)
            return self.head(torch.zeros(x_spec.shape[0], 8))

    class _ToyClassifier(BaseDualSpectralClassifier):
        def _build_dual_model(self, *, input_dim_spec, input_dim_side, n_classes):
            return _ToyDualModule()

    clf = _ToyClassifier(n_spec_bins=24)
    clf.input_dim_ = 24 + 16
    clf.n_classes_ = 2
    wrapped = clf._build_model()
    x = torch.zeros(5, 40)
    out = wrapped(x)
    assert out.shape == (5, 2)
    assert wrapped.inner.recorded_spec_shape == (5, 24)
    assert wrapped.inner.recorded_side_shape == (5, 16)


def test_dual_branch_defaults():
    clf = MaldiDualBranchClassifier(n_spec_bins=64)
    assert clf.spec_hidden_dims == (256, 128)
    assert clf.fp_hidden_dims == (256, 128)
    assert clf.fp_dropout == 0.2
    assert clf.backbone is None


def test_dual_branch_rejects_side_dim_zero(dual_data):
    """n_spec_bins == X.shape[1] -> side dim is zero -> error."""
    clf = MaldiDualBranchClassifier(
        n_spec_bins=dual_data["X"].shape[1],  # whole X is spectrum
        spec_hidden_dims=(16,),
        fp_hidden_dims=(8,),
        epochs=1,
        batch_size=8,
    )
    with pytest.raises(ValueError, match="side input"):
        clf.fit(dual_data["X"], dual_data["y"])


def test_dual_branch_fit_predict_runs(dual_data):
    clf = MaldiDualBranchClassifier(
        n_spec_bins=dual_data["n_spec"],
        spec_hidden_dims=(16,),
        spec_dropout=0.0,
        fp_hidden_dims=(8,),
        fp_dropout=0.0,
        epochs=1,
        batch_size=8,
        random_state=0,
    )
    clf.fit(dual_data["X"], dual_data["y"])
    proba = clf.predict_proba(dual_data["X"])
    preds = clf.predict(dual_data["X"])
    assert proba.shape == (dual_data["X"].shape[0], 2)
    assert preds.shape == (dual_data["X"].shape[0],)


def test_dual_branch_sklearn_compatibility(dual_data):
    """The class plays nicely with sklearn.base.clone."""
    from sklearn.base import clone

    clf = MaldiDualBranchClassifier(
        n_spec_bins=dual_data["n_spec"],
        spec_hidden_dims=(16,),
        fp_hidden_dims=(8,),
        epochs=1,
        batch_size=8,
    )
    twin = clone(clf)
    assert twin.n_spec_bins == dual_data["n_spec"]
    assert twin is not clf


@pytest.mark.slow
def test_dual_branch_with_pipeline(dual_data):
    """Pipeline + StandardScaler upstream must not break the class."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    pipe = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "clf",
                MaldiDualBranchClassifier(
                    n_spec_bins=dual_data["n_spec"],
                    spec_hidden_dims=(16,),
                    fp_hidden_dims=(8,),
                    epochs=1,
                    batch_size=8,
                    random_state=0,
                ),
            ),
        ]
    )
    pipe.fit(dual_data["X"], dual_data["y"])
    assert pipe.predict(dual_data["X"]).shape == (dual_data["X"].shape[0],)


def test_dual_branch_with_mlp_backbone(dual_data):
    clf = MaldiDualBranchClassifier(
        n_spec_bins=dual_data["n_spec"],
        backbone=MaldiMLPClassifier(
            hidden_dim=16, head_dims=(8,), epochs=1, random_state=0
        ),
        fp_hidden_dims=(8,),
        epochs=1,
        batch_size=8,
        random_state=0,
    )
    clf.fit(dual_data["X"], dual_data["y"])
    assert clf.predict(dual_data["X"]).shape == (dual_data["X"].shape[0],)


def test_dual_branch_with_cnn_backbone(dual_data):
    clf = MaldiDualBranchClassifier(
        n_spec_bins=dual_data["n_spec"],
        backbone=MaldiCNNClassifier(
            channels=(4, 8),
            kernel_size=3,
            pool_size=2,
            head_dim=16,
            epochs=1,
            random_state=0,
        ),
        fp_hidden_dims=(8,),
        epochs=1,
        batch_size=8,
        random_state=0,
    )
    clf.fit(dual_data["X"], dual_data["y"])
    assert clf.predict(dual_data["X"]).shape == (dual_data["X"].shape[0],)


def test_dual_branch_rejects_invalid_backbone():
    clf = MaldiDualBranchClassifier(
        n_spec_bins=24,
        backbone="not-a-classifier",
        epochs=1,
        batch_size=8,
    )
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 40)).astype(np.float32)
    y = rng.integers(0, 2, size=40).astype(np.int64)
    with pytest.raises(TypeError, match="BaseSpectralClassifier"):
        clf.fit(X, y)


class TestSideBlockIsNotPreprocessedAsSpectrum:
    """Spectrum preprocessing must stop at the ``n_spec_bins`` boundary."""

    @pytest.fixture
    def packed(self, rng):
        n_spec, n_side = 24, 16
        X_spec = np.abs(rng.standard_normal((60, n_spec))).astype("float32") * 100
        X_side = rng.integers(0, 2, size=(60, n_side)).astype("float32")
        y = rng.integers(0, 2, size=60).astype(np.int64)
        return np.hstack([X_spec, X_side]), y, n_spec, X_side

    def _fit(self, X, y, n_spec, **kwargs):
        return MaldiDualBranchClassifier(
            n_spec_bins=n_spec,
            spec_hidden_dims=(16,),
            fp_hidden_dims=(8,),
            epochs=1,
            batch_size=8,
            random_state=0,
            **kwargs,
        ).fit(X, y)

    def test_transform_stats_cover_spectrum_only(self, packed):
        X, y, n_spec, _ = packed
        clf = self._fit(X, y, n_spec, input_transform="log1p+standardize")
        assert clf.feature_mean_.shape == (n_spec,)
        assert clf.feature_std_.shape == (n_spec,)

    def test_side_block_passes_through_untouched(self, packed):
        from maldideepkit.base.data import (
            _transform_leading_cols,
            apply_input_transform,
        )

        X, y, n_spec, X_side = packed
        clf = self._fit(X, y, n_spec, input_transform="log1p+standardize")
        out = _transform_leading_cols(
            X,
            clf._n_transform_cols(),
            lambda A: apply_input_transform(A, clf.input_transform_state_),
        )
        np.testing.assert_array_equal(out[:, n_spec:], X_side)
        assert not np.allclose(out[:, :n_spec], X[:, :n_spec])

    def test_single_input_models_still_transform_every_column(self, packed):
        X, y, _, _ = packed
        clf = MaldiMLPClassifier(
            hidden_dim=16,
            head_dims=(8,),
            epochs=1,
            batch_size=8,
            random_state=0,
            input_transform="standardize",
        ).fit(X, y)
        assert clf._n_transform_cols() is None
        assert clf.feature_mean_.shape == (X.shape[1],)

    def test_transform_rejects_width_changing_step(self):
        from maldideepkit.base.data import _transform_leading_cols

        X = np.zeros((4, 10), dtype=np.float32)
        with pytest.raises(ValueError, match="changed the spectrum block width"):
            _transform_leading_cols(X, 6, lambda A: A[:, :3])

    def test_augment_sees_only_the_spectrum_block(self, packed):
        X, y, n_spec, _ = packed
        widths: list[int] = []

        class _WidthSpy:
            def __call__(self, xb):
                widths.append(int(xb.shape[1]))
                return xb

        self._fit(X, y, n_spec, augment=_WidthSpy())
        assert widths and set(widths) == {n_spec}

    def test_fit_predict_runs_with_every_confined_knob(self, packed):
        X, y, n_spec, _ = packed
        clf = self._fit(
            X, y, n_spec, cutmix_alpha=1.0, mixup_alpha=0.2, standardize=True
        )
        assert clf.predict(X).shape == (X.shape[0],)


class TestCutmixConfinement:
    """CutMix windows must not straddle the spectrum / side boundary."""

    @staticmethod
    def _batch():
        x = torch.arange(6 * 10, dtype=torch.float32).reshape(6, 10)
        y_oh = torch.eye(2)[torch.tensor([0, 1, 0, 1, 0, 1])]
        return x, y_oh

    def test_window_stays_inside_the_spectrum_block(self):
        from maldideepkit.augment.mixing import apply_cutmix

        x, y_oh = self._batch()
        gen = torch.Generator().manual_seed(0)
        spec_values = set(x[:, :6].flatten().tolist())
        for _ in range(50):
            x_mixed, _ = apply_cutmix(x, y_oh, 1.0, generator=gen, n_spec_cols=6)
            # Every spectrum-side value still comes from some row's spectrum.
            assert set(x_mixed[:, :6].flatten().tolist()) <= spec_values

    def test_side_block_is_blended_not_spliced(self):
        from maldideepkit.augment.mixing import apply_cutmix

        x, y_oh = self._batch()
        gen = torch.Generator().manual_seed(0)
        blended = 0
        for _ in range(50):
            x_mixed, _ = apply_cutmix(x, y_oh, 1.0, generator=gen, n_spec_cols=6)
            if not torch.isin(x_mixed[:, 6:], x[:, 6:]).all():
                blended += 1
        assert blended > 0

    def test_unconfined_call_is_unchanged(self):
        from maldideepkit.augment.mixing import apply_cutmix

        x, y_oh = self._batch()
        a = apply_cutmix(x, y_oh, 1.0, generator=torch.Generator().manual_seed(3))
        b = apply_cutmix(
            x, y_oh, 1.0, generator=torch.Generator().manual_seed(3), n_spec_cols=None
        )
        torch.testing.assert_close(a[0], b[0])
        torch.testing.assert_close(a[1], b[1])

    def test_rejects_non_positive_n_spec_cols(self):
        from maldideepkit.augment.mixing import apply_cutmix

        x, y_oh = self._batch()
        with pytest.raises(ValueError, match="n_spec_cols"):
            apply_cutmix(x, y_oh, 1.0, n_spec_cols=0)


def test_smiles_to_fingerprint_behavior():
    import importlib.util

    if importlib.util.find_spec("rdkit") is None:
        with pytest.raises(ImportError, match="RDKit"):
            smiles_to_fingerprint(["CCO"])
    else:
        fp = smiles_to_fingerprint(["CCO"], fp_dim=512, radius=2)
        assert fp.shape == (1, 512)
        assert fp.dtype == np.float32


@pytest.mark.skipif(
    __import__("importlib.util", fromlist=["util"]).find_spec("rdkit") is None,
    reason="requires the rdkit extra",
)
class TestSmilesFailuresAreLoud:
    """An unparseable SMILES must not become a silent all-zero row."""

    def test_warns_and_names_the_failing_rows(self):
        with pytest.warns(UserWarning, match=r"could not parse 1 of 3") as record:
            fp = smiles_to_fingerprint(["CCO", "not-a-molecule", "c1ccccc1"], fp_dim=64)
        assert "1 ('not-a-molecule')" in str(record[0].message)
        assert not fp[1].any()
        assert fp[0].any() and fp[2].any()

    def test_truncates_a_long_failure_list(self):
        bad = [f"bad-{i}" for i in range(8)]
        with pytest.warns(UserWarning, match="and 3 more") as record:
            smiles_to_fingerprint(bad, fp_dim=64)
        assert "could not parse 8 of 8" in str(record[0].message)

    def test_stays_quiet_when_every_smiles_parses(self, recwarn):
        smiles_to_fingerprint(["CCO", "c1ccccc1"], fp_dim=64)
        assert [w for w in recwarn if issubclass(w.category, UserWarning)] == []


class TestDualPersistence:
    """save()/load() round-trips for the dual-branch classifier."""

    def _fit(self, dual_data, **kw):
        return MaldiDualBranchClassifier(
            n_spec_bins=dual_data["n_spec"],
            spec_hidden_dims=(16,),
            fp_hidden_dims=(8,),
            epochs=1,
            batch_size=16,
            random_state=0,
            device="cpu",
            **kw,
        ).fit(dual_data["X"], dual_data["y"])

    def test_roundtrip_default_branch(self, dual_data, tmp_path):
        clf = self._fit(dual_data)
        clf.save(tmp_path / "m")
        loaded = MaldiDualBranchClassifier.load(tmp_path / "m")
        np.testing.assert_allclose(
            loaded.predict_proba(dual_data["X"]),
            clf.predict_proba(dual_data["X"]),
            rtol=1e-5,
        )

    def test_roundtrip_with_backbone(self, dual_data, tmp_path):
        clf = self._fit(
            dual_data,
            backbone=MaldiCNNClassifier(channels=(4, 8), head_dim=16),
        )
        clf.save(tmp_path / "m")
        loaded = MaldiDualBranchClassifier.load(tmp_path / "m")
        assert isinstance(loaded.backbone, MaldiCNNClassifier)
        assert tuple(loaded.backbone.channels) == (4, 8)
        np.testing.assert_allclose(
            loaded.predict_proba(dual_data["X"]),
            clf.predict_proba(dual_data["X"]),
            rtol=1e-5,
        )

    def test_generic_base_class_load(self, dual_data, tmp_path):
        from maldideepkit import BaseSpectralClassifier

        clf = self._fit(dual_data)
        clf.save(tmp_path / "m")
        loaded = BaseSpectralClassifier.load(tmp_path / "m")
        assert isinstance(loaded, MaldiDualBranchClassifier)


@pytest.mark.filterwarnings("ignore:.*contains no Dropout layers.*:UserWarning")
class TestUncertaintyOnDualModels:
    """The uncertainty estimators preprocess dual inputs the same way
    predict_proba does (spectrum columns only). Dropout is set to 0.0
    on purpose so the MC passes are deterministic (hence the silenced
    no-Dropout-layers warning)."""

    def _fit(self, dual_data, **kw):
        return MaldiDualBranchClassifier(
            n_spec_bins=dual_data["n_spec"],
            spec_hidden_dims=(16,),
            spec_dropout=0.0,
            fp_hidden_dims=(8,),
            fp_dropout=0.0,
            epochs=1,
            batch_size=16,
            random_state=0,
            device="cpu",
            **kw,
        ).fit(np.abs(dual_data["X"]), dual_data["y"])

    def test_mc_dropout_standardize_does_not_crash(self, dual_data):
        from maldideepkit.uncertainty import MCDropoutEstimator

        clf = self._fit(dual_data, standardize=True)
        result = MCDropoutEstimator(clf, n_samples=3).predict_with_uncertainty(
            np.abs(dual_data["X"])
        )
        assert result.predictions.shape == (dual_data["X"].shape[0],)

    def test_mc_dropout_log1p_matches_predict_proba(self, dual_data):
        from maldideepkit.uncertainty import MCDropoutEstimator

        clf = self._fit(dual_data, input_transform="log1p")
        # With every dropout rate at 0.0 the MC passes are
        # deterministic, so proba_mean must equal predict_proba exactly
        # - it diverges if the side block gets log1p'd on one path only.
        result = MCDropoutEstimator(clf, n_samples=2).predict_with_uncertainty(
            np.abs(dual_data["X"])
        )
        np.testing.assert_allclose(
            result.proba_mean, clf.predict_proba(np.abs(dual_data["X"])), rtol=1e-5
        )


def test_find_lr_respects_dual_split(dual_data):
    from maldideepkit.utils.lr_finder import find_lr

    clf = MaldiDualBranchClassifier(
        n_spec_bins=dual_data["n_spec"],
        spec_hidden_dims=(16,),
        fp_hidden_dims=(8,),
        standardize=True,
        epochs=1,
        batch_size=16,
        random_state=0,
        device="cpu",
    )
    result = find_lr(clf, np.abs(dual_data["X"]), dual_data["y"], num_iter=3)
    assert result["suggested_lr"] > 0
    assert len(clf.feature_mean_) == dual_data["n_spec"]
    assert len(clf.feature_std_) == dual_data["n_spec"]


def test_minimal_dual_subclass_survives_clone(dual_data):
    """A subclass overriding only _build_dual_model must expose the full
    inherited hyperparameter surface to get_params/clone."""
    from sklearn.base import clone

    class _MinimalDual(BaseDualSpectralClassifier):
        def _build_dual_model(self, *, input_dim_spec, input_dim_side, n_classes):
            return _make_linear_dual(input_dim_spec, input_dim_side, n_classes)

    clf = _MinimalDual(n_spec_bins=dual_data["n_spec"], epochs=7, learning_rate=1e-4)
    params = clf.get_params()
    assert "epochs" in params and "learning_rate" in params
    cloned = clone(clf)
    assert cloned.epochs == 7
    assert cloned.learning_rate == pytest.approx(1e-4)
    assert cloned.n_spec_bins == dual_data["n_spec"]


def _make_linear_dual(input_dim_spec, input_dim_side, n_classes):
    class _M(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(input_dim_spec + input_dim_side, n_classes)

        def forward(self, x_spec, x_side):
            return self.lin(torch.cat([x_spec, x_side], dim=1))

    return _M()


@pytest.mark.skipif(
    __import__("importlib.util", fromlist=["util"]).find_spec("rdkit") is None,
    reason="RDKit not installed",
)
def test_smiles_to_fingerprint_treats_non_strings_as_failures():
    """NaN/None cells (a missing pandas entry) must become all-zero
    rows with a warning, not crash inside RDKit's C++ binding."""
    with pytest.warns(UserWarning, match="could not parse 2 of 3"):
        fp = smiles_to_fingerprint(["CCO", float("nan"), None])
    assert fp.shape == (3, 2048)
    assert fp[0].sum() > 0
    assert fp[1].sum() == 0
    assert fp[2].sum() == 0
