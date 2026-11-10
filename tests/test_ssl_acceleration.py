"""Device-resident pre-training data, device-side corruption and ``torch_compile``."""

from __future__ import annotations

import inspect
import pickle
import warnings

import numpy as np
import pytest
import torch
from sklearn.base import clone
from torch import nn
from torch.utils.data import DataLoader, Dataset

from maldideepkit import (
    BaseDualSpectralClassifier,
    BaseSpectralClassifier,
    MaldiCNNClassifier,
    MaldiDualBranchClassifier,
    MaldiMLPClassifier,
    MaldiResNetClassifier,
    MaldiTransformerClassifier,
)
from maldideepkit.augment import SpectrumAugment
from maldideepkit.base import classifier as classifier_module
from maldideepkit.base.data import (
    INPUT_TRANSFORMS,
    apply_input_transform,
    apply_input_transform_torch,
    fit_input_transform,
)
from maldideepkit.pretraining import (
    ContrastivePretext,
    MAEPretext,
    MaldiSpectrumEncoder,
    ShufflePretext,
    SSLModel,
    make_pretraining_loaders,
)
from maldideepkit.pretraining import encoder as encoder_module
from maldideepkit.pretraining._pretext import (
    _sampling_generator,
    contrastive_views,
    random_bin_mask,
    shuffle_corruption,
)
from maldideepkit.pretraining._training import _FittedTransform, ssl_train_loop
from maldideepkit.pretraining.data import stack_collate
from maldideepkit.utils.training import EarlyStopping, resolve_amp_dtype
from tests.conftest import ALL_FACTORIES

HAS_CUDA = torch.cuda.is_available()
GPU = [pytest.mark.gpu, pytest.mark.skipif(not HAS_CUDA, reason="needs CUDA")]
DEVICES = ["cpu", pytest.param("cuda", marks=GPU)]
INPUT_DIM = 64
IGNORE_SMALL_BATCH = pytest.mark.filterwarnings(
    "ignore:pretext='contrastive' with batch_size:UserWarning"
)


def _template(**kwargs) -> MaldiCNNClassifier:
    return MaldiCNNClassifier(channels=(4, 8), kernel_size=5, head_dim=8, **kwargs)


def _encoder(**kwargs) -> MaldiSpectrumEncoder:
    defaults = dict(
        backbone=_template(input_transform="log1p+standardize"),
        epochs=2,
        batch_size=16,
        warmup_epochs=1,
        device="cpu",
        random_state=0,
    )
    defaults.update(kwargs)
    return MaldiSpectrumEncoder(**defaults)


@pytest.fixture
def deterministic_algorithms():
    """Deterministic torch kernels (cuDNN included) for one test, restored after.

    ``warn_only``: an op without a deterministic implementation warns
    instead of raising (and cuBLAS warns without
    ``CUBLAS_WORKSPACE_CONFIG``, though it is deterministic on one
    stream).
    """
    previous = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.benchmark,
    )
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    yield
    torch.use_deterministic_algorithms(previous[0], warn_only=previous[1])
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = previous[2:]


@pytest.fixture
def pool(rng) -> np.ndarray:
    X = np.abs(rng.standard_normal((40, INPUT_DIM))).astype(np.float32)
    X[:, [5, 20, 40]] += 5.0
    return X


@pytest.fixture
def eager_compile(monkeypatch):
    """Route every ``torch.compile`` call to the fast "eager" backend and record it."""
    calls: list[object] = []
    real = torch.compile

    def spy(fn, **kwargs):
        calls.append(fn)
        return real(fn, **{**kwargs, "backend": "eager"})

    monkeypatch.setattr(torch, "compile", spy)
    torch._dynamo.reset()
    yield calls
    torch._dynamo.reset()


class TestTorchTransformTwin:
    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("mode", INPUT_TRANSFORMS)
    def test_matches_numpy_within_1e6(self, rng, mode, device):
        X = (rng.standard_normal((120, 50)) * rng.lognormal(0, 2, (120, 50))).astype(
            np.float32
        )
        X[:, 3] = 0.0
        state = fit_input_transform(X[:80], mode)
        expected = apply_input_transform(X, state)
        out = apply_input_transform_torch(torch.from_numpy(X).to(device), state)
        assert out.dtype == torch.float32 and out.device.type == device
        torch.testing.assert_close(
            out.cpu(), torch.from_numpy(expected), rtol=1e-6, atol=1e-6
        )

    def test_tensor_statistics_and_unknown_mode(self, rng):
        X = rng.standard_normal((30, 8)).astype(np.float32)
        state = fit_input_transform(X, "robust")
        as_tensors = {
            k: v if k == "mode" else torch.from_numpy(v) for k, v in state.items()
        }
        x = torch.from_numpy(X)
        assert torch.equal(
            apply_input_transform_torch(x, as_tensors),
            apply_input_transform_torch(x, state),
        )
        with pytest.raises(ValueError, match="Unknown input_transform state"):
            apply_input_transform_torch(x, {"mode": "bogus"})

    @pytest.mark.parametrize("device", DEVICES)
    def test_fitted_transform_never_shares_memory(self, rng, device):
        x = torch.from_numpy(rng.standard_normal((4, 8)).astype(np.float32))
        x = x.to(device)
        out = _FittedTransform({"mode": "none"})(x)
        assert torch.equal(out, x) and out.data_ptr() != x.data_ptr()


class _RowsOnly(Dataset):
    """Per-row view of a dataset (hides its batched ``__getitems__``)."""

    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        return self.ds[i]


def _epochs(loader, n_epochs=2):
    torch.manual_seed(0)
    return [[tuple(t.cpu() for t in b) for b in loader] for _ in range(n_epochs)]


class TestHostLoaders:
    def test_batch_reads_equal_row_reads(self, pool, tmp_path):
        path = tmp_path / "pool.npy"
        np.save(path, pool)
        X = np.load(path, mmap_mode="r")
        y = np.array([0, 1, -1, -1] * 10)
        tr, val, _ = make_pretraining_loaders(
            X, y, batch_size=7, aux_target="classification"
        )
        for loader, shuffle in ((tr, True), (val, False)):
            reference = DataLoader(
                _RowsOnly(loader.dataset),
                batch_size=7,
                shuffle=shuffle,
                drop_last=loader.drop_last,
                collate_fn=stack_collate,
            )
            for got, want in zip(_epochs(loader), _epochs(reference), strict=True):
                assert len(got) == len(want)
                for b_got, b_want in zip(got, want, strict=True):
                    for t_got, t_want in zip(b_got, b_want, strict=True):
                        assert torch.equal(t_got, t_want)

    def test_custom_collate_sees_row_items(self, pool):
        seen = []

        def collate(items):
            seen.append(items)
            xs, ts, ms = zip(*items, strict=True)
            return torch.stack(xs), torch.stack(ts), torch.stack(ms)

        _, val, _ = make_pretraining_loaders(pool, batch_size=3, collate_fn=collate)
        batch = next(iter(val))
        items = seen[0]
        assert len(items) == 3 and len(items[:2]) == 2
        for i, (x, t, m) in enumerate(items):
            assert torch.equal(x, val.dataset[i][0])
            assert torch.equal(x, batch[0][i]) and t.shape == () and m.shape == ()

    @pytest.mark.gpu
    @pytest.mark.skipif(not HAS_CUDA, reason="needs CUDA")
    def test_pin_memory(self, pool):
        tr, _, _ = make_pretraining_loaders(pool, batch_size=8, pin_memory=True)
        assert all(t.is_pinned() for t in next(iter(tr)))

    def test_workers_rejected_with_device_pool(self, pool):
        with pytest.raises(ValueError, match="num_workers must be 0"):
            make_pretraining_loaders(pool, num_workers=1, data_device="cuda")


@pytest.mark.gpu
@pytest.mark.skipif(not HAS_CUDA, reason="needs CUDA")
class TestDevicePool:
    def test_same_batches_as_host_path(self, pool):
        y = np.array(["a", "b", None, "a"] * 10, dtype=object)
        kwargs = dict(batch_size=7, aux_target="classification", unlabeled_marker=None)
        host = make_pretraining_loaders(pool, y, **kwargs)
        dev = make_pretraining_loaders(pool, y, data_device="cuda", **kwargs)
        for loader in (dev[0], dev[1]):
            assert all(t.device.type == "cuda" for t in next(iter(loader)))
        for host_loader, dev_loader in ((host[0], dev[0]), (host[1], dev[1])):
            for got, want in zip(
                _epochs(dev_loader, 3), _epochs(host_loader, 3), strict=True
            ):
                for b_got, b_want in zip(got, want, strict=True):
                    for t_got, t_want in zip(b_got, b_want, strict=True):
                        assert torch.equal(t_got, t_want)

    def test_custom_collate_on_device_pool(self, pool):
        def collate(items):
            xs, ts, ms = zip(*items, strict=True)
            return [torch.stack(xs), torch.stack(ts), torch.stack(ms)]

        tr, _, _ = make_pretraining_loaders(
            pool, batch_size=8, data_device="cuda", collate_fn=collate
        )
        x, _, _ = next(iter(tr))
        assert x.device.type == "cuda" and x.shape == (8, INPUT_DIM)


def _assert_generator_advanced_once(gen_state, generator):
    """``generator`` is one ``_sampling_generator`` draw past ``gen_state``."""
    ref = torch.Generator().set_state(gen_state)
    torch.randint(0, 2**62, (1,), generator=ref)
    assert torch.equal(ref.get_state(), generator.get_state())


class TestDeviceSampling:
    def test_cpu_generator_is_kept_on_cpu(self):
        gen = torch.Generator().manual_seed(0)
        assert _sampling_generator(gen, "cpu") is gen

    @pytest.mark.gpu
    @pytest.mark.skipif(not HAS_CUDA, reason="needs CUDA")
    def test_random_bin_mask_on_device(self):
        def draw(seed):
            gen = torch.Generator().manual_seed(seed)
            state = gen.get_state()
            mask = random_bin_mask(5, INPUT_DIM, 0.25, generator=gen, device="cuda")
            _assert_generator_advanced_once(state, gen)
            return mask

        mask = draw(0)
        assert mask.device.type == "cuda" and mask.dtype == torch.bool
        assert (mask.sum(dim=1) == 16).all()
        assert torch.equal(mask, draw(0)) and not torch.equal(mask, draw(1))

    @pytest.mark.gpu
    @pytest.mark.skipif(not HAS_CUDA, reason="needs CUDA")
    def test_shuffle_corruption_on_device(self, rng):
        x = torch.from_numpy(rng.standard_normal((12, INPUT_DIM)).astype(np.float32))
        x_raw = x.abs()
        x, x_raw = x.cuda(), x_raw.cuda()

        def corrupt(seed):
            gen = torch.Generator().manual_seed(seed)
            return shuffle_corruption(x, x_raw, 0.25, 0.5, generator=gen)

        x_corrupt, sampled, noise = corrupt(0)
        assert x_corrupt.device.type == "cuda"
        assert (sampled.sum(dim=1) == 16).all()
        # Continuous values: every swapped bin changed, 8 per row.
        assert (noise.sum(dim=1) == 8).all()
        assert not (noise & ~sampled).any()
        assert torch.equal(noise, x_corrupt != x)
        for got, want in zip(corrupt(0), (x_corrupt, sampled, noise), strict=True):
            assert torch.equal(got, want)
        assert not torch.equal(corrupt(1)[1], sampled)

    @pytest.mark.gpu
    @pytest.mark.skipif(not HAS_CUDA, reason="needs CUDA")
    def test_shuffle_sampling_follows_intensity_on_device(self):
        x_raw = torch.full((2000, INPUT_DIM), 1.0)
        x_raw[:, 0] = 1000.0
        gen = torch.Generator().manual_seed(0)
        _, sampled, _ = shuffle_corruption(
            x_raw.cuda(), x_raw.cuda(), 0.05, 0.5, generator=gen
        )
        freq = sampled.float().mean(dim=0).cpu()
        assert freq[0] > 0.99 and freq[1:].mean() < 0.1

    @pytest.mark.gpu
    @pytest.mark.skipif(not HAS_CUDA, reason="needs CUDA")
    def test_contrastive_views_on_device(self, pool):
        state = fit_input_transform(pool, "log1p")
        x_raw = torch.from_numpy(pool[:8]).cuda()

        def views(seed):
            gen = torch.Generator().manual_seed(seed)
            return contrastive_views(
                x_raw,
                _FittedTransform(state),
                0.25,
                generator=gen,
                device=x_raw.device,
            )

        a, b = views(0)
        assert a.device.type == "cuda" and not torch.equal(a, b)
        assert ((a == 0).sum(dim=1) >= 16).all()
        for got, want in zip(views(0), (a, b), strict=True):
            assert torch.equal(got, want)
        assert not torch.equal(views(1)[0], a)

    @pytest.mark.gpu
    @pytest.mark.skipif(not HAS_CUDA, reason="needs CUDA")
    @IGNORE_SMALL_BATCH
    @pytest.mark.parametrize(
        "pretext",
        [
            MAEPretext(),
            ShufflePretext(),
            ContrastivePretext(
                projection_dim=4,
                augment=SpectrumAugment(noise_std=0.2, mz_shift_max_bins=1),
            ),
        ],
        ids=["mae", "shuffle", "contrastive"],
    )
    def test_validation_loss_is_identical_across_epochs(self, pool, pretext):
        """The device draws are re-seeded every validation pass (weights fixed)."""
        torch.manual_seed(0)
        backbone = nn.Sequential(nn.Linear(INPUT_DIM, 8), nn.ReLU(), nn.Linear(8, 2))
        model = SSLModel(backbone, pretext.build_head(8, INPUT_DIM)).cuda()
        tr, val, stats = make_pretraining_loaders(
            pool, batch_size=8, transform_mode="log1p", data_device="cuda"
        )
        _, history = ssl_train_loop(
            model,
            tr,
            val,
            pretext.loss,
            torch.optim.SGD([nn.Parameter(torch.zeros(1))], lr=0.0),
            torch.device("cuda"),
            3,
            EarlyStopping(patience=5),
            input_transform_state=stats["input_transform_state"],
        )
        assert len(set(history["val_loss"])) == 1
        assert len(set(history["train_loss"])) == 3


class TestEncoderDataDevice:
    def test_cpu_training_uses_the_host_pool(self, pool):
        enc = _encoder().fit(pool)
        assert enc.data_device_ == torch.device("cpu")

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"data_device": "bogus"}, "data_device must be"),
            ({"data_device": 3.5}, "data_device must be"),
            ({"data_device": "cuda"}, "must be the training device"),
        ],
    )
    def test_invalid_data_device(self, pool, kwargs, match):
        with pytest.raises(ValueError, match=match):
            _encoder(**kwargs).fit(pool)

    def test_save_load_and_clone(self, pool, tmp_path):
        enc = _encoder(data_device="cpu", epochs=1).fit(pool)
        enc.save(tmp_path / "enc")
        loaded = MaldiSpectrumEncoder.load(tmp_path / "enc")
        assert loaded.data_device == "cpu"
        assert loaded.data_device_ == torch.device("cpu")
        assert clone(enc).get_params()["data_device"] == "cpu"

    def test_transform_equals_whole_matrix_preprocessing(self, pool):
        """Chunked host transform == the v0.4.0 whole-matrix path, bit for bit."""
        enc = _encoder(batch_size=7).fit(pool)
        X_pre = enc._preprocess(pool)
        with torch.no_grad():
            expected = np.concatenate(
                [
                    enc.model_.forward_features(torch.from_numpy(X_pre[s : s + 7]))
                    .float()
                    .numpy()
                    for s in range(0, len(pool), 7)
                ]
            )
        assert np.array_equal(enc.transform(pool), expected)

    @pytest.mark.gpu
    @pytest.mark.skipif(not HAS_CUDA, reason="needs CUDA")
    def test_auto_keeps_the_pool_on_the_gpu_or_falls_back(self, pool, monkeypatch):
        enc = _encoder(device="cuda", epochs=1).fit(pool)
        assert enc.data_device_.type == "cuda"
        assert _encoder(device="cuda", epochs=0).fit(pool).data_device_.type == "cpu"
        need = 4 * (pool.size + 4 * INPUT_DIM) + 9 * len(pool)
        monkeypatch.setattr(
            torch.cuda, "mem_get_info", lambda device=None: (2 * need - 1, 10**12)
        )
        enc = _encoder(device="cuda", epochs=1).fit(pool)
        assert enc.data_device_ == torch.device("cpu")
        monkeypatch.setattr(
            torch.cuda, "mem_get_info", lambda device=None: (2 * need, 10**12)
        )
        assert _encoder(device="cuda", epochs=1).fit(pool).data_device_.type == "cuda"

    def test_auto_queries_the_free_memory_of_an_indexed_device(self, monkeypatch):
        calls = []

        def mem_get_info(device=None):
            calls.append(device)
            return 10**12, 10**12

        monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
        monkeypatch.setattr(torch.cuda, "mem_get_info", mem_get_info)
        enc = _encoder()
        for device, expected in (("cuda", 0), (torch.device("cuda", 1), 1)):
            target = enc._resolve_data_device(torch.device(device), 40, INPUT_DIM)
            assert target == torch.device("cuda", expected)
        assert calls == [0, 1]

    @pytest.mark.parametrize("data_device", ["auto", "cuda"])
    def test_out_of_memory_upload(self, pool, monkeypatch, data_device):
        """``"auto"`` falls back to the host pool, an explicit device raises.

        Runs on CPU: the pool placement is forced to a GPU and its upload
        fails, the model trains on the CPU either way.
        """
        real = encoder_module.make_pretraining_loaders

        def upload_fails(*args, data_device, **kwargs):
            if torch.device(data_device).type == "cuda":
                raise torch.cuda.OutOfMemoryError("CUDA out of memory (test)")
            return real(*args, data_device=data_device, **kwargs)

        monkeypatch.setattr(encoder_module, "make_pretraining_loaders", upload_fails)
        monkeypatch.setattr(
            MaldiSpectrumEncoder,
            "_resolve_data_device",
            lambda self, device, n_rows, n_bins: torch.device("cuda", 0),
        )
        enc = _encoder(epochs=1, data_device=data_device)
        if data_device == "cuda":
            with pytest.raises(torch.cuda.OutOfMemoryError):
                enc.fit(pool)
            return
        with pytest.warns(UserWarning, match="did not fit on cuda:0"):
            enc.fit(pool)
        assert enc.data_device_ == torch.device("cpu")
        monkeypatch.undo()
        host = _encoder(epochs=1, data_device="cpu").fit(pool)
        for key in ("train_loss", "val_loss"):
            assert enc.history_[key] == host.history_[key]
        state = host.model_.state_dict()
        for key, value in enc.model_.state_dict().items():
            assert torch.equal(value, state[key]), key

    @pytest.mark.gpu
    @pytest.mark.skipif(not HAS_CUDA, reason="needs CUDA")
    @pytest.mark.usefixtures("deterministic_algorithms")
    @IGNORE_SMALL_BATCH
    @pytest.mark.parametrize("mode", ["standardize", "log1p+standardize"])
    @pytest.mark.parametrize(
        "pretext", ["mae", "shuffle", "shuffle+mae", "contrastive"]
    )
    def test_gpu_pool_matches_host_pool(self, pool, pretext, mode):
        """Same batches and device draws: only the transform kernels can differ.

        ``standardize`` is the same float32 arithmetic in NumPy and torch,
        so with deterministic GPU kernels both pools train bit for bit
        identically; ``log1p`` may differ in the last bit, which Adam
        amplifies a little over the epochs. The bitwise check is dropped
        when torch reports an op without a deterministic implementation.
        """
        y = np.array([0, 1, -1, -1] * 10)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fits = {
                dd: _encoder(
                    backbone=_template(input_transform=mode),
                    device="cuda",
                    data_device=dd,
                    pretext=pretext,
                    aux_lambda=0.5,
                ).fit(pool, y)
                for dd in ("cuda", "cpu")
            }
        bitwise = mode == "standardize" and not any(
            "does not have a deterministic implementation" in str(w.message)
            for w in caught
        )
        assert fits["cuda"].data_device_.type == "cuda"
        assert fits["cpu"].data_device_.type == "cpu"
        hist = {dd: enc.history_ for dd, enc in fits.items()}
        for key in ("train_loss", "val_loss"):
            if bitwise:
                assert hist["cuda"][key] == hist["cpu"][key]
            else:
                np.testing.assert_allclose(
                    hist["cuda"][key], hist["cpu"][key], rtol=1e-3, atol=1e-6
                )
        if bitwise:
            state = fits["cpu"].model_.state_dict()
            for key, value in fits["cuda"].model_.state_dict().items():
                assert torch.equal(value, state[key]), key
            np.testing.assert_array_equal(
                fits["cuda"].transform(pool), fits["cpu"].transform(pool)
            )

    @pytest.mark.gpu
    @pytest.mark.skipif(not HAS_CUDA, reason="needs CUDA")
    def test_transform_on_device_matches_host(self, pool):
        enc = _encoder(device="cuda", epochs=1).fit(pool)
        on_device = enc.transform(pool)
        enc.set_params(data_device="cpu")
        np.testing.assert_allclose(enc.transform(pool), on_device, atol=1e-5)


def _state_keys(module: nn.Module) -> list[str]:
    return list(module.state_dict())


class TestTorchCompile:
    @IGNORE_SMALL_BATCH
    @pytest.mark.parametrize("pretext", ["mae", "shuffle+mae", "contrastive"])
    def test_encoder_trains_identically_and_never_keeps_the_compiled_function(
        self, pool, eager_compile, tmp_path, pretext
    ):
        plain = _encoder(pretext=pretext).fit(pool)
        enc = _encoder(pretext=pretext, torch_compile=True).fit(pool)
        assert len(eager_compile) == 1
        assert enc.history_["train_loss"] == plain.history_["train_loss"]
        assert enc.history_["val_loss"] == plain.history_["val_loss"]
        assert enc.model_._features == enc.model_.backbone.forward_features
        assert not any("_orig_mod" in k for k in _state_keys(enc.model_))
        assert _state_keys(enc.model_) == _state_keys(plain.model_)
        pickle.loads(pickle.dumps(enc))
        assert clone(enc).get_params()["torch_compile"] is True
        enc.save(tmp_path / "enc")
        saved = torch.load(tmp_path / "enc.pt", weights_only=True)
        assert list(saved) == _state_keys(plain.model_)
        loaded = MaldiSpectrumEncoder.load(tmp_path / "enc")
        np.testing.assert_array_equal(loaded.transform(pool), enc.transform(pool))

    def test_init_from_a_compiled_encoder(self, pool, eager_compile):
        enc = _encoder(torch_compile=True).fit(pool)
        clf = _template(
            input_transform="log1p+standardize",
            init_from=enc,
            epochs=1,
            batch_size=8,
            torch_compile=True,
        ).fit(pool, np.repeat([0, 1], 20))
        assert len(eager_compile) == 2
        assert not any("_orig_mod" in k for k in _state_keys(clf.model_))
        assert type(clf.model_).__name__ == "SpectralCNN1D"

    # The factories train on CUDA when it is available, where cuDNN is not
    # deterministic by default: the fixture makes the two fits comparable
    # (its warn_only notices are expected, see its docstring).
    @pytest.mark.usefixtures("deterministic_algorithms")
    @pytest.mark.filterwarnings("ignore:Deterministic behavior was enabled:UserWarning")
    @pytest.mark.filterwarnings("ignore:Memory Efficient attention:UserWarning")
    @pytest.mark.parametrize("name", sorted(ALL_FACTORIES))
    def test_classifier_trains_identically(
        self, name, synthetic_binary, eager_compile, tmp_path
    ):
        X, y = synthetic_binary
        if name == "MaldiDualBranchClassifier":
            X = np.hstack([X, X[:, :16]])
        kwargs = {"ema_decay": 0.9} if name == "MaldiCNNClassifier" else {}
        plain = ALL_FACTORIES[name](**kwargs).fit(X, y)
        clf = ALL_FACTORIES[name](torch_compile=True, **kwargs).fit(X, y)
        assert len(eager_compile) == 1
        assert not isinstance(clf.model_, torch._dynamo.eval_frame.OptimizedModule)
        assert _state_keys(clf.model_) == _state_keys(plain.model_)
        for key, value in plain.model_.state_dict().items():
            assert torch.equal(value, clf.model_.state_dict()[key]), key
        np.testing.assert_array_equal(clf.predict_proba(X), plain.predict_proba(X))
        restored = pickle.loads(pickle.dumps(clf))
        np.testing.assert_array_equal(restored.predict_proba(X), plain.predict_proba(X))
        assert clone(clf).get_params()["torch_compile"] is True
        clf.save(tmp_path / "clf")
        saved = torch.load(tmp_path / "clf.pt", weights_only=True)
        assert list(saved) == _state_keys(plain.model_)
        loaded = type(clf).load(tmp_path / "clf")
        assert loaded.torch_compile is True
        np.testing.assert_array_equal(loaded.predict_proba(X), plain.predict_proba(X))

    @pytest.mark.slow
    @pytest.mark.gpu
    @pytest.mark.skipif(not HAS_CUDA, reason="needs CUDA")
    def test_real_backend_smoke(self, pool):
        """The default (inductor) backend compiles and trains on CUDA."""
        torch._dynamo.reset()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            enc = _encoder(
                device="cuda", torch_compile=True, use_amp=True, epochs=2
            ).fit(pool)
            clf = MaldiMLPClassifier(
                hidden_dim=16,
                head_dims=(8,),
                epochs=2,
                batch_size=8,
                device="cuda",
                torch_compile=True,
            ).fit(pool, np.repeat([0, 1], 20))
        torch._dynamo.reset()
        assert np.isfinite(enc.history_["val_loss"]).all()
        assert enc.transform(pool).shape == (len(pool), enc.feature_dim_)
        assert clf.predict_proba(pool).shape == (len(pool), 2)
        assert not any("_orig_mod" in k for k in _state_keys(enc.model_))
        assert not any("_orig_mod" in k for k in _state_keys(clf.model_))


# Constructor parameters of v0.3.0, in order (shared tail of every classifier).
_V030_BASE = [
    "learning_rate",
    "weight_decay",
    "grad_clip_norm",
    "label_smoothing",
    "loss",
    "focal_gamma",
    "use_amp",
    "swa_start_epoch",
    "tune_threshold",
    "threshold_metric",
    "calibrate_temperature",
    "min_val_auroc_for_threshold_tune",
    "use_sam",
    "sam_rho",
    "batch_size",
    "epochs",
    "early_stopping_patience",
    "val_fraction",
    "warmup_epochs",
    "standardize",
    "input_transform",
    "warping",
    "metrics_log_path",
    "track_train_metrics",
    "augment",
    "mixup_alpha",
    "cutmix_alpha",
    "ema_decay",
    "retry_on_val_auroc_below",
    "max_retries",
    "class_weight",
    "device",
    "random_state",
    "verbose",
]
_V030_HEADS = {
    BaseSpectralClassifier: ["input_dim", "n_classes"],
    MaldiMLPClassifier: [
        "input_dim",
        "n_classes",
        "hidden_dim",
        "head_dims",
        "use_attention",
        "dropout_high",
        "dropout_low",
    ],
    MaldiCNNClassifier: [
        "input_dim",
        "n_classes",
        "channels",
        "kernel_size",
        "pool_size",
        "head_dim",
        "dropout",
    ],
    MaldiResNetClassifier: [
        "input_dim",
        "n_classes",
        "stem_channels",
        "stage_channels",
        "blocks_per_stage",
        "stem_kernel_size",
        "stem_stride",
        "block_kernel_size",
        "use_stem_pool",
        "dropout",
    ],
    MaldiTransformerClassifier: [
        "input_dim",
        "n_classes",
        "patch_size",
        "embed_dim",
        "depth",
        "num_heads",
        "mlp_ratio",
        "dropout",
        "attention_dropout",
        "drop_path_rate",
        "layerscale_init",
        "pool",
        "head_dim",
    ],
    BaseDualSpectralClassifier: ["n_spec_bins", "input_dim", "n_classes"],
    MaldiDualBranchClassifier: [
        "n_spec_bins",
        "spec_hidden_dims",
        "spec_dropout",
        "fp_hidden_dims",
        "fp_dropout",
        "backbone",
        "input_dim",
        "n_classes",
    ],
}


@pytest.mark.parametrize("cls", list(_V030_HEADS), ids=lambda c: c.__name__)
def test_constructor_order_and_kinds(cls):
    """v0.3.0 parameters keep their order, new ones are appended, and only the
    identifying parameters stay positional."""
    v030 = _V030_HEADS[cls] + _V030_BASE
    params = list(inspect.signature(cls.__init__).parameters.values())[1:]
    assert [p.name for p in params[: len(v030)]] == v030
    assert [p.name for p in params[len(v030) :]] == [
        "init_from",
        "freeze_trunk_epochs",
        "torch_compile",
        "amp_dtype",
        "optimizer",
        "scheduler",
        "callbacks",
    ]
    by_name = {p.name: p for p in params}
    assert by_name["torch_compile"].default is False
    assert by_name["amp_dtype"].default == "float16"
    n_positional = 1 if issubclass(cls, BaseDualSpectralClassifier) else 2
    expected = [inspect.Parameter.KEYWORD_ONLY] * len(params)
    expected[:n_positional] = [inspect.Parameter.POSITIONAL_OR_KEYWORD] * n_positional
    assert [p.kind for p in params] == expected


def test_encoder_appends_the_acceleration_params():
    params = list(inspect.signature(MaldiSpectrumEncoder.__init__).parameters)
    assert params[-6:] == [
        "verbose",
        "data_device",
        "torch_compile",
        "optimizer",
        "scheduler",
        "callbacks",
    ]
    enc = MaldiSpectrumEncoder()
    assert enc.data_device == "auto" and enc.torch_compile is False


def test_compile_options_are_static_shapes():
    assert classifier_module._TORCH_COMPILE_OPTIONS == {"dynamic": False}


class TestAmpDtype:
    def test_resolve(self):
        cpu = torch.device("cpu")
        assert resolve_amp_dtype("float16", cpu) is torch.float16
        assert resolve_amp_dtype("bfloat16", cpu) is torch.bfloat16
        assert resolve_amp_dtype("auto", cpu) is torch.float16
        with pytest.raises(ValueError, match="amp_dtype='half'"):
            resolve_amp_dtype("half", cpu)

    @pytest.mark.skipif(not HAS_CUDA, reason="needs CUDA")
    def test_auto_follows_bf16_support(self):
        expected = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        assert resolve_amp_dtype("auto", torch.device("cuda")) is expected

    @pytest.mark.parametrize("name", sorted(ALL_FACTORIES))
    def test_classifier_param(self, name):
        clf = ALL_FACTORIES[name]()
        assert clf.get_params()["amp_dtype"] == "float16"
        assert clone(clf.set_params(amp_dtype="bfloat16")).amp_dtype == "bfloat16"

    def test_invalid_values_raise_at_fit(self, synthetic_binary, pool):
        X, y = synthetic_binary
        with pytest.raises(ValueError, match="amp_dtype"):
            ALL_FACTORIES["MaldiMLPClassifier"](amp_dtype="half").fit(X, y)
        assert MaldiSpectrumEncoder().amp_dtype == "auto"
        with pytest.raises(ValueError, match="amp_dtype"):
            _encoder(amp_dtype="half").fit(pool)

    @pytest.mark.gpu
    @pytest.mark.skipif(not HAS_CUDA, reason="needs CUDA")
    @pytest.mark.parametrize(
        ("amp_dtype", "n_scalers"), [("float16", 1), ("bfloat16", 0)]
    )
    def test_grad_scaler_only_in_float16(
        self, monkeypatch, synthetic_binary, pool, amp_dtype, n_scalers
    ):
        if amp_dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
            pytest.skip("needs bfloat16 support")
        made = []
        real = torch.amp.GradScaler

        def scaler(*args, **kwargs):
            made.append(args)
            return real(*args, **kwargs)

        monkeypatch.setattr(torch.amp, "GradScaler", scaler)
        X, y = synthetic_binary
        clf = ALL_FACTORIES["MaldiCNNClassifier"](
            device="cuda", use_amp=True, amp_dtype=amp_dtype
        ).fit(X, y)
        assert len(made) == n_scalers
        assert np.isfinite(clf.predict_proba(X)).all()
        made.clear()
        enc = _encoder(device="cuda", use_amp=True, amp_dtype=amp_dtype).fit(pool)
        assert len(made) == n_scalers
        assert np.isfinite(enc.history_["train_loss"]).all()
