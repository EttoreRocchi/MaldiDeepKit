"""Abstract base class for sklearn-compatible spectral classifiers."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import uuid
import warnings
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted
from torch import nn
from torch.utils.data import DataLoader

from ..utils.callbacks import callback_list, dispatch
from ..utils.loss import FocalLoss
from ..utils.optim import (
    build_optimizer,
    build_scheduler,
    check_scheduler,
    resolve_optimizer,
)
from ..utils.reproducibility import resolve_device, seed_everything
from ..utils.training import EarlyStopping, resolve_amp_dtype, train_loop
from ._config import same_config
from ._registry import _REGISTERED
from ._serialization import (
    _decode_param,
    _encode_param,
    _estimator_spec,
    _import_class,
    _importable_class_path,
    _init_parameters,
    _UntrustedClassError,
)
from ._trunk import (
    TrunkFreezer,
    load_trunk_state,
    split_trunk,
    trunk_prefix,
    trunk_state,
)
from .data import (
    SpectralDataset,
    _to_numpy,
    apply_fitted_preprocessing,
    make_loaders,
    resolve_transform_mode,
)

_CHECKPOINT_SUFFIXES = (".warper.pkl", ".json", ".pth", ".pt")

# Keyword arguments of every ``torch.compile`` call made for
# ``torch_compile=True`` (static shapes: a training run sees at most a
# handful of batch sizes). The unit tests switch the backend to "eager".
_TORCH_COMPILE_OPTIONS: dict[str, Any] = {"dynamic": False}


def _torch_compile(fn: Any) -> Any:
    """Return ``torch.compile(fn)`` with :data:`_TORCH_COMPILE_OPTIONS`.

    Used for ``torch_compile=True`` by the classifiers (on the training
    module) and by :class:`~maldideepkit.pretraining.MaldiSpectrumEncoder`
    (on the backbone's feature extractor).
    """
    return torch.compile(fn, **_TORCH_COMPILE_OPTIONS)


def _checkpoint_base(path: str | os.PathLike) -> Path:
    """Strip one known checkpoint suffix from ``path``.

    Only ``.pt``, ``.pth``, ``.json``, and ``.warper.pkl`` are stripped;
    any other dot is part of the name, so ``"enc_mask0.05"`` stays
    ``enc_mask0.05`` (where :meth:`pathlib.Path.with_suffix` would turn
    it into ``enc_mask0``).
    """
    base = Path(path)
    for ext in _CHECKPOINT_SUFFIXES:
        if base.name.endswith(ext) and base.name != ext:
            return base.with_name(base.name[: -len(ext)])
    return base


def _checkpoint_file(base: Path, ext: str) -> Path:
    """Append ``ext`` to the checkpoint base path ``base`` (no suffix swap)."""
    return base.with_name(base.name + ext)


def _write_checkpoint(
    base: Path, state_dict: dict[str, Any], meta_json: str, warper: Any | None
) -> None:
    """Write the ``.pt``, ``.json`` and (with a ``warper``) ``.warper.pkl`` of ``base``.

    Every file is first written to a hidden temporary name in the
    target directory, then all of them are renamed over their targets,
    so a failing write leaves neither a partial checkpoint nor a mix of
    old and new files. A stale ``.warper.pkl`` is removed when
    ``warper`` is ``None``.

    Parameters
    ----------
    base : Path
        Checkpoint base path (see :func:`_checkpoint_base`).
    state_dict : dict
        Module state dict, written with :func:`torch.save`.
    meta_json : str
        Serialised JSON metadata.
    warper : object or None
        Fitted warper, written with :func:`joblib.dump`.
    """
    base.parent.mkdir(parents=True, exist_ok=True)
    exts = (".pt", ".json") if warper is None else (".pt", ".warper.pkl", ".json")
    token = uuid.uuid4().hex
    tmp = {ext: base.with_name(f".{base.name}{ext}.{token}.tmp") for ext in exts}
    try:
        torch.save(state_dict, tmp[".pt"])
        if warper is not None:
            import joblib

            joblib.dump(warper, tmp[".warper.pkl"])
        tmp[".json"].write_text(meta_json)
    except BaseException:
        for path in tmp.values():
            path.unlink(missing_ok=True)
        raise
    for ext in exts:
        os.replace(tmp[ext], _checkpoint_file(base, ext))
    if warper is None:
        _checkpoint_file(base, ".warper.pkl").unlink(missing_ok=True)


def _serialise_transform_state(state: dict[str, Any] | None) -> dict[str, Any] | None:
    """Convert numpy arrays inside an input-transform state to lists (JSON-safe)."""
    if state is None:
        return None
    out: dict[str, Any] = {}
    for k, v in state.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        else:
            out[k] = v
    return out


def _deserialise_transform_state(
    state: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Inverse of :func:`_serialise_transform_state`; numpy-ifies list values."""
    if state is None:
        return None
    out: dict[str, Any] = {}
    for k, v in state.items():
        if k == "mode":
            out[k] = v
        elif isinstance(v, list):
            out[k] = np.asarray(v, dtype=np.float32)
        else:
            out[k] = v
    return out


def _classifier_registry() -> dict[str, type]:
    """Map every exported concrete classifier class by name.

    Built dynamically from :mod:`maldideepkit`'s public exports so
    :meth:`~maldideepkit.BaseSpectralClassifier.load` picks up new classifier classes
    (e.g. :class:`~maldideepkit.MaldiDualBranchClassifier`) without a
    hand-maintained list.
    """
    import maldideepkit

    registry: dict[str, type] = {}
    for name in maldideepkit.__all__:
        obj = getattr(maldideepkit, name, None)
        if (
            isinstance(obj, type)
            and issubclass(obj, BaseSpectralClassifier)
            and not getattr(obj, "__abstractmethods__", frozenset())
        ):
            registry[obj.__name__] = obj
    registry.update({cls.__name__: cls for cls in _REGISTERED.values()})
    return registry


def _decode_hparams(
    hparams: dict[str, Any], registry: dict[str, type], *, trusted: bool = False
) -> dict[str, Any]:
    """Rebuild constructor kwargs from a saved hyperparameter dict.

    Drops the unserialisable ``warping`` placeholder, reconstructs
    estimator-valued values stored by :meth:`_hparam_dict` as
    ``{"__estimator__": <class name>, "params": {...}}`` specs
    (recursively, so a backbone's own estimator params survive too),
    and decodes every other value with
    :func:`~maldideepkit.base._serialization._decode_param` (a
    ``Pretext`` / ``SpectrumAugment`` class is re-imported, from a
    module other than MaldiDeepKit's or an imported one only when
    ``trusted``).
    """
    hparams.pop("warping", None)
    for key, value in hparams.items():
        if isinstance(value, dict) and "__estimator__" in value:
            sub_name = value["__estimator__"]
            if sub_name in registry:
                sub_cls = registry[sub_name]
            elif value.get("__class__") is not None:
                sub_cls = _import_class(
                    value["__class__"], BaseSpectralClassifier, key, trusted=trusted
                )
            else:
                raise ValueError(
                    f"Unknown estimator class {sub_name!r} in saved "
                    f"hyperparameter {key!r}."
                )
            sub_params = _decode_hparams(
                dict(value["params"]), registry, trusted=trusted
            )
            hparams[key] = sub_cls(**sub_params)
        else:
            hparams[key] = _decode_param(value, trusted=trusted)
    return hparams


def _is_fitted_encoder(obj: Any) -> bool:
    """Return whether ``obj`` is a fitted ``MaldiSpectrumEncoder``."""
    if obj is None or isinstance(obj, (str, os.PathLike)):
        return False
    # Lazy import: ``maldideepkit.pretraining`` imports this module.
    from ..pretraining.encoder import MaldiSpectrumEncoder

    return isinstance(obj, MaldiSpectrumEncoder) and hasattr(obj, "model_")


def _load_init_encoder(init_from: Any) -> tuple[Any, dict[str, Any]]:
    """Resolve an ``init_from`` value into a fitted encoder plus provenance.

    Parameters
    ----------
    init_from : str, path-like or MaldiSpectrumEncoder
        Base path of an encoder checkpoint (``.pt`` / ``.json`` suffix
        optional) or a fitted encoder instance.

    Returns
    -------
    encoder : MaldiSpectrumEncoder
        The fitted encoder (the instance itself, not a copy).
    provenance : dict
        ``{"path": <absolute base path>, "sha256": <hex digest of the
        .pt>}`` for a path, ``{"path": None, "sha256": None}`` for an
        in-memory encoder.

    Raises
    ------
    ValueError
        If ``init_from`` is an unfitted encoder, or a checkpoint that
        names a class or a function that ``MaldiSpectrumEncoder.load``
        refuses with the default ``trusted=False`` (see its ``trusted``
        parameter).
    TypeError
        If ``init_from`` is neither a path nor an encoder.
    """
    from ..pretraining.encoder import MaldiSpectrumEncoder

    if isinstance(init_from, (str, os.PathLike)):
        base = _checkpoint_base(Path(init_from).expanduser())
        try:
            encoder = MaldiSpectrumEncoder.load(base)
        except _UntrustedClassError as exc:
            raise ValueError(
                f"init_from: {exc} A checkpoint path in init_from is loaded "
                "with trusted=False; for a trusted checkpoint, load it with "
                "MaldiSpectrumEncoder.load(path, trusted=True) and pass the "
                "encoder instance as init_from."
            ) from exc
        digest = hashlib.sha256(_checkpoint_file(base, ".pt").read_bytes()).hexdigest()
        return encoder, {"path": str(base.resolve()), "sha256": digest}
    if isinstance(init_from, MaldiSpectrumEncoder):
        if not hasattr(init_from, "model_"):
            raise ValueError(
                "init_from is an unfitted MaldiSpectrumEncoder; fit it first "
                "or pass the path of a saved encoder checkpoint. A "
                "hyperparameter search over init_from (e.g. GridSearchCV) "
                "clones each candidate, which drops the fitted state of an "
                "encoder instance: search over checkpoint paths instead "
                "(encoder.save(path), then {'init_from': [path_a, path_b]})."
            )
        return init_from, {"path": None, "sha256": None}
    raise TypeError(
        "init_from must be None, the path of a MaldiSpectrumEncoder "
        "checkpoint, or a fitted MaldiSpectrumEncoder; got "
        f"{type(init_from).__name__}."
    )


def _format_arch(name: str, params: dict[str, Any]) -> str:
    """Render ``name(k=v, ...)`` for architecture error messages."""
    return f"{name}({', '.join(f'{k}={v!r}' for k, v in params.items())})"


class BaseSpectralClassifier(ClassifierMixin, BaseEstimator, metaclass=ABCMeta):  # type: ignore[misc]
    """Abstract base for all MaldiDeepKit classifiers.

    Concrete subclasses only need to override :meth:`_build_model`,
    which should return a :class:`torch.nn.Module` that maps an input of
    shape ``(batch, input_dim)`` to logits of shape ``(batch, n_classes)``.
    Everything else (device placement, validation split, early stopping,
    checkpointing, predict / predict_proba, save / load) is provided here.

    Parameters
    ----------
    input_dim : int or None, default=None
        Number of input bins. If ``None``, inferred from ``X`` at
        :meth:`fit` time and stored as :attr:`input_dim_`.
    n_classes : int, default=2
        Number of output classes. Overwritten with the true number of
        classes found in ``y`` at :meth:`fit` time.
    learning_rate : float, default=1e-3
        Initial learning rate for the optimizer (Adam by default; AdamW
        when ``weight_decay > 0``).
    weight_decay : float, default=0.0
        L2 penalty applied via decoupled weight decay. When ``> 0`` the
        optimizer switches from ``Adam`` to ``AdamW``.
    grad_clip_norm : float or None, default=None
        If set, clip gradient global L2 norm to this value before every
        optimizer step. ``1.0`` is a common default for transformers.
    label_smoothing : float, default=0.0
        Label smoothing factor in ``[0, 1)`` passed to the loss. Applied
        to both cross-entropy and focal-loss paths.
    loss : {"cross_entropy", "focal"}, nn.Module or callable, default="cross_entropy"
        Classification loss. ``"focal"`` uses
        :class:`~maldideepkit.utils.FocalLoss` with ``gamma=focal_gamma``.
        Good for highly imbalanced problems. A module (or module class,
        instantiated without arguments) or a function is called as
        ``loss(logits, target)`` and returns one value per row or a scalar
        (a scalar cannot be combined with ``sample_weight``). ``target``
        holds class indices, or probabilities of shape ``(batch,
        n_classes)`` on training batches with ``mixup_alpha`` /
        ``cutmix_alpha``. ``class_weight`` and ``label_smoothing`` only
        apply to the built-in losses. A module is copied at :meth:`fit`.
    focal_gamma : float, default=2.0
        Focal-loss focusing parameter. Ignored when
        ``loss="cross_entropy"``.
    use_amp : bool, default=False
        If ``True`` and the resolved device is a GPU, run the forward pass
        and the loss under :func:`torch.autocast` in ``amp_dtype``. About
        2x faster on recent NVIDIA GPUs; no effect on CPU.
    swa_start_epoch : int or None, default=None
        If set, start Stochastic Weight Averaging at this epoch. The SWA
        average replaces the best-val checkpoint at the end of fit.
        Typical value: 60-80% of ``epochs``.
    tune_threshold : bool, default=False
        (Binary classification only.) After fit, sweep thresholds on
        the validation split and store the one that maximises
        ``threshold_metric``. :meth:`predict` uses this threshold
        instead of ``argmax @ 0.5``.
    threshold_metric : {"balanced_accuracy", "f1", "youden"}, default="balanced_accuracy"
        Metric used by ``tune_threshold``.
    calibrate_temperature : bool, default=False
        If ``True``, after fit run LBFGS-based temperature scaling on
        held-out validation logits (Guo et al., 2017). The fitted
        temperature is stored as :attr:`temperature_` and applied in
        :meth:`predict_proba` to sharpen / smooth probabilities
        without changing the argmax.
    min_val_auroc_for_threshold_tune : float, default=0.6
        Binary-classification guardrail on ``tune_threshold=True``: if
        the validation AUROC is below this value, the threshold sweep
        is skipped and ``threshold_`` falls back to ``0.5``. Set to
        ``0.0`` to disable.
    use_sam : bool, default=False
        If ``True``, wrap the base optimizer in
        :class:`~maldideepkit.utils.SAMOptimizer` and run the two-step
        Sharpness-Aware Minimization update. Doubles forward / backward
        compute per step; typically helps generalisation on small
        datasets.
    sam_rho : float, default=0.05
        Size of the SAM ascent step. Ignored when ``use_sam=False``.
    batch_size : int, default=32
        Training mini-batch size.
    epochs : int, default=100
        Maximum number of training epochs.
    early_stopping_patience : int, default=10
        Number of epochs without validation-loss improvement before
        training stops.
    val_fraction : float, default=0.1
        Fraction of the training data held out for the internal
        validation split.
    warmup_epochs : int, default=0
        If positive, linearly ramp each optimizer param group's learning
        rate from ``0`` to its configured target over the first
        ``warmup_epochs`` epochs. Useful for transformer architectures
        that can diverge at full learning rate during the first few steps.
    standardize : bool, default=False
        Shorthand for ``input_transform="standardize"`` (when True) or
        ``"none"`` (when False). Kept for backwards compatibility;
        ``input_transform`` is the modern interface and wins when
        both are supplied.
    input_transform : str or None, default=None
        One of ``{"none", "standardize", "log1p", "robust",
        "log1p+standardize"}``, or ``None`` to follow ``standardize``.
        Fit on the (warped) training split only and stored as
        :attr:`input_transform_state_`; reapplied at :meth:`predict` /
        :meth:`predict_proba` time.
    warping : sklearn-style transformer or None, default=None
        Spectral alignment / warping transformer applied **before**
        standardization. Fitted on the training split only, then
        used to transform both splits during training and new data
        at :meth:`predict` / :meth:`predict_proba` time. The fitted
        transformer is stored as :attr:`warper_`.
    metrics_log_path : str, Path or None, default=None
        If set, write a per-epoch metrics CSV to this path during
        :meth:`fit`. One row per epoch with columns ``epoch,
        train_loss, val_loss, lr, mean_grad_norm, n_grad_updates``
        (+ ``train_auroc, val_auroc`` when ``track_train_metrics=True``).
    track_train_metrics : bool, default=False
        Only used when ``metrics_log_path`` is set. If ``True``, after
        every epoch run a no-grad forward pass over the full training
        split and record ``train_auroc`` + ``val_auroc`` alongside the
        losses. Adds one extra pass per epoch; binary classification
        only.
    augment : callable or None, default=None
        Per-batch augmentation applied to training batches only. The
        usual choice is :class:`~maldideepkit.augment.SpectrumAugment`.
    mixup_alpha : float, default=0.0
        If positive, apply MixUp augmentation per training batch with a
        Beta(``mixup_alpha``, ``mixup_alpha``) mixing coefficient.
        ``0.0`` disables MixUp. Composable with ``cutmix_alpha``.
    cutmix_alpha : float, default=0.0
        If positive, apply CutMix augmentation per training batch with
        a Beta(``cutmix_alpha``, ``cutmix_alpha``) mixing coefficient.
        ``0.0`` disables CutMix.
    ema_decay : float or None, default=None
        If set (typically ``0.999``), maintain an exponential moving
        average of model weights during training and use the EMA weights
        at inference time.
    retry_on_val_auroc_below : float or None, default=None
        Binary-classification guardrail. If set and the post-fit
        validation AUROC is below this threshold, retrain with a
        different RNG seed up to ``max_retries`` times. Useful for
        unstable small-data fits.
    max_retries : int, default=2
        Maximum number of automatic refits triggered by
        ``retry_on_val_auroc_below``. Ignored when that guardrail is
        unset.
    class_weight : {"balanced", None} or array-like, default=None
        Per-class weights applied to :class:`~torch.nn.CrossEntropyLoss`.
        ``"balanced"`` uses ``n_samples / (n_classes * class_count)``.
    device : {"auto", "cpu", "cuda"} or torch.device, default="auto"
        Device used for training and inference.
    random_state : int, default=0
        Seeds Python, NumPy, and PyTorch RNGs and the validation split.
    verbose : bool, default=False
        If ``True``, print one line per epoch.
    init_from : str, path-like, MaldiSpectrumEncoder or None, default=None
        Initialise the trunk (every parameter and buffer of the
        backbone module except its ``final_layer``) from a
        self-supervised :class:`~maldideepkit.pretraining.MaldiSpectrumEncoder`:
        either the base path of a checkpoint written by its ``save``
        (clone-safe, works in ``GridSearchCV``) or a fitted encoder
        instance (kept by reference across :func:`sklearn.base.clone`
        copies, so it is shared and must be treated as read-only).
        A search **over** ``init_from`` itself (e.g.
        ``GridSearchCV(clf, {"init_from": [...]})``) must list
        checkpoint paths: the search clones every candidate value,
        which turns a fitted encoder instance into an unfitted copy and
        makes :meth:`fit` raise :class:`ValueError`.
        A path is loaded with the default ``trusted=False`` (see
        :meth:`MaldiSpectrumEncoder.load
        <maldideepkit.pretraining.MaldiSpectrumEncoder.load>`); for a
        trusted checkpoint, pass
        ``MaldiSpectrumEncoder.load(path, trusted=True)``. A
        ``.warper.pkl`` is a pickle, read whatever ``trusted`` is: load
        checkpoints with a warper only from trusted sources.
        Resolved once per :meth:`fit`. The classifier then **inherits
        the encoder's fitted preprocessing** (warper and input-transform
        state) instead of fitting its own on the training split, so
        ``standardize`` / ``input_transform`` (compared on the resolved
        mode), ``warping`` (compared by configuration) and the number
        of input bins must match the encoder's, else
        :class:`ValueError`. A trunk tensor missing on either side or
        with a different shape also raises :class:`ValueError` (no
        partial load). ``fit(warm_start=True)`` with an existing
        :attr:`model_` takes precedence on the first attempt; every
        rebuilt model (including retries) is initialised from the
        encoder. :meth:`load` never re-applies ``init_from``.
    freeze_trunk_epochs : int, default=0
        Linear probing then fine-tuning (LP-FT; Kumar et al., 2022): for
        the first ``freeze_trunk_epochs`` epochs only the
        ``final_layer`` is trained, the trunk parameters are frozen and
        the trunk modules run in ``eval()`` mode (BatchNorm running
        statistics and dropout frozen); then the trunk is unfrozen and
        the early-stopping patience restarts. A value ``>= epochs``
        gives a pure linear probe. Early stopping also acts during the
        frozen phase: with ``early_stopping_patience <
        freeze_trunk_epochs`` training can end before the trunk is
        unfrozen, which leaves a pure linear probe
        (``trunk_unfroze_=False``). Valid without ``init_from``, but
        only meaningful with it. The ``requires_grad`` flags are
        restored after :meth:`fit` (a parameter frozen by the user
        stays frozen).
    torch_compile : bool, default=False
        If ``True``, the training and validation forward passes of
        :meth:`fit` run through ``torch.compile(model)``. Compilation
        happens on the first batches (and again for each new batch
        shape, e.g. the last batch of an epoch) and takes seconds to
        minutes, so it pays off on long runs, mostly on a GPU. The
        compiled wrapper only lives during :meth:`fit`: :attr:`model_`,
        the :meth:`save` checkpoint and the ``init_from`` transfer
        always use the underlying module (state-dict keys without the
        ``_orig_mod.`` prefix), inference runs eagerly, and pickling
        or :func:`~sklearn.base.clone` never see a compiled module.
    amp_dtype : {"float16", "bfloat16", "auto"}, default="float16"
        Autocast dtype of ``use_amp=True`` on a GPU (ignored otherwise).
        ``"float16"`` scales the loss with :class:`torch.amp.GradScaler`.
        ``"bfloat16"`` keeps the exponent range of ``float32``, so an
        activation cannot overflow (as a wide ``Linear`` over a
        flattened feature map can in ``float16`` once its weights have
        grown, e.g. in a trunk transferred with ``init_from``), and
        trains without a ``GradScaler``. ``"auto"`` picks
        ``"bfloat16"`` when the GPU supports it, else ``"float16"``.
    optimizer : None, str or callable, default=None
        ``None`` uses :class:`torch.optim.AdamW` when ``weight_decay > 0``,
        else :class:`torch.optim.Adam`. A string is the case-insensitive
        name of a :mod:`torch.optim` class (e.g. ``"sgd"``). A class or
        factory is called as ``optimizer(param_groups, lr=learning_rate,
        weight_decay=weight_decay)``, each keyword only when it accepts it
        and a :func:`functools.partial` does not already fix it, so
        ``functools.partial(torch.optim.SGD, momentum=0.9)`` works.
        Combines with ``use_sam``. A closure-based optimizer such as
        :class:`torch.optim.LBFGS` is not supported.
    scheduler : None, str or callable, default=None
        Learning-rate schedule after the ``warmup_epochs`` epochs.
        ``None`` or ``"cosine"`` anneals to ``1e-6`` over the remaining
        epochs; ``"constant"`` keeps the learning rate. A class or factory
        is called as ``scheduler(optimizer)``. It is stepped once per
        epoch (:class:`~torch.optim.lr_scheduler.ReduceLROnPlateau` with
        the validation loss); a per-batch schedule fits in a callback.
    callbacks : Callback, sequence of Callback or None, default=None
        Objects whose hooks (:class:`~maldideepkit.utils.Callback`) run
        during :meth:`fit`. Clones of the estimator share them.

    Attributes
    ----------
    model_ : torch.nn.Module
        The fitted PyTorch model.
    classes_ : ndarray of shape (n_classes,)
        Original class labels seen during :meth:`fit`.
    input_dim_ : int
        Resolved number of input features.
    n_classes_ : int
        Resolved number of classes.
    feature_mean_ : ndarray or None
        Per-feature mean fitted on the training split. Populated when
        ``standardize=True`` or ``input_transform`` includes
        standardization (``"standardize"`` or ``"log1p+standardize"``);
        ``None`` otherwise.
    feature_std_ : ndarray or None
        Per-feature standard deviation fitted on the training split.
        Populated under the same conditions as :attr:`feature_mean_`.
        For dual-input classifiers both arrays cover the spectrum
        block only (see :class:`~maldideepkit.BaseDualSpectralClassifier`).
    n_features_in_ : int
        Number of features seen at :meth:`fit` (scikit-learn convention).
    trunk_unfroze_ : bool
        ``False`` when ``freeze_trunk_epochs > 0`` and training ended
        before epoch ``freeze_trunk_epochs`` (the trunk stayed frozen
        and is then restored exactly to its initial weights, undoing
        EMA / SWA averaging drift and ``update_bn``); ``True``
        otherwise (always with ``freeze_trunk_epochs=0``).
    best_epoch_phase_ : {"frozen", "unfrozen"}
        Training phase that produced the deployed weights:
        ``"frozen"`` when the trunk never unfroze or the
        best-validation epoch fell in the frozen phase, ``"unfrozen"``
        otherwise (EMA / SWA averages including an unfrozen epoch count
        as ``"unfrozen"``; always ``"unfrozen"`` with
        ``freeze_trunk_epochs=0``).
    """

    def __init__(
        self,
        input_dim: int | None = None,
        n_classes: int = 2,
        *,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        grad_clip_norm: float | None = None,
        label_smoothing: float = 0.0,
        loss: Any = "cross_entropy",
        focal_gamma: float = 2.0,
        use_amp: bool = False,
        swa_start_epoch: int | None = None,
        tune_threshold: bool = False,
        threshold_metric: str = "balanced_accuracy",
        calibrate_temperature: bool = False,
        min_val_auroc_for_threshold_tune: float = 0.6,
        use_sam: bool = False,
        sam_rho: float = 0.05,
        batch_size: int = 32,
        epochs: int = 100,
        early_stopping_patience: int = 10,
        val_fraction: float = 0.1,
        warmup_epochs: int = 0,
        standardize: bool = False,
        input_transform: str | None = None,
        warping: Any | None = None,
        metrics_log_path: str | Path | None = None,
        track_train_metrics: bool = False,
        augment: Callable[[torch.Tensor], torch.Tensor] | None = None,
        mixup_alpha: float = 0.0,
        cutmix_alpha: float = 0.0,
        ema_decay: float | None = None,
        retry_on_val_auroc_below: float | None = None,
        max_retries: int = 2,
        class_weight: str | np.ndarray | list | None = None,
        device: str | torch.device = "auto",
        random_state: int = 0,
        verbose: bool = False,
        init_from: str | Path | Any | None = None,
        freeze_trunk_epochs: int = 0,
        torch_compile: bool = False,
        amp_dtype: str = "float16",
        optimizer: Any = None,
        scheduler: Any = None,
        callbacks: Any = None,
    ) -> None:
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.grad_clip_norm = grad_clip_norm
        self.label_smoothing = label_smoothing
        self.loss = loss
        self.focal_gamma = focal_gamma
        self.use_amp = use_amp
        self.swa_start_epoch = swa_start_epoch
        self.tune_threshold = tune_threshold
        self.threshold_metric = threshold_metric
        self.calibrate_temperature = calibrate_temperature
        self.min_val_auroc_for_threshold_tune = min_val_auroc_for_threshold_tune
        self.use_sam = use_sam
        self.sam_rho = sam_rho
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.val_fraction = val_fraction
        self.warmup_epochs = warmup_epochs
        self.standardize = standardize
        self.input_transform = input_transform
        self.warping = warping
        self.metrics_log_path = metrics_log_path
        self.track_train_metrics = track_train_metrics
        self.augment = augment
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.ema_decay = ema_decay
        self.retry_on_val_auroc_below = retry_on_val_auroc_below
        self.max_retries = max_retries
        self.class_weight = class_weight
        self.device = device
        self.random_state = random_state
        self.verbose = verbose
        self.init_from = init_from
        self.freeze_trunk_epochs = freeze_trunk_epochs
        self.torch_compile = torch_compile
        self.amp_dtype = amp_dtype
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.callbacks = callbacks

    @abstractmethod
    def _build_model(self) -> nn.Module:
        """Return a fresh :class:`nn.Module` for the current hyperparameters.

        Implementations should use :attr:`input_dim_` and
        :attr:`n_classes_` rather than the constructor arguments, since
        those are the values resolved at :meth:`fit` time.
        """

    def _arch_params(self) -> dict[str, Any]:
        """Return the architecture kwargs :meth:`_build_model` passes to the module.

        Excludes the fit-time ``input_dim`` / ``n_classes``, so it is
        available on an unfitted template. Built-in classifiers return
        exactly the kwargs of their backbone module; the default (for
        custom subclasses) is empty. Used in ``init_from`` error
        messages.
        """
        return {}

    @classmethod
    def _get_param_names(cls) -> list[str]:
        """Return the constructor parameters, following ``**kwargs`` up the MRO.

        As :class:`~sklearn.base.BaseEstimator`, except that a subclass
        ``__init__`` taking ``**kwargs`` also has the parameters of the
        next ``__init__`` in the MRO. So
        ``def __init__(self, hidden=16, **kwargs): super().__init__(**kwargs)``
        keeps every base parameter through ``clone``, ``GridSearchCV``,
        and ``save`` / ``load``. Such a subclass must forward ``kwargs``
        unchanged: fixing a base parameter in its ``super().__init__``
        call raises :class:`TypeError` when it is cloned.
        """
        return sorted(_init_parameters(cls))

    def __sklearn_clone__(self) -> BaseSpectralClassifier:
        """Clone as :class:`~sklearn.base.BaseEstimator` does, sharing a fitted encoder.

        A fitted :class:`~maldideepkit.pretraining.MaldiSpectrumEncoder`
        passed as ``init_from`` is reattached by reference (a plain
        clone would drop its fitted state); everything else, including
        metadata-routing requests, follows the scikit-learn default.
        """
        new = super().__sklearn_clone__()
        if _is_fitted_encoder(self.init_from):
            new.init_from = self.init_from
        return new

    def set_params(self, **params: Any) -> BaseSpectralClassifier:
        """Set parameters, refusing to mutate a shared fitted encoder.

        Same as :meth:`sklearn.base.BaseEstimator.set_params`, except
        that while ``init_from`` is a fitted encoder (shared by
        reference across clones) an ``init_from__<name>`` key whose
        value differs from the encoder's raises :class:`ValueError`.
        Keys whose value is unchanged are skipped, so no-op round trips
        such as ``set_params(**get_params(deep=True))`` are accepted.
        """
        encoder = params.get("init_from", self.init_from)
        if _is_fitted_encoder(encoder):
            current = encoder.get_params(deep=True)
            for key in [k for k in params if k.startswith("init_from__")]:
                name = key[len("init_from__") :]
                if name not in current:
                    continue
                if not same_config(params[key], current[name]):
                    raise ValueError(
                        f"Cannot set {key}: init_from is a fitted "
                        "MaldiSpectrumEncoder shared by reference across clones "
                        "and must stay read-only. Fit a new encoder instead, or "
                        "pass init_from as a checkpoint path."
                    )
                del params[key]
        return super().set_params(**params)

    def _optimizer_param_groups(self, model: nn.Module) -> list[dict[str, Any]]:
        """Return parameter groups for the optimizer.

        Default: a single group containing every parameter of ``model``.
        Override in a subclass to give specific parameters a different
        learning rate or weight decay.

        Parameters
        ----------
        model : nn.Module
            The freshly built, on-device training model.

        Returns
        -------
        list of dict
            Param-group dicts suitable for :class:`torch.optim.Adam`.
            Groups without an explicit ``lr`` or ``weight_decay``
            inherit the defaults from :meth:`fit`.
        """
        return [{"params": list(model.parameters())}]

    def _resolve_device(self) -> torch.device:
        return resolve_device(self.device)

    def _compute_class_weight(self, y: np.ndarray) -> torch.Tensor | None:
        if self.class_weight is None:
            return None
        if isinstance(self.class_weight, str):
            if self.class_weight != "balanced":
                raise ValueError(
                    f"Unknown class_weight={self.class_weight!r}; "
                    "use 'balanced', None, or an array."
                )
            counts = np.bincount(y, minlength=self.n_classes_)
            if np.any(counts == 0):
                missing = np.flatnonzero(counts == 0).tolist()
                raise ValueError(
                    "class_weight='balanced' requires every class to be "
                    f"present in y; missing class indices: {missing}."
                )
            weights = len(y) / (self.n_classes_ * counts)
            return torch.tensor(weights, dtype=torch.float32)
        weights = np.asarray(self.class_weight, dtype=np.float32)
        if weights.shape != (self.n_classes_,):
            raise ValueError(
                f"class_weight has shape {weights.shape}, "
                f"expected ({self.n_classes_},)."
            )
        return torch.from_numpy(weights)

    def _build_criterion(
        self, class_weight: torch.Tensor | None, *, reduction: str = "none"
    ) -> nn.Module | Callable[..., torch.Tensor]:
        if not isinstance(self.loss, str):
            loss = self.loss() if isinstance(self.loss, type) else self.loss
            if isinstance(loss, nn.Module):
                return copy.deepcopy(loss)
            if callable(loss):
                return loss
            raise TypeError(
                "loss must be 'cross_entropy', 'focal', an nn.Module or a callable; "
                f"got {type(loss).__name__}."
            )
        if self.loss == "cross_entropy":
            return nn.CrossEntropyLoss(
                weight=class_weight,
                label_smoothing=float(self.label_smoothing),
                reduction=reduction,
            )
        if self.loss == "focal":
            return FocalLoss(
                weight=class_weight,
                gamma=float(self.focal_gamma),
                label_smoothing=float(self.label_smoothing),
                reduction=reduction,
            )
        raise ValueError(
            f"Unknown loss={self.loss!r}; expected 'cross_entropy', 'focal', an "
            "nn.Module or a callable."
        )

    @staticmethod
    def _validate_sample_weight(
        sample_weight: Any, n_samples: int
    ) -> np.ndarray | None:
        """Coerce ``sample_weight`` to a validated ``float32`` ndarray.

        Returns ``None`` when the input is ``None``. Otherwise enforces
        shape ``(n_samples,)``, finite values, and non-negativity, in
        the spirit of :func:`sklearn.utils.validation._check_sample_weight`.
        """
        if sample_weight is None:
            return None
        sw = np.asarray(sample_weight, dtype=np.float32).ravel()
        if sw.shape[0] != n_samples:
            raise ValueError(
                f"sample_weight has {sw.shape[0]} entries but X has {n_samples} rows."
            )
        if not np.all(np.isfinite(sw)):
            raise ValueError("sample_weight contains non-finite values (NaN/Inf).")
        if np.any(sw < 0):
            raise ValueError("sample_weight must be non-negative.")
        return sw

    def _make_mix_generator(self) -> torch.Generator | None:
        if float(self.mixup_alpha) <= 0 and float(self.cutmix_alpha) <= 0:
            return None
        gen = torch.Generator()
        gen.manual_seed(int(self.random_state))
        return gen

    def _prepare_inputs(self, X: Any, y: Any) -> tuple[np.ndarray, np.ndarray]:
        X_np = _to_numpy(X)
        if hasattr(y, "to_numpy"):
            y = y.to_numpy()
        y_np = np.asarray(y).ravel()
        if X_np.shape[0] != y_np.shape[0]:
            raise ValueError(f"X has {X_np.shape[0]} rows but y has {y_np.shape[0]}.")
        self.classes_ = unique_labels(y_np)
        self.n_classes_ = int(len(self.classes_))
        if self.n_classes_ < 2:
            raise ValueError(
                f"{type(self).__name__} needs at least 2 classes in y; "
                f"got {self.n_classes_}."
            )
        self.input_dim_ = (
            int(X_np.shape[1]) if self.input_dim is None else int(self.input_dim)
        )
        if self.input_dim_ != X_np.shape[1]:
            raise ValueError(
                f"input_dim={self.input_dim_} does not match X.shape[1]={X_np.shape[1]}."
            )
        self.n_features_in_ = self.input_dim_
        y_encoded = np.searchsorted(self.classes_, y_np).astype(np.int64)
        return X_np, y_encoded

    def _resolve_init_encoder(self) -> Any | None:
        """Resolve ``init_from`` into a fitted encoder checked against this classifier.

        Called once per :meth:`fit`, after :meth:`_prepare_inputs`. A
        path is loaded with ``MaldiSpectrumEncoder.load``; an instance
        must be fitted. Records the checkpoint provenance written by
        :meth:`save`.

        Returns
        -------
        MaldiSpectrumEncoder or None
            The encoder, or ``None`` when ``init_from`` is unset.
        """
        self._init_from_provenance: dict[str, Any] | None = None
        if self.init_from is None:
            return None
        encoder, provenance = _load_init_encoder(self.init_from)
        self._check_init_encoder(encoder, int(self.input_dim_))
        self._init_from_provenance = provenance
        return encoder

    def _check_init_encoder(self, encoder: Any, n_spec_bins: int) -> None:
        """Check that this classifier's preprocessing config matches ``encoder``.

        With ``init_from`` the classifier inherits the encoder's fitted
        preprocessing, so its config is an assertion: the number of
        spectrum bins, the resolved input-transform mode and the
        warping configuration (both absent, or :func:`same_config`
        against the encoder's fitted ``warper_``) must all match.

        Raises
        ------
        ValueError
            Naming the first mismatching field.
        """
        if int(encoder.input_dim_) != int(n_spec_bins):
            raise ValueError(
                f"init_from: input_dim mismatch: the encoder was fitted on "
                f"{encoder.input_dim_} bins but this classifier has "
                f"{n_spec_bins} spectrum bins."
            )
        mode = resolve_transform_mode(bool(self.standardize), self.input_transform)
        encoder_mode = encoder.input_transform_state_.get("mode", "none")
        if mode != encoder_mode:
            raise ValueError(
                f"init_from: input_transform mismatch: this classifier resolves "
                f"standardize={self.standardize!r}, "
                f"input_transform={self.input_transform!r} to {mode!r} but the "
                f"encoder was fitted with {encoder_mode!r}. The classifier "
                "inherits the encoder's preprocessing, so set "
                f"input_transform={encoder_mode!r}."
            )
        warper = encoder.warper_
        if (self.warping is None) != (warper is None) or (
            warper is not None and not same_config(self.warping, warper)
        ):
            raise ValueError(
                f"init_from: warping mismatch: this classifier has "
                f"warping={self.warping!r} but the encoder was fitted with "
                f"warper {warper!r}. The classifier inherits the encoder's "
                "fitted warper, so its warping config must be identical."
            )

    def _frozen_epochs(self) -> int:
        """Return the number of epochs of the linear-probing phase."""
        return int(self.freeze_trunk_epochs)

    def _trunk_target(self, model: nn.Module) -> tuple[nn.Module, str | None]:
        """Return ``(module, skip_prefix)`` locating the trunk inside ``model``.

        ``init_from`` copies the encoder trunk into every tensor of
        ``module`` outside ``skip_prefix``; ``freeze_trunk_epochs``
        freezes those tensors. Single-input classifiers return the
        whole model and the qualified name of its final layer: its
        ``final_layer``, or the final :class:`~torch.nn.Linear` of a
        custom module without one (see
        :func:`~maldideepkit.base._trunk.trunk_prefix`).
        """
        return model, trunk_prefix(model, self.input_dim_)

    def _build_initial_model(
        self, device: torch.device, encoder: Any | None
    ) -> nn.Module:
        """Build a fresh model on ``device``, trunk copied from ``encoder`` if given.

        Every trunk parameter and buffer is copied; the final layer
        keeps its fresh initialisation. The trunk key sets must be
        equal in both directions (final layer excluded on both sides)
        with equal shapes, else :class:`ValueError` (no partial load).
        """
        model = self._build_model().to(device)
        if encoder is None:
            return model
        target, skip = self._trunk_target(model)
        own = trunk_state(target, skip)
        backbone = encoder.model_.backbone
        source = trunk_state(backbone, trunk_prefix(backbone, encoder.input_dim_))
        missing = sorted(set(own) - set(source))
        unexpected = sorted(set(source) - set(own))
        mismatched = sorted(
            k for k in set(own) & set(source) if own[k].shape != source[k].shape
        )
        if missing or unexpected or mismatched:
            details = []
            if missing:
                details.append(
                    f"{len(missing)} trunk tensor(s) missing from the encoder "
                    f"(e.g. {missing[:3]})"
                )
            if unexpected:
                details.append(
                    f"{len(unexpected)} encoder trunk tensor(s) absent from this "
                    f"classifier (e.g. {unexpected[:3]})"
                )
            if mismatched:
                details.append(
                    "shape mismatch "
                    + ", ".join(
                        f"{k}: {tuple(own[k].shape)} vs {tuple(source[k].shape)}"
                        for k in mismatched[:3]
                    )
                )
            template = encoder.backbone_template_
            own_arch = self._arch_params()
            raise ValueError(
                "init_from: the encoder trunk does not match this classifier's "
                f"trunk ({'; '.join(details)}). The encoder was pre-trained from "
                f"{_format_arch(type(template).__name__, template._arch_params())}; "
                "this classifier builds "
                + (
                    _format_arch(type(self).__name__, own_arch)
                    if own_arch
                    else type(target).__name__
                )
                + ". Pre-train the encoder from a template with the same "
                "architecture."
            )
        load_trunk_state(target, skip, source)
        return model

    def _best_epoch_phase(
        self, freezer: TrunkFreezer | None, early: EarlyStopping
    ) -> str:
        """Return the training phase of the deployed weights (``"frozen"`` / ``"unfrozen"``)."""
        if freezer is None:
            return "unfrozen"
        if not freezer.unfroze:
            return "frozen"
        averaged = self.ema_decay is not None or (
            self.swa_start_epoch is not None
            and freezer.last_epoch >= int(self.swa_start_epoch)
        )
        if (
            not averaged
            and early.best_epoch is not None
            and early.best_epoch < freezer.n_frozen
        ):
            return "frozen"
        return "unfrozen"

    def fit(
        self,
        X: Any,
        y: Any,
        *,
        sample_weight: Any | None = None,
        warm_start: bool = False,
    ) -> BaseSpectralClassifier:
        """Fit the model on ``(X, y)``.

        Parameters
        ----------
        X : array-like or MaldiSet of shape (n_samples, n_bins)
            Training spectra. NumPy arrays, pandas DataFrames, and
            objects with a DataFrame-like ``.X`` attribute are accepted.
        y : array-like of shape (n_samples,)
            Integer or string class labels. Re-encoded to ``0..n_classes-1``
            internally; original labels are preserved in :attr:`classes_`.
        sample_weight : array-like of shape (n_samples,) or None, default=None
            Per-sample weights applied inside the loss. The training
            batch loss becomes the sklearn-style weighted mean
            ``sum_i (sw_i * cw_{y_i}) * l_i / sum_i sw_i * cw_{y_i}``,
            composable with ``class_weight``. Doubling a sample's weight
            is equivalent to including the row twice. Weights are split
            alongside ``X`` and ``y`` into the training and validation
            folds via the same stratified split, and the val-side
            weights flow into the early-stopping signal, the post-fit
            AUROC reporting, ``tune_threshold``, ``calibrate_temperature``,
            and the retry-on-collapse guardrail. Must be non-negative
            and finite; cast to ``float32``. Under MixUp / CutMix, the
            weights are convex-combined with the same ``lam`` used to
            mix labels.
        warm_start : bool, default=False
            When ``True`` and the estimator already has a fitted
            :attr:`model_`, the underlying :class:`torch.nn.Module` is
            reused as the starting point of training instead of being
            rebuilt from scratch via :meth:`_build_model`, so ``fit()``
            *resumes* from the current weights rather than
            reinitialising them (as federated learning, continual
            learning, and fine-tuning workflows need). ``warm_start`` applies only
            to the first training attempt; retries triggered by
            ``retry_on_val_auroc_below`` always rebuild via
            :meth:`_build_model` (the warm-start weights already failed
            once). When ``warm_start=True`` but no prior ``model_``
            exists, falls back silently to a fresh build (scikit-learn
            convention). Every fresh build starts from the ``init_from``
            trunk when it is set; the warm-started module does not
            (its weights take precedence), but the encoder's
            preprocessing is still inherited.

        Returns
        -------
        self : BaseSpectralClassifier
            The fitted estimator.
        """
        X_np, y_encoded = self._prepare_inputs(X, y)
        # Resolved before seeding: loading an encoder checkpoint builds
        # a module and so consumes PyTorch RNG draws.
        encoder = self._resolve_init_encoder()
        n_frozen = self._frozen_epochs()
        if n_frozen < 0:
            raise ValueError(f"freeze_trunk_epochs must be >= 0; got {n_frozen!r}.")
        resolve_optimizer(self.optimizer, float(self.weight_decay))
        check_scheduler(self.scheduler)
        callbacks = callback_list(self.callbacks)
        seed_everything(int(self.random_state))
        sw_np = self._validate_sample_weight(sample_weight, X_np.shape[0])
        device = self._resolve_device()

        if encoder is None:
            warper = clone(self.warping) if self.warping is not None else None
            fitted_preprocessing = None
        else:
            # Inherit the encoder's fitted preprocessing (copied, so the
            # shared encoder is never mutated through this classifier).
            warper = None
            fitted_preprocessing = {
                "warper": copy.deepcopy(encoder.warper_),
                "input_transform_state": copy.deepcopy(encoder.input_transform_state_),
            }
        train_loader, val_loader, stats = make_loaders(
            X_np,
            y_encoded,
            batch_size=int(self.batch_size),
            val_size=float(self.val_fraction),
            random_state=int(self.random_state),
            standardize=bool(self.standardize),
            input_transform=self.input_transform,
            warper=warper,
            sample_weight=sw_np,
            n_transform_cols=self._n_transform_cols(),
            fitted_preprocessing=fitted_preprocessing,
        )
        self.feature_mean_ = stats["mean"]
        self.feature_std_ = stats["std"]
        self.warper_ = stats["warper"]
        self.input_transform_state_ = stats["input_transform_state"]
        sw_val_np: np.ndarray | None = stats.get("sample_weight_val")

        builtin_loss = isinstance(self.loss, str)
        if not builtin_loss and (
            self.class_weight is not None or float(self.label_smoothing) > 0
        ):
            warnings.warn(
                "class_weight and label_smoothing only apply to the built-in "
                f"losses: they are ignored with loss={self.loss!r}.",
                UserWarning,
                stacklevel=2,
            )
        class_weight = self._compute_class_weight(y_encoded) if builtin_loss else None
        if class_weight is not None:
            class_weight = class_weight.to(device)
        # Per-sample losses are only needed when a weighting scheme has
        # to compose inside the reduction; the fully unweighted path
        # keeps the criterion's fused mean reduction (a scalar loss is
        # passed through unchanged by ``_weighted_loss_reduce``).
        needs_per_sample = sw_np is not None or class_weight is not None
        criterion = self._build_criterion(
            class_weight, reduction="none" if needs_per_sample else "mean"
        )
        if not builtin_loss and isinstance(criterion, nn.Module):
            criterion = criterion.to(device)

        X_val_t, y_val_t = self._collect_validation(val_loader, device)
        sw_val_t: torch.Tensor | None = (
            None
            if sw_val_np is None
            else torch.from_numpy(np.asarray(sw_val_np, dtype=np.float32)).to(device)
        )

        # Retry-on-collapse: when ``retry_on_val_auroc_below`` is set,
        # reseed and retrain up to ``max_retries`` more times if the
        # final val AUROC is below threshold.
        retry_threshold = self.retry_on_val_auroc_below
        max_retries = max(0, int(self.max_retries))
        total_attempts = 1 + (max_retries if retry_threshold is not None else 0)
        final_val_auroc = float("nan")
        base_seed = int(self.random_state)
        fitted_model: nn.Module | None = None
        freezer: TrunkFreezer | None = None
        for attempt in range(total_attempts):
            if attempt > 0:
                seed_everything(base_seed + 1_000_003 * attempt)

            if (
                warm_start
                and attempt == 0
                and getattr(self, "model_", None) is not None
            ):
                # Resume from the previously-fitted module. Skip
                # _build_model entirely so the federated / continual
                # learning caller's pre-loaded weights are not wiped.
                model = self.model_.to(device)
            else:
                model = self._build_initial_model(device, encoder)

            optimizer = build_optimizer(
                self.optimizer,
                self._optimizer_param_groups(model),
                learning_rate=float(self.learning_rate),
                weight_decay=float(self.weight_decay),
                use_sam=bool(self.use_sam),
                sam_rho=float(self.sam_rho),
            )
            scheduler = build_scheduler(
                self.scheduler,
                optimizer,
                epochs=int(self.epochs),
                warmup_epochs=int(self.warmup_epochs),
            )

            metrics_recorder = self._build_metrics_recorder(
                model, train_loader, X_val_t, y_val_t, device, sw_val_t
            )

            early = EarlyStopping(patience=int(self.early_stopping_patience))
            if int(self.epochs) > 0:
                dispatch(callbacks, "on_train_start", model, self)
            freezer = None
            if n_frozen > 0:
                freezer = TrunkFreezer(*self._trunk_target(model), n_frozen, early)
            compiled = _torch_compile(model) if bool(self.torch_compile) else None
            try:
                fitted_model = train_loop(
                    model,
                    train_loader,
                    (X_val_t, y_val_t, sw_val_t),
                    criterion,
                    optimizer,
                    scheduler,
                    device,
                    int(self.epochs),
                    early,
                    verbose=bool(self.verbose),
                    on_epoch_start=freezer,
                    warmup_epochs=int(self.warmup_epochs),
                    grad_clip_norm=(
                        float(self.grad_clip_norm)
                        if self.grad_clip_norm is not None
                        else None
                    ),
                    use_amp=bool(self.use_amp),
                    swa_start_epoch=(
                        int(self.swa_start_epoch)
                        if self.swa_start_epoch is not None
                        else None
                    ),
                    use_sam=bool(self.use_sam),
                    metrics_recorder=metrics_recorder,
                    augment=self.augment,
                    mixup_alpha=float(self.mixup_alpha),
                    cutmix_alpha=float(self.cutmix_alpha),
                    n_classes=int(self.n_classes_),
                    mix_generator=self._make_mix_generator(),
                    ema_decay=(
                        float(self.ema_decay) if self.ema_decay is not None else None
                    ),
                    class_weight_t=class_weight,
                    n_augment_cols=self._n_transform_cols(),
                    compiled_model=compiled,
                    amp_dtype=resolve_amp_dtype(str(self.amp_dtype), device),
                    callbacks=callbacks,
                )
                if freezer is not None and not freezer.unfroze:
                    # The trunk stayed frozen for the whole run: undo the
                    # float drift of EMA / SWA averaging and update_bn so
                    # it equals its initial weights exactly.
                    freezer.restore_initial_trunk()
            finally:
                if freezer is not None:
                    freezer.restore_flags()
            if int(self.epochs) > 0:
                dispatch(callbacks, "on_train_end", fitted_model)

            if retry_threshold is None or attempt == total_attempts - 1:
                break
            final_val_auroc = self._attempt_val_auroc(
                fitted_model, X_val_t, y_val_t, device, sw_val_t
            )
            if not np.isfinite(final_val_auroc) or final_val_auroc >= float(
                retry_threshold
            ):
                break
        assert fitted_model is not None
        self.model_ = fitted_model
        self._device_ = device
        self.trunk_unfroze_ = freezer is None or freezer.unfroze
        self.best_epoch_phase_ = self._best_epoch_phase(freezer, early)
        self.threshold_ = None
        self.temperature_ = None

        if self.metrics_log_path is not None:
            self._write_post_fit_sidecar(X_val_t, y_val_t, sw_val_t)

        if bool(self.tune_threshold) or bool(self.calibrate_temperature):
            self._fit_post_hoc_calibration(X_val_t, y_val_t, sw_val_t)

        return self

    def _attempt_val_auroc(
        self,
        model: nn.Module,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
        device: torch.device,
        sample_weight: torch.Tensor | None = None,
    ) -> float:
        """Return binary val AUROC of a freshly-fitted model, or NaN.

        Returns ``NaN`` for multi-class or when the val split has only
        one class present.
        """
        from sklearn.metrics import roc_auc_score

        model.eval()
        with torch.no_grad():
            logits = model(X_val)
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        y_np = y_val.detach().cpu().numpy()
        if probs.shape[1] != 2 or len(np.unique(y_np)) < 2:
            return float("nan")
        sw_np = None if sample_weight is None else sample_weight.detach().cpu().numpy()
        try:
            return float(roc_auc_score(y_np, probs[:, 1], sample_weight=sw_np))
        except ValueError:
            return float("nan")

    def _write_post_fit_sidecar(
        self,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
        sample_weight: torch.Tensor | None = None,
    ) -> None:
        """Compute val loss + AUROC on the deployed model and write a JSON sidecar."""
        from sklearn.metrics import roc_auc_score

        log_path = Path(self.metrics_log_path)
        sidecar_path = log_path.with_suffix(log_path.suffix + ".post_fit.json")

        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(X_val)
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
            y_np = y_val.detach().cpu().numpy()
            per_sample = nn.CrossEntropyLoss(reduction="none")(logits, y_val)
            if sample_weight is None:
                val_loss = float(per_sample.mean().item())
            else:
                sw_t = sample_weight.to(
                    dtype=per_sample.dtype, device=per_sample.device
                )
                val_loss = float(
                    ((sw_t * per_sample).sum() / sw_t.sum().clamp_min(1e-12)).item()
                )

        sw_np = None if sample_weight is None else sample_weight.detach().cpu().numpy()
        val_auroc: float | None = None
        n_classes = probs.shape[1]
        try:
            if n_classes == 2:
                val_auroc = float(roc_auc_score(y_np, probs[:, 1], sample_weight=sw_np))
            else:
                val_auroc = float(
                    roc_auc_score(
                        y_np,
                        probs,
                        multi_class="ovr",
                        average="macro",
                        sample_weight=sw_np,
                    )
                )
        except ValueError:
            pass

        if self.ema_decay is not None:
            source = "ema"
        elif self.swa_start_epoch is not None:
            source = "swa"
        else:
            source = "best_val"

        payload = {
            "val_loss": val_loss,
            "val_auroc": val_auroc,
            "weights_source": source,
            "n_classes": int(n_classes),
        }
        sidecar_path.write_text(json.dumps(payload, indent=2))

    def _build_metrics_recorder(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        X_val_t: torch.Tensor,
        y_val_t: torch.Tensor,
        device: torch.device,
        sw_val_t: torch.Tensor | None = None,
    ) -> Callable[[dict[str, float]], None] | None:
        """Return a per-epoch recorder that appends diagnostics to a CSV.

        Returns ``None`` (no recording) when ``metrics_log_path`` is unset.
        """
        if self.metrics_log_path is None:
            return None

        log_path = Path(self.metrics_log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if log_path.exists():
            log_path.unlink()

        track_train = bool(self.track_train_metrics)
        binary = self.n_classes_ == 2

        sw_val_np_recorder = (
            None if sw_val_t is None else sw_val_t.detach().cpu().numpy()
        )

        def _collect_train_val_auroc() -> tuple[float, float]:
            from sklearn.metrics import roc_auc_score

            if not binary:
                import math

                return math.nan, math.nan
            y_val_np = y_val_t.detach().cpu().numpy()
            y_tr_parts: list[np.ndarray] = []
            proba_tr_parts: list[np.ndarray] = []
            sw_tr_parts: list[np.ndarray] = []
            has_sw = False
            model.eval()
            with torch.no_grad():
                for batch in train_loader:
                    if len(batch) == 3:
                        xb, yb, wb = batch
                        has_sw = True
                        sw_tr_parts.append(np.asarray(wb).ravel())
                    else:
                        xb, yb = batch
                    xb = xb.to(device, non_blocking=True)
                    logits = model(xb).detach()
                    proba = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                    proba_tr_parts.append(proba)
                    y_tr_parts.append(np.asarray(yb).ravel())
                val_logits = model(X_val_t).detach()
                val_proba = torch.softmax(val_logits, dim=-1)[:, 1].cpu().numpy()
            model.train()
            y_tr = np.concatenate(y_tr_parts)
            proba_tr = np.concatenate(proba_tr_parts)
            sw_tr_arr = np.concatenate(sw_tr_parts) if has_sw else None
            try:
                train_auroc = float(
                    roc_auc_score(y_tr, proba_tr, sample_weight=sw_tr_arr)
                )
            except ValueError:
                train_auroc = float("nan")
            try:
                val_auroc = float(
                    roc_auc_score(y_val_np, val_proba, sample_weight=sw_val_np_recorder)
                )
            except ValueError:
                val_auroc = float("nan")
            return train_auroc, val_auroc

        header_written = False

        def recorder(payload: dict[str, float]) -> None:
            nonlocal header_written
            row: dict[str, float | int] = dict(payload)
            if track_train:
                train_auroc, val_auroc = _collect_train_val_auroc()
                row["train_auroc"] = train_auroc
                row["val_auroc"] = val_auroc
            columns = [
                "epoch",
                "train_loss",
                "val_loss",
                "lr",
                "mean_grad_norm",
                "n_grad_updates",
            ]
            if track_train:
                columns += ["train_auroc", "val_auroc"]
            with open(log_path, "a") as fh:
                if not header_written:
                    fh.write(",".join(columns) + "\n")
                    header_written = True
                fh.write(",".join(str(row.get(k, "")) for k in columns) + "\n")

        return recorder

    def _fit_post_hoc_calibration(
        self,
        X_val_t: torch.Tensor,
        y_val_t: torch.Tensor,
        sample_weight: torch.Tensor | None = None,
    ) -> None:
        """Collect held-out logits once, then fit threshold / temperature."""
        from ..utils.calibration import fit_temperature, tune_threshold

        self.model_.eval()
        with torch.no_grad():
            val_logits = self.model_(X_val_t).detach().cpu()
        y_val_np = y_val_t.detach().cpu().numpy()
        sw_val_np = (
            None if sample_weight is None else sample_weight.detach().cpu().numpy()
        )

        if bool(self.calibrate_temperature):
            self.temperature_ = float(
                fit_temperature(val_logits, y_val_np, sample_weight=sw_val_np)
            )

        if bool(self.tune_threshold):
            if self.n_classes_ != 2:
                self.threshold_ = None
            else:
                from sklearn.metrics import roc_auc_score

                logits_np = val_logits.numpy()
                if self.temperature_ is not None:
                    logits_np = logits_np / float(self.temperature_)
                logits_np = logits_np - logits_np.max(axis=1, keepdims=True)
                exp = np.exp(logits_np)
                proba = exp / exp.sum(axis=1, keepdims=True)
                try:
                    val_auroc = float(
                        roc_auc_score(y_val_np, proba[:, 1], sample_weight=sw_val_np)
                    )
                except ValueError:
                    val_auroc = float("nan")
                gate = float(self.min_val_auroc_for_threshold_tune)
                if np.isfinite(val_auroc) and val_auroc >= gate:
                    self.threshold_ = float(
                        tune_threshold(
                            y_val_np,
                            proba[:, 1],
                            metric=self.threshold_metric,
                            sample_weight=sw_val_np,
                        )
                    )
                else:
                    import logging

                    logging.getLogger(__name__).info(
                        "tune_threshold skipped: val AUROC=%.3f < %.2f; "
                        "falling back to threshold_=0.5",
                        val_auroc,
                        gate,
                    )
                    self.threshold_ = 0.5

    @staticmethod
    def _collect_validation(
        val_loader: DataLoader, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        xs, ys = [], []
        for xb, yb in val_loader:
            xs.append(xb)
            ys.append(yb)
        X_val = torch.cat(xs, dim=0).to(device)
        y_val = torch.cat(ys, dim=0).to(device)
        return X_val, y_val

    def _n_transform_cols(self) -> int | None:
        """Return how many leading columns spectrum preprocessing applies to.

        ``None`` (the default) means "every column", which is correct
        for single-input classifiers. Dual-input subclasses override
        this to return their spectrum width so that ``warping``,
        ``standardize``, ``input_transform``, ``augment``, and CutMix
        never touch the packed side-feature block. See
        :class:`~maldideepkit.BaseDualSpectralClassifier`.
        """
        return None

    def _check_input_dim(self, X: np.ndarray) -> None:
        if X.shape[1] != self.input_dim_:
            raise ValueError(
                f"X has {X.shape[1]} features but estimator was fitted with "
                f"input_dim={self.input_dim_}. Retrain or re-bin your data "
                "to match the original resolution."
            )

    def _preprocess_input(self, X: Any) -> np.ndarray:
        """Apply the fitted inference-time preprocessing chain to ``X``.

        Runs optional spectral warping, the fitted ``input_transform``
        state, and the legacy ``standardize`` path (each confined to
        the leading :meth:`_n_transform_cols` columns so dual-input
        models never transform their packed side block) through
        :func:`~maldideepkit.base.data.apply_fitted_preprocessing`.
        This is the single entry point for inference preprocessing:
        both :meth:`_forward_logits` and the estimators in
        :mod:`maldideepkit.uncertainty` go through it.

        Parameters
        ----------
        X : array-like or MaldiSet of shape (n_samples, n_bins)
            Raw input spectra (plus packed side features for
            dual-input models).

        Returns
        -------
        ndarray of shape (n_samples, n_bins)
            Preprocessed feature matrix as ``float32``.
        """
        X_np = _to_numpy(X)
        self._check_input_dim(X_np)
        return apply_fitted_preprocessing(
            X_np,
            warper=getattr(self, "warper_", None),
            input_transform_state=getattr(self, "input_transform_state_", None),
            standardize=bool(self.standardize),
            feature_mean=getattr(self, "feature_mean_", None),
            feature_std=getattr(self, "feature_std_", None),
            n_transform_cols=self._n_transform_cols(),
        )

    def _forward_logits(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "model_")
        X_np = self._preprocess_input(X)
        device = self._device_
        self.model_.eval()
        # Batch inference so large test folds do not run out of memory
        # on attention architectures.
        X_t_full = torch.from_numpy(X_np.astype(np.float32)).to(device)
        chunk = int(getattr(self, "batch_size", 32))
        chunk = max(1, chunk)
        logits_chunks: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, X_t_full.shape[0], chunk):
                X_t = X_t_full[start : start + chunk]
                logits_chunks.append(self.model_(X_t).detach().cpu().numpy())
        logits = (
            np.concatenate(logits_chunks, axis=0)
            if logits_chunks
            else np.empty((0, self.n_classes_), dtype=np.float32)
        )
        if logits.ndim == 1:
            logits = logits.reshape(-1, 1)
        return logits

    def get_embeddings(self, X: Any) -> np.ndarray:
        """Return the embeddings that the final layer of the model reads.

        The counterpart of
        :meth:`maldideepkit.pretraining.MaldiSpectrumEncoder.transform` for
        a fitted classifier: the fitted preprocessing is applied first.
        For a dual-input classifier these are the fused features of the
        two branches.

        Parameters
        ----------
        X : array-like or MaldiSet of shape (n_samples, n_bins)
            Raw input spectra (plus packed side features for dual-input
            models).

        Returns
        -------
        ndarray of shape (n_samples, feature_dim)
            ``float32`` embeddings (``eval`` mode, batched by
            ``batch_size``).
        """
        check_is_fitted(self, "model_")
        X_np = self._preprocess_input(X)
        split = split_trunk(self.model_, int(self.input_dim_))
        self.model_.eval()
        X_t = torch.from_numpy(X_np.astype(np.float32))
        chunk = max(1, int(self.batch_size))
        parts: list[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, X_t.shape[0], chunk):
                rows = X_t[start : start + chunk].to(self._device_)
                parts.append(split.features(rows).float())
        if not parts:
            return np.empty((0, split.feature_dim), dtype=np.float32)
        return torch.cat(parts).cpu().numpy()

    @classmethod
    def from_spectrum(
        cls, bin_width: int, input_dim: int, **overrides: Any
    ) -> BaseSpectralClassifier:
        """Construct a classifier for a given ``(bin_width, input_dim)`` layout.

        The default forwards ``input_dim`` and ``**overrides`` only. The
        classifiers whose architecture depends on the bin width override
        it to scale their parameters.

        Parameters
        ----------
        bin_width : int
            Bin width in Daltons (e.g. 3 for the MaldiAMRKit default).
        input_dim : int
            Number of bins in the input.
        **overrides
            Any other constructor argument.

        Returns
        -------
        BaseSpectralClassifier
            An unfitted estimator.
        """
        del bin_width
        return cls(**{"input_dim": input_dim, **overrides})

    def predict_proba(self, X: Any) -> np.ndarray:
        """Return softmax class probabilities of shape ``(n_samples, n_classes)``.

        Parameters
        ----------
        X : array-like or MaldiSet of shape (n_samples, n_bins)
            Spectra to score. Must have the same number of features as
            the training matrix.

        Returns
        -------
        ndarray of shape (n_samples, n_classes)
            Softmax probabilities that sum to 1 along the class axis.

        Raises
        ------
        ValueError
            If ``X.shape[1] != input_dim_``.
        """
        logits = self._forward_logits(X)
        temperature = getattr(self, "temperature_", None)
        if temperature is not None:
            logits = logits / float(temperature)
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / exp.sum(axis=1, keepdims=True)

    def predict(self, X: Any) -> np.ndarray:
        """Return hard class predictions.

        Parameters
        ----------
        X : array-like or MaldiSet of shape (n_samples, n_bins)
            Spectra to classify.

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted labels, drawn from :attr:`classes_`.

        Notes
        -----
        For binary classifiers fit with ``tune_threshold=True``, the
        decision uses the fitted :attr:`threshold_` on the positive
        class probability instead of ``argmax``.
        """
        proba = self.predict_proba(X)
        threshold = getattr(self, "threshold_", None)
        if threshold is not None and proba.shape[1] == 2:
            idx = (proba[:, 1] >= float(threshold)).astype(int)
        else:
            idx = np.argmax(proba, axis=1)
        return self.classes_[idx]

    def score(self, X: Any, y: Any) -> float:
        """Return mean accuracy on ``(X, y)``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_bins)
        y : array-like of shape (n_samples,)

        Returns
        -------
        float
            Accuracy in ``[0, 1]``.
        """
        if hasattr(y, "to_numpy"):
            y = y.to_numpy()
        y = np.asarray(y).ravel()
        preds = self.predict(X)
        return float(np.mean(preds == y))

    def _hparam_dict(self) -> dict[str, Any]:
        params = self.get_params(deep=False)
        if isinstance(params.get("device"), torch.device):
            params["device"] = str(params["device"])
        if isinstance(params.get("class_weight"), np.ndarray):
            params["class_weight"] = params["class_weight"].tolist()
        params["warping"] = None if params.get("warping") is None else "<provided>"
        # A checkpoint path is stored absolute: the one resolved at fit
        # time (``init_from_provenance``) when available, so a later
        # change of working directory cannot redirect it. An in-memory
        # encoder cannot be serialised (see ``init_from_provenance`` in
        # save()).
        provenance = getattr(self, "_init_from_provenance", None) or {}
        fit_path = provenance.get("path")
        init_from = params.get("init_from")
        if init_from is not None:
            params["init_from"] = (
                (fit_path or str(Path(init_from).expanduser().resolve()))
                if isinstance(init_from, (str, os.PathLike))
                else None
            )
        for key, value in params.items():
            if isinstance(value, BaseSpectralClassifier):
                # Estimator-valued hyperparameters (e.g. the dual-branch
                # ``backbone``) are stored as a reconstructable spec.
                sub_params = value._hparam_dict()
                if (
                    init_from is None
                    and fit_path is not None
                    and isinstance(sub_params.get("init_from"), str)
                ):
                    # The dual-branch classifier resolves its template's
                    # ``init_from`` and keeps the provenance itself.
                    sub_params["init_from"] = fit_path
                params[key] = _estimator_spec(value, sub_params)
            else:
                # A SpectrumAugment ``augment`` is stored as its class
                # path plus constructor parameters.
                params[key] = _encode_param(key, value)
        return params

    def _saved_class_path(self) -> str | None:
        """Return the import path for :meth:`load`, or ``None`` if it cannot be used.

        Only ``BaseSpectralClassifier.load`` needs it: ``Cls.load`` knows
        the class, so a class that cannot be re-imported (a local class,
        a notebook cell run again) still saves.
        """
        try:
            return _importable_class_path(
                type(self), BaseSpectralClassifier, "classifier", "class"
            )
        except ValueError:
            return None

    def save(self, path: str | Path) -> None:
        """Persist the fitted estimator to ``path.pt`` + ``path.json``.

        The PyTorch state dict is written to ``<path>.pt`` and the
        hyperparameters plus fitted metadata to ``<path>.json``. A
        single ``.pt``, ``.pth``, ``.json``, or ``.warper.pkl`` suffix
        on ``path`` is stripped so ``clf.save("model")`` and
        ``clf.save("model.pt")`` produce the same pair of files; any
        other dot is kept, so ``clf.save("lr0.05")`` writes
        ``lr0.05.pt``. When the last :meth:`fit` used
        ``init_from``, ``meta["init_from_provenance"]`` records the
        encoder checkpoint (absolute path and sha256 of its ``.pt``,
        both ``None`` for an in-memory encoder); it is ``None``
        otherwise.

        An object hyperparameter (``loss``, ``optimizer``, ``scheduler``,
        ``callbacks``, ``augment``) is stored by import path, plus
        constructor parameters for an instance: its class or function
        must be defined at module level in an importable module (a lambda
        cannot be saved), and an instance must keep each required
        constructor argument in an attribute of the same name. Clear a
        parameter that cannot be saved, e.g.
        ``set_params(callbacks=None)``, before :meth:`save`; the weights
        do not depend on it. The metadata is serialised before any file
        is written and every file goes through a temporary name, so a
        failing :meth:`save` writes nothing.

        Parameters
        ----------
        path : str or Path
            Base path without extension.

        Raises
        ------
        TypeError
            If an object hyperparameter does not keep a required
            constructor argument in an attribute of the same name.
        ValueError
            If the class or function of a hyperparameter cannot be
            re-imported from its import path (e.g. a local class or a
            lambda).
        """
        check_is_fitted(self, "model_")
        warper = getattr(self, "warper_", None)
        meta: dict[str, Any] = {
            "class_name": type(self).__name__,
            "class_path": self._saved_class_path(),
            "version": 2,
            "hparams": self._hparam_dict(),
            "fitted": {
                "input_dim_": int(self.input_dim_),
                "n_classes_": int(self.n_classes_),
                "classes_": np.asarray(self.classes_).tolist(),
                "n_features_in_": int(self.n_features_in_),
                "feature_mean_": (
                    None
                    if self.feature_mean_ is None
                    else np.asarray(self.feature_mean_).tolist()
                ),
                "feature_std_": (
                    None
                    if self.feature_std_ is None
                    else np.asarray(self.feature_std_).tolist()
                ),
                "threshold_": getattr(self, "threshold_", None),
                "temperature_": getattr(self, "temperature_", None),
                "has_warper": warper is not None,
                "input_transform_state_": _serialise_transform_state(
                    getattr(self, "input_transform_state_", None)
                ),
                "trunk_unfroze_": getattr(self, "trunk_unfroze_", None),
                "best_epoch_phase_": getattr(self, "best_epoch_phase_", None),
            },
            "init_from_provenance": getattr(self, "_init_from_provenance", None),
        }
        _write_checkpoint(
            _checkpoint_base(path),
            self.model_.state_dict(),
            json.dumps(meta, indent=2),
            warper,
        )

    @classmethod
    def load(cls, path: str | Path, *, trusted: bool = False) -> BaseSpectralClassifier:
        """Load a saved estimator from a ``save()``-produced pair of files.

        The saved weights are restored as they are: ``init_from`` is
        kept as a parameter but never re-applied. A classifier saved
        with a fitted warper keeps its ``warping`` as an unfitted copy
        of the loaded warper (``None`` when that warper cannot be
        cloned). Checkpoints written by v0.3.0 under a dotted base name
        (``save("lr0.05")`` wrote ``lr0.pt`` there) still load: when
        ``<path>.pt`` does not exist but the v0.3.0 file name does and
        its ``.json`` was written before v0.4.0 (it has no
        ``"init_from_provenance"`` key), the latter is used and a
        ``UserWarning`` naming it is emitted. A v0.4.0 file under the
        v0.3.0 name is another checkpoint, so the requested name raises
        :class:`FileNotFoundError`.

        Parameters
        ----------
        path : str or Path
            Base path given to :meth:`save` (a single ``.pt``, ``.pth``,
            ``.json``, or ``.warper.pkl`` suffix is stripped).
        trusted : bool, default=False
            With the default, no new module is imported: every class
            named by the checkpoint must come from MaldiDeepKit or from
            an imported module. An object given as ``optimizer``,
            ``scheduler``, ``loss`` or ``callbacks`` must also be a torch
            module, optimizer or scheduler, or a ``Callback``, and a
            function must be defined in MaldiDeepKit or in
            ``torch.nn.functional``. ``True`` lifts these limits, for
            checkpoints from a trusted source.

        Returns
        -------
        BaseSpectralClassifier
            Fitted estimator ready for :meth:`predict` /
            :meth:`predict_proba`.

        Raises
        ------
        ValueError
            If the JSON file identifies a different class from ``cls``,
            or names a class or a function that the ``trusted`` rule
            refuses or that cannot be imported.
        FileNotFoundError
            If the ``.pt``, the ``.json``, or a required ``.warper.pkl``
            is missing.

        Notes
        -----
        A ``.warper.pkl`` is a pickle, read whatever ``trusted`` is:
        load checkpoints with a warper only from trusted sources.
        """
        base = _checkpoint_base(path)
        pt_path = _checkpoint_file(base, ".pt")
        json_path = _checkpoint_file(base, ".json")
        warper_path = _checkpoint_file(base, ".warper.pkl")
        legacy_pt = base.with_suffix(".pt")
        meta: dict[str, Any] | None = None
        if not pt_path.exists() and legacy_pt != pt_path and legacy_pt.exists():
            # v0.3.0 named the files with ``with_suffix``, so a dotted
            # base such as ``"model_0.05"`` was written as ``model_0.pt``.
            # Every later classifier save writes ``init_from_provenance``
            # and every encoder save ``kind``: such a file is another
            # checkpoint, not the requested one.
            legacy_json = base.with_suffix(".json")
            if not legacy_json.exists():
                raise FileNotFoundError(pt_path)
            with open(legacy_json) as fh:
                meta = json.load(fh)
            if "init_from_provenance" in meta or "kind" in meta:
                raise FileNotFoundError(pt_path)
            warnings.warn(
                f"{pt_path} does not exist; loading the v0.3.0 checkpoint "
                f"{legacy_pt} instead (v0.3.0 dropped the last dotted part of "
                "the name). Save it again to store it under the requested name.",
                UserWarning,
                stacklevel=2,
            )
            pt_path = legacy_pt
            json_path = legacy_json
            warper_path = base.with_suffix(".warper.pkl")
        if not pt_path.exists():
            raise FileNotFoundError(pt_path)
        if not json_path.exists():
            raise FileNotFoundError(json_path)

        if meta is None:
            with open(json_path) as fh:
                meta = json.load(fh)

        if cls is not BaseSpectralClassifier and meta["class_name"] != cls.__name__:
            raise ValueError(
                f"Saved class is {meta['class_name']!r} but load() was called "
                f"on {cls.__name__!r}."
            )

        registry = _classifier_registry()
        target_cls = cls
        if cls is BaseSpectralClassifier:
            if meta["class_name"] in registry:
                target_cls = registry[meta["class_name"]]
            elif meta.get("class_path") is not None:
                target_cls = _import_class(
                    meta["class_path"],
                    BaseSpectralClassifier,
                    "classifier",
                    trusted=trusted,
                )
            else:
                raise ValueError(f"Unknown saved class: {meta['class_name']!r}.")

        hparams = _decode_hparams(dict(meta["hparams"]), registry, trusted=trusted)
        instance = target_cls(**hparams)
        fitted = meta["fitted"]
        instance.input_dim_ = int(fitted["input_dim_"])
        instance.n_classes_ = int(fitted["n_classes_"])
        instance.classes_ = np.asarray(fitted["classes_"])
        instance.n_features_in_ = int(fitted["n_features_in_"])
        instance.feature_mean_ = (
            None
            if fitted["feature_mean_"] is None
            else np.asarray(fitted["feature_mean_"], dtype=np.float32)
        )
        instance.feature_std_ = (
            None
            if fitted["feature_std_"] is None
            else np.asarray(fitted["feature_std_"], dtype=np.float32)
        )
        instance.threshold_ = fitted.get("threshold_")
        instance.temperature_ = fitted.get("temperature_")
        # Absent from checkpoints written before v0.4.0.
        for name in ("trunk_unfroze_", "best_epoch_phase_"):
            if fitted.get(name) is not None:
                setattr(instance, name, fitted[name])
        instance._init_from_provenance = meta.get("init_from_provenance")
        instance.input_transform_state_ = _deserialise_transform_state(
            fitted.get("input_transform_state_")
        )
        instance.warper_ = None
        if fitted.get("has_warper"):
            import joblib

            if not warper_path.exists():
                raise FileNotFoundError(warper_path)
            instance.warper_ = joblib.load(warper_path)
            # Refit and clone read ``warping``, which the JSON cannot hold:
            # restore it as an unfitted copy of the loaded warper. A warper
            # that cannot be cloned (fitted by an older maldiamrkit) leaves
            # it at ``None``, as before.
            try:
                instance.warping = clone(instance.warper_)
            except (AttributeError, TypeError):
                instance.warping = None

        device = resolve_device(instance.device)
        model = instance._build_model().to(device)
        state = torch.load(pt_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.eval()
        instance.model_ = model
        instance._device_ = device
        return instance

    def __sklearn_is_fitted__(self) -> bool:
        return hasattr(self, "model_")

    def __sklearn_tags__(self):  # pragma: no cover (scikit-learn >= 1.6)
        try:
            tags = super().__sklearn_tags__()
        except AttributeError:
            return None
        tags.input_tags.two_d_array = True
        tags.input_tags.sparse = False
        tags.classifier_tags.multi_class = True
        tags.classifier_tags.poor_score = True
        tags.non_deterministic = False
        return tags

    def _more_tags(self) -> dict[str, Any]:  # pragma: no cover (scikit-learn < 1.6)
        return {
            "binary_only": False,
            "multioutput": False,
            "poor_score": True,
            "requires_positive_X": False,
            "X_types": ["2darray"],
        }


__all__ = ["BaseSpectralClassifier", "SpectralDataset"]
