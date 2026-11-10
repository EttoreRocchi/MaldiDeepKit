"""Self-supervised spectrum encoder wrapping a MaldiDeepKit backbone."""

from __future__ import annotations

import inspect
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.base import (
    BaseEstimator,
    ClassNamePrefixFeaturesOutMixin,
    TransformerMixin,
    clone,
)
from sklearn.utils.validation import check_is_fitted
from torch import nn

from ..augment.spectra import SpectrumAugment
from ..base._config import same_config
from ..base._registry import get_backbone_class
from ..base._serialization import (
    _class_path,
    _encode_param,
    _estimator_spec,
    _import_class,
)
from ..base._trunk import split_trunk
from ..base.classifier import (
    BaseSpectralClassifier,
    _checkpoint_base,
    _checkpoint_file,
    _classifier_registry,
    _decode_hparams,
    _deserialise_transform_state,
    _serialise_transform_state,
    _torch_compile,
    _write_checkpoint,
)
from ..base.data import (
    _to_numpy,
    _warp_numpy,
    apply_fitted_preprocessing,
    resolve_transform_mode,
)
from ..base.dual import BaseDualSpectralClassifier
from ..utils.callbacks import callback_list, dispatch
from ..utils.optim import build_optimizer, check_scheduler, resolve_optimizer
from ..utils.reproducibility import resolve_device, seed_everything
from ..utils.training import AMP_DTYPES, EarlyStopping, resolve_amp_dtype
from ._pretext import (
    Pretext,
    SSLModel,
    _check_augment,
    _check_pretext_params,
    get_pretext_class,
)
from ._training import SSLBatch, _FittedTransform, ssl_train_loop
from .data import AUX_TARGETS, make_pretraining_loaders

_CHECKPOINT_KIND = "encoder"
_CHECKPOINT_VERSION = 1

_DATA_DEVICE_MEMORY_FRACTION = 0.5

_PRETEXT_PARAMS = (
    "mask_ratio",
    "shuffle_fraction",
    "projection_dim",
    "temperature",
    "augment",
)


def _estimator_registry(*specs: Any, trusted: bool = False) -> dict[str, type]:
    """Return the classifier registry extended with the classes of ``specs``.

    A spec naming a built-in class (same name and import path) uses the
    registry; any other class is imported from its ``"__class__"`` path
    (``trusted`` is passed to
    :func:`~maldideepkit.base._serialization._import_class`).
    """
    registry = dict(_classifier_registry())
    for spec in specs:
        if not (isinstance(spec, dict) and "__estimator__" in spec):
            continue
        name, path = spec["__estimator__"], spec.get("__class__")
        known = registry.get(name)
        if path is None or (known is not None and _class_path(known) == path):
            continue
        registry[name] = _import_class(
            path, BaseSpectralClassifier, "backbone", trusted=trusted
        )
    return registry


class MaldiSpectrumEncoder(
    ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator
):
    """Self-supervised pre-training of a MaldiDeepKit backbone on binned spectra.

    Wraps the module of a MaldiDeepKit classifier, trains it with a
    self-supervised pre-text task on unlabeled (or partially labeled)
    spectra, and exposes the learned embeddings
    (:meth:`transform`), the fitted preprocessing state and a
    checkpoint (:meth:`save` / :meth:`load`). The trunk (everything but
    the backbone's ``final_layer``) is what a classifier later
    warm-starts from.

    Preprocessing (``standardize``, ``input_transform``, ``warping``)
    is taken from the backbone template: a string backbone uses the
    class defaults (e.g. ``"resnet"`` implies
    ``input_transform="log1p"``), an estimator template uses its own
    settings. The template's preprocessing is fitted on the training
    split of the pre-training pool. The template's ``init_from`` and
    ``freeze_trunk_epochs`` are ignored (pre-training starts from a
    random backbone and trains every layer), with a ``UserWarning``
    when either is set.

    Parameters
    ----------
    backbone : str or BaseSpectralClassifier, default="mlp"
        ``"mlp"``, ``"cnn"``, ``"resnet"``, ``"transformer"``, or a name
        added with :func:`~maldideepkit.pretraining.register_backbone`
        (class defaults), or an unfitted single-input classifier
        instance used as an architecture and preprocessing template (it
        is cloned, never modified). ``"peakformer"`` is reserved for the peak-set
        transformer of MaldiDeepKit v0.5.0.
    pretext : str or Pretext, default="mae"
        Self-supervised task: a registered name or a
        :class:`~maldideepkit.pretraining.Pretext` instance. Every task
        works on the input-transformed spectra and adds a head on the
        backbone embedding (``forward_features``).
        The built-in names are:

        - ``"mae"`` (:class:`~maldideepkit.pretraining.MAEPretext`):
          ``mask_ratio`` of the bins, drawn uniformly, are set to
          ``0``; a per-bin decoder ``Linear(feature_dim -> input_dim)``
          reconstructs them (MSE on the masked bins). With an input
          transform that leaves the bins on the raw intensity scale
          (``"log1p"``, the ``"resnet"`` default, or ``"none"``) the
          target is the model input standardized per bin with the
          training-split statistics (:attr:`recon_mean_`,
          :attr:`recon_std_`).
        - ``"shuffle"`` (:class:`~maldideepkit.pretraining.ShufflePretext`):
          ``mask_ratio`` of the bins are sampled without replacement
          with probability proportional to the raw (pre-transform)
          intensity; ``shuffle_fraction`` of them take the value of the
          same bin in another spectrum of the batch. A per-bin
          discriminator ``Linear(feature_dim -> input_dim)`` predicts
          which sampled bins are "noise" (BCE on the sampled bins); a
          swap that leaves the value unchanged (e.g. a zero for a zero)
          is labeled "real".
        - ``"shuffle+mae"``
          (:class:`~maldideepkit.pretraining.ShuffleMAEPretext`): both
          tasks with their own heads and corruptions, losses summed.
        - ``"contrastive"``
          (:class:`~maldideepkit.pretraining.ContrastivePretext`): two
          views per spectrum (``augment`` on the raw spectra, then the
          input transform and ``"mae"``-style masking), a projection
          head (``projection_dim``) and the NT-Xent loss
          (``temperature``) with the other spectra of the batch as
          negatives. A ``UserWarning`` is emitted when
          ``batch_size < 64`` (few negatives).

        A name is resolved with
        :func:`~maldideepkit.pretraining.get_pretext_class` (names added
        by :func:`~maldideepkit.pretraining.register_pretext` included)
        and built from the encoder-level pre-text parameters below
        (``mask_ratio``, ``shuffle_fraction``, ``projection_dim``,
        ``temperature``, ``augment``) that its class accepts. A
        :class:`~maldideepkit.pretraining.Pretext` instance holds its
        own parameters (searchable as ``pretext__<param>``) and the
        encoder-level ones are ignored, with a ``UserWarning`` when any
        differs from its default.
    mask_ratio : float, default=0.15
        Fraction of bins masked (``"mae"``, ``"contrastive"``) or
        sampled (``"shuffle"``) per spectrum, in ``(0, 1]``.
    shuffle_fraction : float, default=0.5
        Fraction of the sampled bins that are swapped by the
        ``"shuffle"`` pre-texts, in ``[0, 1]`` (the others are kept and
        labeled "real").
    projection_dim : int, default=128
        Output width of the contrastive projection head
        (``feature_dim -> feature_dim -> projection_dim`` MLP).
    temperature : float, default=0.1
        NT-Xent temperature of the contrastive pre-text (``> 0``).
    augment : SpectrumAugment or None, default=None
        Augmentation of the raw spectra drawing the two contrastive
        views, as the classifiers' ``augment``. ``None`` uses the
        default contrastive augmentation (a per-spectrum intensity
        jitter ``1 + U(-0.2, 0.2)`` and a global m/z shift of up to 1
        bin). Every batch uses a copy re-seeded from the corruption
        generator (the instance is never modified and its
        ``random_state`` is ignored), so the validation loss uses the
        same views every epoch. Only used by ``"contrastive"``: a
        ``UserWarning`` is emitted when it is set with another
        pre-text.
    aux_lambda : float, default=0.0
        Weight of the semi-supervised auxiliary loss: the objective is
        ``L_pretext + aux_lambda * L_aux``. ``0`` disables the
        auxiliary head; ``> 0`` adds it and uses ``y`` in :meth:`fit`.
        The auxiliary head reads a second, clean (uncorrupted) forward
        pass of the backbone, run in train mode, so BatchNorm running
        statistics are updated by both the corrupted pre-text batch and
        the clean batch.
    aux_target : {"classification", "regression"}, default="classification"
        Type of the auxiliary target (used when ``aux_lambda > 0``).
        ``"classification"``: ``Linear(feature_dim -> n_classes)`` and
        cross-entropy, classes of any dtype (e.g. strings) encoded in
        :attr:`aux_classes_`. ``"regression"``:
        ``Linear(feature_dim -> 1)`` and MSE on ``float32`` targets.
        The head is trained on the embedding of the uncorrupted
        spectra, labeled rows only; a batch without labeled rows
        contributes ``0``.
    unlabeled_marker : object, default=-1
        Value of ``y`` that marks an unlabeled row. Missing values
        (``np.nan``, ``None``) are always unlabeled, so
        ``unlabeled_marker=np.nan`` relies on them only. With
        ``aux_target="regression"`` a numeric marker emits a
        ``UserWarning`` (it may be a valid value, e.g. a log2-MIC).
    learning_rate : float, default=1e-3
        Peak learning rate (Adam; AdamW when ``weight_decay > 0``).
    weight_decay : float, default=0.0
        Decoupled weight decay.
    batch_size : int, default=256
        Mini-batch size for training and :meth:`transform`.
    epochs : int, default=100
        Maximum number of pre-training epochs. ``0`` fits the
        preprocessing state and builds a randomly initialised backbone
        without any self-supervised training.
    warmup_epochs : int, default=5
        Linear learning-rate warmup length, followed by cosine
        annealing.
    grad_clip_norm : float or None, default=1.0
        Global gradient-norm clipping threshold.
    early_stopping_patience : int, default=10
        Number of epochs without validation-loss improvement before
        training stops.
    val_fraction : float, default=0.1
        Fraction of the pool held out (random split) for the
        validation loss.
    use_amp : bool, default=False
        Mixed precision on a GPU (no-op on CPU).
    amp_dtype : {"auto", "float16", "bfloat16"}, default="auto"
        Autocast dtype of ``use_amp=True``. ``"auto"`` picks
        ``"bfloat16"`` when the GPU supports it, else ``"float16"``
        (with :class:`torch.amp.GradScaler`). ``bfloat16`` keeps the
        exponent range of ``float32``: in ``float16`` a scale-free
        layer whose weights grow during pre-training (e.g. a ``Linear``
        followed by ``LayerNorm``, unconstrained without
        ``weight_decay``) can overflow on some spectra and turn the
        loss into ``NaN``.
    device : {"auto", "cpu", "cuda"} or torch.device, default="auto"
        Device used for training and inference.
    random_state : int, default=0
        Seeds the RNGs, the split, and the corruption.
    verbose : bool, default=False
        If ``True``, print one line per epoch.
    data_device : {"auto", "cpu", "cuda"} or torch.device, default="auto"
        Where the pre-training pool lives during :meth:`fit`.
        ``"cuda"`` (the training device): the (warped) pool is copied
        to the GPU once, in row chunks, every batch is gathered and
        input-transformed there (by the PyTorch twin of the transform,
        equal to the NumPy values up to ``float32`` rounding) and
        nothing but the batch row indices crosses the host-device bus.
        ``"cpu"``: the pool stays on the host (a memory-mapped pool on
        disk) and every batch is read at once, transformed in NumPy,
        and copied to the device (from pinned memory on a GPU).
        ``"auto"``: ``"cuda"`` when training on a GPU and the pool
        (``n_samples * n_bins`` ``float32`` values, plus the targets
        and the transform statistics) takes at most half of the GPU
        memory free at the start of :meth:`fit`
        (:func:`torch.cuda.mem_get_info`), else ``"cpu"``; an upload
        that still runs out of GPU memory falls back to ``"cpu"`` with
        a ``UserWarning`` (an explicit ``"cuda"`` raises). Both paths
        draw the same batches (same sampler, same global RNG
        consumption), and on a GPU batch the corruptions are drawn on
        the GPU either way. Warping, if any, is applied once to the
        whole pool on the host first. Without training
        (``epochs=0``) nothing is uploaded. :meth:`transform` also
        streams its batches to the device and transforms them there,
        unless ``data_device="cpu"``. The chosen path is stored in
        :attr:`data_device_`.
    torch_compile : bool, default=False
        If ``True``, the backbone's feature extractor
        (``model_.forward_features``) runs through ``torch.compile``
        during :meth:`fit` (static shapes: one compilation per batch
        shape and train / eval mode, done on the first batches). The
        compiled function only lives during :meth:`fit`: :attr:`model_`,
        the :meth:`save` checkpoint, the ``init_from`` transfer and
        :meth:`transform` use the plain module (no ``_orig_mod.`` key
        prefix), and pickling or :func:`~sklearn.base.clone` never see
        a compiled object.
    optimizer : None, str or callable, default=None
        As on the classifiers (see
        :class:`~maldideepkit.BaseSpectralClassifier`): ``None`` keeps
        Adam (AdamW when ``weight_decay > 0``).
    scheduler : None, str or callable, default=None
        As on the classifiers: ``None`` keeps the warmup plus cosine
        schedule.
    callbacks : Callback, sequence of Callback or None, default=None
        As on the classifiers; their hooks run after those of the
        pre-text task.

    Attributes
    ----------
    model_ : SSLModel
        Fitted module: ``model_.backbone`` (the full backbone, final
        layer included), ``model_.pretext`` (pre-text head, or a
        ``ModuleDict`` with ``"shuffle"`` and ``"mae"`` heads) and
        ``model_.aux`` (auxiliary head, ``None`` when ``aux_lambda == 0``
        or when a classification ``y`` has no labeled row).
        ``model_.forward_features`` returns the embeddings.
    pretext_ : Pretext
        Pre-text task used for training: a clone of a
        :class:`~maldideepkit.pretraining.Pretext` given as ``pretext``,
        or the registered class built from the encoder-level parameters
        for a name.
    backbone_template_ : BaseSpectralClassifier
        Unfitted clone of the resolved template, with ``input_dim_`` and
        ``n_classes_`` set; ``backbone_template_._build_model()``
        rebuilds a module with the backbone's architecture.
    input_dim_ : int
        Number of input bins.
    n_features_in_ : int
        Same as :attr:`input_dim_` (scikit-learn convention).
    feature_dim_ : int
        Embedding width (``model_.feature_dim``).
    input_transform_state_ : dict
        Fitted input-transform state (its ``"mode"`` is the resolved
        transform mode).
    feature_mean_, feature_std_ : ndarray or None
        Standardization statistics of :attr:`input_transform_state_`
        (``None`` for modes without standardization).
    recon_mean_, recon_std_ : ndarray or None
        Per-bin mean and standard deviation of the model input over the
        training split, standardizing the reconstruction target of
        ``"mae"`` and ``"shuffle+mae"``; ``None`` unless the resolved
        input transform is ``"log1p"`` or ``"none"``.
    warper_ : object or None
        Fitted warper when the template sets ``warping``.
    aux_classes_ : ndarray or None
        Classes of the auxiliary classification target (``None``
        unless ``aux_lambda > 0`` and ``aux_target="classification"``).
    history_ : dict of list
        Per-epoch ``"epoch"``, ``"train_loss"``, ``"val_loss"``,
        ``"lr"``, ``"time_s"`` (empty when ``epochs=0``).
    data_device_ : torch.device
        Where the pool lived during :meth:`fit` (the training GPU, or
        ``cpu`` for the host path); see ``data_device``.

    Notes
    -----
    The backbone module of a custom classifier template (a
    :class:`~maldideepkit.BaseSpectralClassifier` subclass) does not
    need ``forward_features`` / ``feature_dim`` / ``final_layer``: as
    in the dual-branch spectral branch, its final layer is the
    ``nn.Linear`` that produces the output, found with a forward pass
    on zeros whatever the order in which the layers are registered (the
    last registered ``nn.Linear``, in ``named_modules()`` order, when
    that pass fails). The trunk is everything else, the embedding width
    is that layer's ``in_features``, and the embedding is the module's
    output with that layer replaced by ``nn.Identity``.

    :meth:`get_feature_names_out` returns ``maldispectrumencoder0``,
    ``maldispectrumencoder1``, ... (one name per embedding dimension),
    so ``set_output(transform="pandas")`` makes :meth:`transform`
    return a ``DataFrame``.

    Examples
    --------
    >>> import numpy as np
    >>> from maldideepkit.pretraining import MaldiSpectrumEncoder
    >>> rng = np.random.default_rng(0)
    >>> X = np.abs(rng.standard_normal((64, 128))).astype("float32")
    >>> enc = MaldiSpectrumEncoder(backbone="mlp", epochs=2, batch_size=16)
    >>> Z = enc.fit(X).transform(X)
    >>> Z.shape == (64, enc.feature_dim_)
    True
    """

    def __init__(
        self,
        backbone: str | BaseSpectralClassifier = "mlp",
        *,
        pretext: str | Pretext = "mae",
        mask_ratio: float = 0.15,
        shuffle_fraction: float = 0.5,
        projection_dim: int = 128,
        temperature: float = 0.1,
        augment: SpectrumAugment | None = None,
        aux_lambda: float = 0.0,
        aux_target: str = "classification",
        unlabeled_marker: Any = -1,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        batch_size: int = 256,
        epochs: int = 100,
        warmup_epochs: int = 5,
        grad_clip_norm: float | None = 1.0,
        early_stopping_patience: int = 10,
        val_fraction: float = 0.1,
        use_amp: bool = False,
        amp_dtype: str = "auto",
        device: str | torch.device = "auto",
        random_state: int = 0,
        verbose: bool = False,
        data_device: str | torch.device = "auto",
        torch_compile: bool = False,
        optimizer: Any = None,
        scheduler: Any = None,
        callbacks: Any = None,
    ) -> None:
        self.backbone = backbone
        self.pretext = pretext
        self.mask_ratio = mask_ratio
        self.shuffle_fraction = shuffle_fraction
        self.projection_dim = projection_dim
        self.temperature = temperature
        self.augment = augment
        self.aux_lambda = aux_lambda
        self.aux_target = aux_target
        self.unlabeled_marker = unlabeled_marker
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.grad_clip_norm = grad_clip_norm
        self.early_stopping_patience = early_stopping_patience
        self.val_fraction = val_fraction
        self.use_amp = use_amp
        self.amp_dtype = amp_dtype
        self.device = device
        self.random_state = random_state
        self.verbose = verbose
        self.data_device = data_device
        self.torch_compile = torch_compile
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.callbacks = callbacks

    def _resolve_pretext(self) -> Pretext:
        """Return the pre-text task for :attr:`pretext` (never the instance itself).

        A name is built from the encoder-level pre-text parameters its
        class accepts; a :class:`Pretext` instance is cloned.
        """
        pretext = self.pretext
        if isinstance(pretext, str):
            cls = get_pretext_class(pretext)
            accepted = set(cls._get_param_names())
            return cls(
                **{
                    name: getattr(self, name)
                    for name in _PRETEXT_PARAMS
                    if name in accepted
                }
            )
        if isinstance(pretext, Pretext):
            return clone(pretext)
        raise TypeError(
            "pretext must be a registered name or a Pretext instance; got "
            f"{type(pretext).__name__}."
        )

    def _check_params(self, pretext: Pretext) -> None:
        """Validate the SSL hyperparameters before fitting.

        ``pretext`` is the resolved task (:meth:`_resolve_pretext`).
        """
        if isinstance(self.pretext, str):
            _check_pretext_params(
                self,
                ("mask_ratio", "shuffle_fraction", "projection_dim", "temperature"),
            )
            _check_augment(self.augment)
            if self.augment is not None and "augment" not in pretext._get_param_names():
                warnings.warn(
                    f"augment is only used by pretext='contrastive'; it is "
                    f"ignored with pretext={self.pretext!r}.",
                    UserWarning,
                    stacklevel=3,
                )
        else:
            defaults = inspect.signature(type(self).__init__).parameters
            changed = [
                name
                for name in _PRETEXT_PARAMS
                if not same_config(getattr(self, name), defaults[name].default)
            ]
            if changed:
                warnings.warn(
                    f"pretext is a {type(self.pretext).__name__} instance, so the "
                    f"encoder-level {', '.join(changed)} "
                    f"{'is' if len(changed) == 1 else 'are'} ignored: set "
                    "pre-text parameters on the instance.",
                    UserWarning,
                    stacklevel=3,
                )
        if self.aux_target not in AUX_TARGETS:
            raise ValueError(
                f"Unknown aux_target={self.aux_target!r}; expected one of "
                f"{AUX_TARGETS}."
            )
        if float(self.aux_lambda) < 0:
            raise ValueError(f"aux_lambda must be >= 0; got {self.aux_lambda!r}.")
        if int(self.epochs) < 0:
            raise ValueError(f"epochs must be >= 0; got {self.epochs!r}.")
        resolve_optimizer(self.optimizer, float(self.weight_decay))
        check_scheduler(self.scheduler)
        callback_list(self.callbacks)
        if self.amp_dtype not in AMP_DTYPES:
            raise ValueError(
                f"Unknown amp_dtype={self.amp_dtype!r}; expected one of {AMP_DTYPES}."
            )
        pretext.check_params(self)

    def _resolve_template(self) -> BaseSpectralClassifier:
        """Return an unfitted classifier template for :attr:`backbone`."""
        backbone = self.backbone
        if isinstance(backbone, str):
            return get_backbone_class(backbone)()
        if isinstance(backbone, BaseDualSpectralClassifier):
            raise TypeError(
                "backbone= must be a single-input classifier template; a "
                f"{type(backbone).__name__} is dual-input. Pre-train its "
                "spectral backbone template instead."
            )
        if isinstance(backbone, BaseSpectralClassifier):
            return clone(backbone)
        raise TypeError(
            "backbone= must be a string or an unfitted MaldiDeepKit "
            f"classifier instance; got {type(backbone).__name__}."
        )

    def _auto_data_device(self) -> bool:
        """Return whether ``data_device`` is ``"auto"``."""
        return isinstance(self.data_device, str) and self.data_device == "auto"

    def _resolve_data_device(
        self, device: torch.device, n_rows: int, n_bins: int
    ) -> torch.device:
        """Return where the pool lives during :meth:`fit` (see ``data_device``).

        Parameters
        ----------
        device : torch.device
            Resolved training device.
        n_rows, n_bins : int
            Shape of the pool.

        Raises
        ------
        ValueError
            If ``data_device`` is not ``"auto"``, ``"cpu"``, or a device
            of the training device's type.
        """
        spec = self.data_device
        if self._auto_data_device():
            if device.type != "cuda":
                return torch.device("cpu")
            # An explicit index: older PyTorch versions reject an unindexed device.
            index = (
                device.index
                if device.index is not None
                else torch.cuda.current_device()
            )
            free, _ = torch.cuda.mem_get_info(index)
            need = 4 * (n_rows * n_bins + 4 * n_bins) + 9 * n_rows
            if need > _DATA_DEVICE_MEMORY_FRACTION * free:
                return torch.device("cpu")
            target = device
        else:
            try:
                target = torch.device(spec)
            except (RuntimeError, TypeError) as exc:
                raise ValueError(
                    "data_device must be 'auto', 'cpu', 'cuda' or a torch.device; "
                    f"got {spec!r}."
                ) from exc
            if target.type == "cpu":
                return target
            if target.type != device.type or (
                target.index is not None
                and device.index is not None
                and target.index != device.index
            ):
                raise ValueError(
                    f"data_device={spec!r} must be the training device "
                    f"(device={self.device!r} resolves to {str(device)!r}), "
                    "'cpu' or 'auto'."
                )
            target = device
        if target.type == "cuda" and target.index is None:
            target = torch.device("cuda", torch.cuda.current_device())
        return target

    def _build_aux_head(self, feature_dim: int) -> nn.Module | None:
        """Return the auxiliary head, or ``None`` when there is none.

        ``None`` when ``aux_lambda == 0``, or for a classification target
        without any labeled row (no class to predict).
        """
        if float(self.aux_lambda) <= 0:
            return None
        if self.aux_target == "regression":
            return nn.Linear(feature_dim, 1)
        n_classes = 0 if self.aux_classes_ is None else len(self.aux_classes_)
        return nn.Linear(feature_dim, n_classes) if n_classes > 0 else None

    def _build_ssl_model(self) -> SSLModel:
        """Build backbone + heads from :attr:`backbone_template_` (random init)."""
        backbone = self.backbone_template_._build_model()
        input_dim = int(self.input_dim_)
        feature_dim = split_trunk(backbone, input_dim).feature_dim
        return SSLModel(
            backbone,
            self.pretext_.build_head(feature_dim, input_dim),
            self._build_aux_head(feature_dim),
            input_dim=input_dim,
        )

    def _pretext_loss(
        self, model: SSLModel, batch: SSLBatch, generator: torch.Generator
    ) -> torch.Tensor:
        """Pre-text loss of one batch, delegated to :attr:`pretext_`."""
        return self.pretext_.loss(model, batch, generator)

    def _aux_loss(self, model: SSLModel, batch: SSLBatch) -> torch.Tensor:
        """Auxiliary loss on the labeled rows of one batch (``0`` without any).

        The auxiliary head reads the embedding of the uncorrupted batch.
        Unlabeled rows are masked out before the reduction (their
        targets are ``-1`` / ``NaN``), so a batch without labeled rows
        gives ``0``, not ``NaN``.
        """
        out = model.aux(model.forward_features(batch.x)).float()
        labeled = batch.labeled
        target = torch.where(labeled, batch.target, torch.zeros_like(batch.target))
        if self.aux_target == "classification":
            per_row = F.cross_entropy(out, target, reduction="none")
        else:
            per_row = (out.squeeze(1) - target.float()).pow(2)
        weights = labeled.to(per_row.dtype)
        return (per_row * weights).sum() / weights.sum().clamp_min(1.0)

    def _ssl_loss(
        self, model: SSLModel, batch: SSLBatch, generator: torch.Generator
    ) -> torch.Tensor:
        """Total objective ``L_pretext (+ aux_lambda * L_aux)`` of one batch."""
        loss = self._pretext_loss(model, batch, generator)
        if model.aux is not None:
            loss = loss + float(self.aux_lambda) * self._aux_loss(model, batch)
        return loss

    def fit(self, X: Any, y: Any | None = None) -> MaldiSpectrumEncoder:
        """Pre-train the backbone on ``X``.

        Parameters
        ----------
        X : array-like or MaldiSet of shape (n_samples, n_bins)
            Unlabeled spectra pool. A float32 ``np.memmap`` stays on
            disk: each batch is read from it with one indexing call
            (host pool), or it is uploaded to the GPU in row chunks
            (device pool). Warping, if any, writes the warped pool to
            host memory, and a stateful input transform (standardize,
            robust) fits its statistics on an in-memory copy of the
            training rows.
        y : array-like of shape (n_samples,) or None, default=None
            Partial targets for the auxiliary head and for a pre-text
            task with :attr:`~maldideepkit.pretraining.Pretext.uses_target`,
            which requires them
            (rows that are missing or equal to ``unlabeled_marker`` are
            unlabeled); otherwise ignored when ``aux_lambda == 0`` (so
            the encoder fits in a :class:`~sklearn.pipeline.Pipeline`).

        Returns
        -------
        self : MaldiSpectrumEncoder
            The fitted encoder.
        """
        pretext = self._resolve_pretext()
        self._check_params(pretext)
        template = self._resolve_template()
        ignored = [
            name
            for name, unset in (("init_from", None), ("freeze_trunk_epochs", 0))
            if getattr(template, name, unset) != unset
        ]
        if ignored:
            warnings.warn(
                f"The backbone template sets {' and '.join(ignored)}, which "
                f"{type(self).__name__} ignores: pre-training starts from a "
                "random backbone and trains every layer.",
                UserWarning,
                stacklevel=2,
            )
        seed_everything(int(self.random_state))
        X_np = _to_numpy(X)
        input_dim = int(X_np.shape[1])
        if template.input_dim is not None and int(template.input_dim) != input_dim:
            raise ValueError(
                f"Backbone template input_dim={template.input_dim} does not "
                f"match X.shape[1]={input_dim}."
            )
        use_aux = float(self.aux_lambda) > 0
        pretext_target = bool(pretext.uses_target)
        if pretext_target and y is None:
            raise ValueError(
                f"{type(pretext).__name__} reads the targets (uses_target=True): "
                "pass them as fit(X, y)."
            )
        use_target = use_aux or pretext_target
        warper = clone(template.warping) if template.warping is not None else None
        device = resolve_device(self.device)
        data_device = self._resolve_data_device(device, int(X_np.shape[0]), input_dim)
        if int(self.epochs) == 0:
            data_device = torch.device("cpu")
        loader_kwargs: dict[str, Any] = dict(
            batch_size=int(self.batch_size),
            val_size=float(self.val_fraction),
            random_state=int(self.random_state),
            warper=warper,
            transform_mode=resolve_transform_mode(
                bool(template.standardize), template.input_transform
            ),
            unlabeled_marker=self.unlabeled_marker,
            aux_target=self.aux_target if use_target else None,
        )
        try:
            loaders = make_pretraining_loaders(
                X_np,
                y if use_target else None,
                data_device=data_device,
                pin_memory=device.type == "cuda" and data_device.type == "cpu",
                **loader_kwargs,
            )
        except torch.cuda.OutOfMemoryError:
            if not (self._auto_data_device() and data_device.type == "cuda"):
                raise
            loaders = None
        if loaders is None:
            # The "auto" estimate fitted the free memory, the upload did
            # not (e.g. fragmentation): keep the pool on the host. Retried
            # outside the ``except`` so the partial upload is released.
            warnings.warn(
                f"data_device='auto': the pre-training pool did not fit on "
                f"{data_device} (CUDA out of memory during the upload), so it "
                "stays on the host (data_device_ is cpu).",
                UserWarning,
                stacklevel=2,
            )
            torch.cuda.empty_cache()
            data_device = torch.device("cpu")
            seed_everything(int(self.random_state))
            loaders = make_pretraining_loaders(
                X_np,
                y if use_target else None,
                data_device=data_device,
                pin_memory=device.type == "cuda",
                **loader_kwargs,
            )
        train_loader, val_loader, stats = loaders
        self.data_device_ = data_device
        if pretext_target and stats["n_labeled"] == 0:
            raise ValueError(
                f"{type(pretext).__name__} reads the targets (uses_target=True) "
                "but y has no labeled row."
            )
        if use_aux and stats["n_labeled"] == 0:
            warnings.warn(
                f"aux_lambda={self.aux_lambda} but y has no labeled row: the "
                "auxiliary loss is off.",
                UserWarning,
                stacklevel=2,
            )
        self.input_dim_ = input_dim
        self.n_features_in_ = input_dim
        self.input_transform_state_ = stats["input_transform_state"]
        self.feature_mean_ = stats["mean"]
        self.feature_std_ = stats["std"]
        recon = stats["recon_stats"]
        self.recon_mean_ = None if recon is None else recon["mean"]
        self.recon_std_ = None if recon is None else recon["std"]
        self.warper_ = stats["warper"]
        self.aux_classes_ = (
            stats["aux_classes"]
            if use_target and self.aux_target == "classification"
            else None
        )

        template.input_dim_ = input_dim
        template.n_classes_ = int(template.n_classes)
        self.backbone_template_ = template
        self.pretext_ = pretext

        model = self._build_ssl_model().to(device)
        history: dict[str, list[float]] = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "lr": [],
            "time_s": [],
        }
        if int(self.epochs) > 0:
            optimizer = build_optimizer(
                self.optimizer,
                model.parameters(),
                learning_rate=float(self.learning_rate),
                weight_decay=float(self.weight_decay),
            )
            hooks = [pretext, *callback_list(self.callbacks)]
            dispatch(hooks, "on_train_start", model, self)
            # The compiled feature extractor only lives during training.
            features = model._features
            if bool(self.torch_compile):
                model._features = _torch_compile(features)
            try:
                model, history = ssl_train_loop(
                    model,
                    train_loader,
                    val_loader,
                    self._ssl_loss,
                    optimizer,
                    device,
                    int(self.epochs),
                    EarlyStopping(patience=int(self.early_stopping_patience)),
                    input_transform_state=self.input_transform_state_,
                    warmup_epochs=int(self.warmup_epochs),
                    grad_clip_norm=(
                        float(self.grad_clip_norm)
                        if self.grad_clip_norm is not None
                        else None
                    ),
                    use_amp=bool(self.use_amp),
                    random_state=int(self.random_state),
                    verbose=bool(self.verbose),
                    amp_dtype=resolve_amp_dtype(str(self.amp_dtype), device),
                    recon_stats=recon,
                    scheduler=self.scheduler,
                    callbacks=hooks,
                )
            finally:
                model._features = features
            dispatch(hooks, "on_train_end", model)
        model.eval()
        self.model_ = model
        self.history_ = history
        self.feature_dim_ = int(model.feature_dim)
        self._device_ = device
        return self

    def _preprocess(self, X: Any, *, warp: bool = True) -> np.ndarray:
        """Apply the fitted preprocessing (warping, input transform) to ``X``.

        ``warp=False`` skips the warper (for rows that are already
        warped).
        """
        X_np = _to_numpy(X)
        if X_np.shape[1] != self.input_dim_:
            raise ValueError(
                f"X has {X_np.shape[1]} features but the encoder was fitted with "
                f"input_dim={self.input_dim_}."
            )
        return apply_fitted_preprocessing(
            X_np,
            warper=self.warper_ if warp else None,
            input_transform_state=self.input_transform_state_,
            standardize=bool(self.backbone_template_.standardize),
            feature_mean=self.feature_mean_,
            feature_std=self.feature_std_,
        )

    def transform(self, X: Any) -> np.ndarray:
        """Return the backbone embeddings of ``X``.

        Parameters
        ----------
        X : array-like or MaldiSet of shape (n_samples, n_bins)
            Raw spectra; the fitted preprocessing is applied first.

        Returns
        -------
        ndarray of shape (n_samples, ``feature_dim_``)
            ``float32`` embeddings from ``model_.forward_features``
            (eval mode, batched by ``batch_size``).

        Notes
        -----
        The fitted warper, if any, is applied to the whole of ``X`` on
        the host first. The input transform is then applied batch by
        batch (``X`` is never transformed as a whole): on a GPU model
        by its PyTorch twin on the device, after a raw copy of the batch
        (unless ``data_device="cpu"``), else in NumPy on the host. The
        embeddings are copied back to the host once, at the end.
        """
        check_is_fitted(self, "model_")
        X_np = _to_numpy(X)
        if X_np.shape[1] != self.input_dim_:
            raise ValueError(
                f"X has {X_np.shape[1]} features but the encoder was fitted with "
                f"input_dim={self.input_dim_}."
            )
        if self.warper_ is not None:
            X_np = _warp_numpy(self.warper_, X_np)
        device = self._device_
        spec = self.data_device
        on_device = device.type != "cpu" and not (
            spec == "cpu" or (isinstance(spec, torch.device) and spec.type == "cpu")
        )
        transform = _FittedTransform(self.input_transform_state_)
        model = self.model_
        model.eval()
        chunk = max(1, int(self.batch_size))
        parts: list[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, X_np.shape[0], chunk):
                # Copy the chunk: X may be a read-only memory map.
                rows = X_np[start : start + chunk]
                if on_device:
                    raw = torch.from_numpy(np.array(rows, dtype=np.float32))
                    xb = transform(raw.to(device, non_blocking=True))
                else:
                    xb = torch.from_numpy(
                        np.array(self._preprocess(rows, warp=False), dtype=np.float32)
                    ).to(device)
                parts.append(model.forward_features(xb).float())
        if not parts:
            return np.empty((0, self.feature_dim_), dtype=np.float32)
        return torch.cat(parts).cpu().numpy()

    def save(self, path: str | Path) -> None:
        """Persist the fitted encoder to ``path.pt`` + ``path.json`` (+ ``path.warper.pkl``).

        ``.pt`` holds the :attr:`model_` state dict (keys ``backbone.*``
        including the final layer, ``pretext.*``, and ``aux.*``);
        ``.json`` holds ``meta["kind"] = "encoder"``, the encoder
        parameters, the backbone template (class + parameters), and the
        fitted preprocessing. A :class:`~maldideepkit.pretraining.Pretext`
        ``pretext`` and a :class:`~maldideepkit.augment.SpectrumAugment`
        (``augment`` or a pre-text parameter) are stored as their class
        import path plus constructor parameters, as is the class of a
        custom backbone template. A single ``.pt`` / ``.pth`` / ``.json`` /
        ``.warper.pkl`` suffix on ``path`` is stripped and any other dot
        is kept (``save("enc_mask0.05")`` writes ``enc_mask0.05.pt``),
        as in :meth:`~maldideepkit.BaseSpectralClassifier.save`. The metadata is
        serialised before any file is written and every file goes
        through a temporary name, so a failing :meth:`save` writes
        nothing.

        Parameters
        ----------
        path : str or Path
            Base path without extension.

        Raises
        ------
        TypeError
            If an object parameter (of the encoder, of a ``Pretext`` or
            of the backbone template) does not keep a required
            constructor argument in an attribute of the same name.
        ValueError
            If a class or function among the parameters, or the class of
            the backbone template, cannot be re-imported from its import
            path (e.g. a class defined inside a function, or a lambda).
        """
        check_is_fitted(self, "model_")
        params = {
            name: (
                _estimator_spec(value)
                if isinstance(value, BaseSpectralClassifier)
                else _encode_param(name, value)
            )
            for name, value in self.get_params(deep=False).items()
        }
        meta: dict[str, Any] = {
            "kind": _CHECKPOINT_KIND,
            "class_name": type(self).__name__,
            "version": _CHECKPOINT_VERSION,
            "params": params,
            "backbone_template": _estimator_spec(self.backbone_template_),
            "fitted": {
                "input_dim_": int(self.input_dim_),
                "n_features_in_": int(self.n_features_in_),
                "feature_dim_": int(self.feature_dim_),
                "n_classes_": int(self.backbone_template_.n_classes_),
                "input_transform_state_": _serialise_transform_state(
                    self.input_transform_state_
                ),
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
                "recon_mean_": (
                    None
                    if self.recon_mean_ is None
                    else np.asarray(self.recon_mean_).tolist()
                ),
                "recon_std_": (
                    None
                    if self.recon_std_ is None
                    else np.asarray(self.recon_std_).tolist()
                ),
                "has_warper": self.warper_ is not None,
                "aux_classes_": (
                    None
                    if self.aux_classes_ is None
                    else np.asarray(self.aux_classes_).tolist()
                ),
                "history_": self.history_,
                "data_device_": str(getattr(self, "data_device_", "cpu")),
            },
        }
        _write_checkpoint(
            _checkpoint_base(path),
            self.model_.state_dict(),
            json.dumps(meta, indent=2),
            self.warper_,
        )

    @classmethod
    def load(cls, path: str | Path, *, trusted: bool = False) -> MaldiSpectrumEncoder:
        """Load an encoder saved with :meth:`save`.

        The template's ``warping`` parameter is not serialised (as for
        the classifiers); the fitted :attr:`warper_` is restored from
        ``.warper.pkl``. The reloaded encoder's ``backbone`` template
        therefore has ``warping=None`` (the fitted warper lives only in
        :attr:`warper_`), so cloning or refitting a loaded encoder does
        not re-apply warping: set ``warping`` on the backbone again
        before refitting if it is needed.

        A custom ``Pretext`` class (or ``SpectrumAugment`` subclass,
        or custom backbone template class) is re-imported from the
        import path stored by :meth:`save`, so it must be importable
        (defined at module level). With the default ``trusted=False``,
        such a class must come from a module already imported in the
        session (import the module that defines it before loading), and
        the other limits of ``trusted`` apply. A pre-text saved by a
        name added with :func:`~maldideepkit.pretraining.register_pretext`
        must be registered again before loading.

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
        MaldiSpectrumEncoder
            Fitted encoder ready for :meth:`transform`.

        Raises
        ------
        FileNotFoundError
            If the ``.pt``, the ``.json``, or a required ``.warper.pkl``
            is missing.
        ValueError
            If the files are not an encoder checkpoint, have an
            unsupported version, or name a class or a function that the
            ``trusted`` rule refuses or that cannot be imported, or a
            pre-text name that is not registered.

        Notes
        -----
        A ``.warper.pkl`` is a pickle, read whatever ``trusted`` is:
        load checkpoints with a warper only from trusted sources.
        """
        base = _checkpoint_base(path)
        pt_path = _checkpoint_file(base, ".pt")
        json_path = _checkpoint_file(base, ".json")
        if not pt_path.exists():
            raise FileNotFoundError(pt_path)
        if not json_path.exists():
            raise FileNotFoundError(json_path)
        with open(json_path) as fh:
            meta = json.load(fh)
        if meta.get("kind") != _CHECKPOINT_KIND:
            raise ValueError(
                f"{json_path} is not a {cls.__name__} checkpoint "
                f"(kind={meta.get('kind')!r})."
            )
        if int(meta.get("version", 0)) != _CHECKPOINT_VERSION:
            raise ValueError(
                f"Unsupported encoder checkpoint version {meta.get('version')!r}; "
                f"expected {_CHECKPOINT_VERSION}."
            )

        registry = _estimator_registry(
            meta["params"].get("backbone"), meta["backbone_template"], trusted=trusted
        )
        params = _decode_hparams(dict(meta["params"]), registry, trusted=trusted)
        instance = cls(**params)
        template = _decode_hparams(
            {"backbone_template": meta["backbone_template"]}, registry, trusted=trusted
        )["backbone_template"]
        fitted = meta["fitted"]
        template.input_dim_ = int(fitted["input_dim_"])
        template.n_classes_ = int(fitted["n_classes_"])
        instance.backbone_template_ = template
        instance.pretext_ = instance._resolve_pretext()
        instance.input_dim_ = int(fitted["input_dim_"])
        instance.n_features_in_ = int(fitted["n_features_in_"])
        instance.feature_dim_ = int(fitted["feature_dim_"])
        instance.input_transform_state_ = _deserialise_transform_state(
            fitted["input_transform_state_"]
        )
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
        for name in ("recon_mean_", "recon_std_"):
            value = fitted.get(name)
            setattr(
                instance,
                name,
                None if value is None else np.asarray(value, dtype=np.float32),
            )
        instance.aux_classes_ = (
            None
            if fitted["aux_classes_"] is None
            else np.asarray(fitted["aux_classes_"])
        )
        instance.history_ = fitted.get("history_", {})
        instance.data_device_ = torch.device(fitted.get("data_device_", "cpu"))
        instance.warper_ = None
        if fitted.get("has_warper"):
            import joblib

            warper_path = _checkpoint_file(base, ".warper.pkl")
            if not warper_path.exists():
                raise FileNotFoundError(warper_path)
            instance.warper_ = joblib.load(warper_path)

        device = resolve_device(instance.device)
        model = instance._build_ssl_model().to(device)
        state = torch.load(pt_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.eval()
        instance.model_ = model
        instance._device_ = device
        return instance

    @property
    def _n_features_out(self) -> int:
        """Embedding width, used by :meth:`get_feature_names_out`."""
        return self.feature_dim_

    def __sklearn_is_fitted__(self) -> bool:
        return hasattr(self, "model_")


__all__ = ["MaldiSpectrumEncoder"]
