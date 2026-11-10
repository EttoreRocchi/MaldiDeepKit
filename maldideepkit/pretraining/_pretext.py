"""Self-supervised pre-text tasks: heads, corruption, losses, and registry.

The pooled binned backbones emit one embedding per spectrum, so every
pre-text task is realised as a small head on top of the backbone
embedding (``forward_features``): a per-bin head
``Linear(feature_dim -> input_dim)`` for ``"mae"`` and ``"shuffle"``, a
projection MLP for ``"contrastive"``. Heads are discarded after
pre-training: only the backbone transfers.

Corruptions act on the input-transformed batch (the model input), except
where raw intensities are needed: the ``"shuffle"`` sampling weights come
from the raw spectra, and the contrastive augmentations are applied to
the raw spectra before the input transform. Every random draw is made on
the batch's device and is driven by the CPU generator passed in: on a
CPU batch it is drawn from that generator, on a GPU batch from a
generator on the GPU seeded with one draw of it (see
:func:`_sampling_generator`). The corruption is therefore a
deterministic function of the generator state for a given device type
(the CPU and GPU draws differ), and the validation corruption, drawn
from a generator re-seeded before every pass, is the same every epoch.

Each task is a :class:`Pretext` (:class:`MAEPretext`,
:class:`ShufflePretext`, :class:`ShuffleMAEPretext`,
:class:`ContrastivePretext`); :func:`register_pretext` maps a name to a
class so that ``MaldiSpectrumEncoder(pretext=name)`` resolves it.
"""

from __future__ import annotations

import copy
import warnings
from abc import ABCMeta, abstractmethod
from collections.abc import Callable, Iterable
from typing import Any

import torch
import torch.nn.functional as F
from sklearn.base import BaseEstimator
from torch import nn

from ..augment.spectra import SpectrumAugment
from ..base._trunk import split_trunk
from ..utils.callbacks import TrainingHooks
from ._training import SSLBatch

PRETEXTS = ("mae", "shuffle", "shuffle+mae", "contrastive")

CONTRASTIVE_AUGMENT: dict[str, Any] = {"intensity_jitter": 0.2, "mz_shift_max_bins": 1}

_SHUFFLE_WEIGHT_FLOOR = 1e-6


class SSLModel(nn.Module):
    """Backbone + pre-text head (+ optional auxiliary head).

    The submodule names fix the encoder checkpoint layout: state-dict
    keys are ``backbone.*`` (the full backbone, final layer included),
    ``pretext.*``, and ``aux.*`` (absent when there is no auxiliary
    head).

    Parameters
    ----------
    backbone : nn.Module
        Backbone module. A built-in backbone exposes
        ``forward_features``, ``feature_dim``, and ``final_layer``; any
        other module uses the dual-branch convention (the trunk is
        everything except its final ``nn.Linear``, see
        :func:`~maldideepkit.base._trunk.split_trunk`).
    pretext : nn.Module
        Pre-text head consuming the backbone embedding (an
        :class:`~torch.nn.ModuleDict` with ``"shuffle"`` and ``"mae"``
        heads for ``pretext="shuffle+mae"``).
    aux : nn.Module or None, default=None
        Semi-supervised auxiliary head consuming the backbone embedding.
    input_dim : int or None, default=None
        Width of the backbone input, used to locate the final layer of a
        custom backbone module.

    Attributes
    ----------
    feature_dim : int
        Width of the backbone embedding.
    """

    def __init__(
        self,
        backbone: nn.Module,
        pretext: nn.Module,
        aux: nn.Module | None = None,
        input_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.pretext = pretext
        self.aux = aux
        split = split_trunk(backbone, input_dim)
        self.feature_dim = split.feature_dim
        self._features = split.features

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``(batch, input_dim)`` to ``(batch, feature_dim)`` backbone embeddings."""
        return self._features(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``(batch, input_dim)`` to ``(batch, feature_dim)`` embeddings."""
        return self.forward_features(x)


class MAEHead(nn.Module):
    """Per-bin reconstruction decoder ``Linear(feature_dim -> input_dim)``.

    Parameters
    ----------
    feature_dim : int
        Width of the backbone embedding.
    input_dim : int
        Number of bins to reconstruct.
    """

    def __init__(self, feature_dim: int, input_dim: int) -> None:
        super().__init__()
        self.decoder = nn.Linear(int(feature_dim), int(input_dim))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Map ``(batch, feature_dim)`` embeddings to ``(batch, input_dim)``."""
        return self.decoder(z)


class ShuffleHead(nn.Module):
    """Per-bin real / noise discriminator ``Linear(feature_dim -> input_dim)``.

    Parameters
    ----------
    feature_dim : int
        Width of the backbone embedding.
    input_dim : int
        Number of bins to classify.
    """

    def __init__(self, feature_dim: int, input_dim: int) -> None:
        super().__init__()
        self.discriminator = nn.Linear(int(feature_dim), int(input_dim))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Map ``(batch, feature_dim)`` embeddings to per-bin "noise" logits."""
        return self.discriminator(z)


class ProjectionHead(nn.Module):
    """Contrastive projection MLP ``feature_dim -> feature_dim -> projection_dim``.

    Parameters
    ----------
    feature_dim : int
        Width of the backbone embedding (also the hidden width).
    projection_dim : int
        Width of the projection fed to the NT-Xent loss.
    """

    def __init__(self, feature_dim: int, projection_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(int(feature_dim), int(feature_dim)),
            nn.ReLU(),
            nn.Linear(int(feature_dim), int(projection_dim)),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Map ``(batch, feature_dim)`` embeddings to ``(batch, projection_dim)``."""
        return self.mlp(z)


def _sampling_generator(
    generator: torch.Generator, device: torch.device | str
) -> torch.Generator:
    """Return the generator of the random draws made on ``device``.

    ``generator`` itself when it lives on a device of the same type
    (e.g. the CPU loss generator and a CPU batch); otherwise a new
    generator on ``device`` seeded with one draw of ``generator``, so
    the draws stay a deterministic function of ``generator``'s state
    and no random tensor crosses the host-device bus.

    Parameters
    ----------
    generator : torch.Generator
        Seeded generator (the CPU loss generator).
    device : torch.device or str
        Device of the draws.

    Returns
    -------
    torch.Generator
        A generator on ``device``.
    """
    device = torch.device(device)
    if generator.device.type == device.type:
        return generator
    seed = torch.randint(
        0, 2**62, (1,), generator=generator, device=generator.device
    ).item()
    return torch.Generator(device=device).manual_seed(int(seed))


def random_bin_mask(
    n_rows: int,
    n_bins: int,
    mask_ratio: float,
    *,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    """Draw a seeded boolean mask with a fixed number of bins per row.

    Every row masks exactly ``round(mask_ratio * n_bins)`` bins (at
    least one), chosen uniformly at random. The mask is drawn on
    ``device``: from ``generator`` on a CPU device, from a generator
    seeded with one draw of it on another device (see
    :func:`_sampling_generator`).

    Parameters
    ----------
    n_rows, n_bins : int
        Mask shape.
    mask_ratio : float
        Fraction of bins to mask, in ``(0, 1]``.
    generator : torch.Generator
        Seeded CPU generator.
    device : torch.device
        Device of the draws and of the returned mask.

    Returns
    -------
    torch.Tensor
        Boolean tensor of shape ``(n_rows, n_bins)``.
    """
    k = min(int(n_bins), max(1, int(round(float(mask_ratio) * n_bins))))
    gen = _sampling_generator(generator, device)
    scores = torch.rand((int(n_rows), int(n_bins)), generator=gen, device=gen.device)
    idx = scores.topk(k, dim=1).indices
    mask = torch.zeros((int(n_rows), int(n_bins)), dtype=torch.bool, device=gen.device)
    mask.scatter_(1, idx, True)
    return mask.to(device)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of ``values`` over the ``True`` entries of ``mask``, without a host sync.

    Boolean indexing (``values[mask]``) needs the number of selected
    entries on the host, which synchronises the host with a GPU; this
    masked sum divided by the (device-side) count gives the same mean
    up to float rounding.
    """
    return torch.where(mask, values, torch.zeros_like(values)).sum() / mask.sum()


def mae_loss(
    backbone: nn.Module,
    head: nn.Module,
    x: torch.Tensor,
    mask: torch.Tensor,
    target: torch.Tensor | None = None,
) -> torch.Tensor:
    """Masked-autoencoding loss: MSE on the masked bins only.

    Masked bins of the (input-transformed) batch are set to ``0``, the
    corrupted batch is embedded by ``backbone.forward_features`` and
    ``head`` reconstructs every bin; the loss compares the
    reconstruction with ``target`` on the masked bins (on a GPU as a
    masked sum over the count, which avoids a host sync and equals the
    mean up to float rounding).

    Parameters
    ----------
    backbone : nn.Module
        Module exposing ``forward_features`` (a backbone or an
        :class:`SSLModel`).
    head : nn.Module
        Reconstruction head (e.g. :class:`MAEHead`).
    x : torch.Tensor of shape (batch, input_dim)
        Input-transformed batch.
    mask : torch.Tensor of shape (batch, input_dim)
        Boolean mask of the bins to hide and reconstruct.
    target : torch.Tensor of shape (batch, input_dim) or None, default=None
        Reconstruction target (``None`` uses the uncorrupted ``x``), e.g.
        :meth:`SSLBatch.recon_target
        <maldideepkit.pretraining.SSLBatch.recon_target>`.

    Returns
    -------
    torch.Tensor
        Scalar loss.
    """
    if target is None:
        target = x
    recon = head(backbone.forward_features(x.masked_fill(mask, 0.0)))
    if x.device.type == "cpu":
        return F.mse_loss(recon[mask].float(), target[mask].float())
    return _masked_mean((recon.float() - target.float()).pow(2), mask)


def shuffle_corruption(
    x: torch.Tensor,
    x_raw: torch.Tensor,
    mask_ratio: float,
    shuffle_fraction: float,
    *,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Intensity-weighted bin shuffling of a batch (the ``"shuffle"`` corruption).

    Per spectrum, ``k = round(mask_ratio * n_bins)`` bins (at least one)
    are sampled without replacement with probability proportional to
    the raw intensity (``x_raw`` clipped at ``0``, plus a floor of
    ``1e-6`` times the row maximum so every bin stays reachable; an
    all-zero row is sampled uniformly). A uniformly random subset of
    ``round(shuffle_fraction * k)`` of the sampled bins receives the
    value of the same bin in another spectrum of the batch: spectrum
    ``i`` takes its values from ``donor[i]``, where ``donor`` is a
    random cyclic permutation of the rows (no fixed point when the
    batch has at least two spectra; a single spectrum is left
    unchanged). A sampled bin is labeled "noise" iff its value
    actually changed, so a swap between two equal values (e.g. a zero
    for a zero) is labeled "real".

    The swap is applied to the input-transformed batch ``x``. All the
    input transforms of :func:`~maldideepkit.base.data.fit_input_transform`
    act element-wise with per-bin statistics, so swapping transformed
    values of the same bin equals transforming the swapped raw values;
    comparing the transformed values also labels as "real" the swaps
    the model cannot see (e.g. two negative raw values that ``log1p``
    clips to the same ``0``).

    Parameters
    ----------
    x : torch.Tensor of shape (batch, n_bins)
        Input-transformed batch (the model input).
    x_raw : torch.Tensor of shape (batch, n_bins)
        Raw (pre-transform) batch, used for the sampling weights only.
    mask_ratio : float
        Fraction of bins sampled per spectrum, in ``(0, 1]``.
    shuffle_fraction : float
        Fraction of the sampled bins that are swapped, in ``[0, 1]``.
    generator : torch.Generator
        Seeded CPU generator driving every draw (the draws are made on
        the device of ``x``, see :func:`random_bin_mask`).

    Returns
    -------
    x_corrupt : torch.Tensor of shape (batch, n_bins)
        Corrupted copy of ``x``.
    sampled : torch.Tensor of shape (batch, n_bins)
        Boolean mask of the sampled bins (the bins scored by the loss).
    noise : torch.Tensor of shape (batch, n_bins)
        Boolean "noise" labels (``True`` only on sampled bins whose
        value changed).
    """
    n_rows, n_bins = int(x.shape[0]), int(x.shape[1])
    k = min(n_bins, max(1, int(round(float(mask_ratio) * n_bins))))
    n_swap = min(k, int(round(float(shuffle_fraction) * k)))

    gen = _sampling_generator(generator, x.device)
    device = gen.device
    weights = x_raw.detach().float().to(device).clamp_min(0.0)
    row_max = weights.amax(dim=1, keepdim=True)
    weights = weights + torch.where(
        row_max > 0, row_max * _SHUFFLE_WEIGHT_FLOOR, torch.ones_like(row_max)
    )
    if device.type == "cpu":
        sampled_idx = torch.multinomial(weights, k, replacement=False, generator=gen)
    else:
        race = torch.empty_like(weights).exponential_(1.0, generator=gen)
        sampled_idx = (weights / race).topk(k, dim=1).indices
    pick = (
        torch.rand((n_rows, k), generator=gen, device=device)
        .topk(n_swap, dim=1)
        .indices
    )
    swap_idx = sampled_idx.gather(1, pick)
    order = torch.randperm(n_rows, generator=gen, device=device)
    donor = torch.empty_like(order)
    donor[order] = order.roll(-1)

    sampled = torch.zeros((n_rows, n_bins), dtype=torch.bool, device=device)
    sampled.scatter_(1, sampled_idx, True)
    swapped = torch.zeros((n_rows, n_bins), dtype=torch.bool, device=device)
    swapped.scatter_(1, swap_idx, True)
    sampled = sampled.to(x.device)
    swapped = swapped.to(x.device)
    x_corrupt = torch.where(swapped, x[donor.to(x.device)], x)
    noise = swapped & (x_corrupt != x)
    return x_corrupt, sampled, noise


def shuffle_loss(
    backbone: nn.Module,
    head: nn.Module,
    x_corrupt: torch.Tensor,
    sampled: torch.Tensor,
    noise: torch.Tensor,
) -> torch.Tensor:
    """Per-bin real / noise discrimination loss: BCE on the sampled bins only.

    On a GPU the mean over the sampled bins is a masked sum over their
    count (no host sync; equal up to float rounding).

    Parameters
    ----------
    backbone : nn.Module
        Module exposing ``forward_features`` (a backbone or an
        :class:`SSLModel`).
    head : nn.Module
        Per-bin discrimination head (e.g. :class:`ShuffleHead`).
    x_corrupt, sampled, noise : torch.Tensor of shape (batch, n_bins)
        Outputs of :func:`shuffle_corruption`.

    Returns
    -------
    torch.Tensor
        Scalar loss.
    """
    logits = head(backbone.forward_features(x_corrupt))
    if x_corrupt.device.type == "cpu":
        return F.binary_cross_entropy_with_logits(
            logits[sampled].float(), noise[sampled].float()
        )
    bce = F.binary_cross_entropy_with_logits(
        logits.float(), noise.float(), reduction="none"
    )
    return _masked_mean(bce, sampled)


def _reseeded(augment: SpectrumAugment, seed: int) -> SpectrumAugment:
    """Return a fresh copy of ``augment`` seeded with ``seed``.

    A shallow copy with its own random state, so ``augment`` and its RNG
    state are untouched, whatever the constructor of a subclass takes.
    """
    reseeded = copy.copy(augment)
    reseeded.random_state = seed
    reseeded._generator = None
    reseeded._np_rng = None
    return reseeded


def contrastive_views(
    x_raw: torch.Tensor,
    transform: Callable[[torch.Tensor], torch.Tensor],
    mask_ratio: float,
    *,
    generator: torch.Generator,
    device: torch.device,
    augment: SpectrumAugment | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw two augmented, masked views of every spectrum of a raw batch.

    Each view is ``augment`` (``None``: a
    :class:`~maldideepkit.augment.SpectrumAugment` with
    :data:`CONTRASTIVE_AUGMENT`) applied to the raw spectra on
    ``device``, then ``transform`` (the fitted input transform), then a
    :func:`random_bin_mask` of ``mask_ratio`` of the bins set to ``0``,
    as in ``"mae"``. The augmentation runs on a copy of ``augment``
    re-seeded from ``generator`` (its own ``random_state`` is ignored;
    the copy seeds its own generator on ``device``), so the views
    depend on ``generator`` only.

    Parameters
    ----------
    x_raw : torch.Tensor of shape (batch, n_bins)
        Raw (pre-transform) batch.
    transform : callable
        Maps a raw ``float32`` batch on ``device`` to the model input
        (e.g. ``SSLBatch.transform``).
    mask_ratio : float
        Fraction of bins masked per view, in ``(0, 1]``.
    generator : torch.Generator
        Seeded CPU generator driving every draw.
    device : torch.device
        Device of the draws and of the returned views.
    augment : SpectrumAugment or None, default=None
        Augmentation of the raw spectra (``None``: the
        :data:`CONTRASTIVE_AUGMENT` default).

    Returns
    -------
    tuple of torch.Tensor
        Two views of shape ``(batch, n_bins)``.
    """
    raw = x_raw.detach().float().to(device)
    seed = int(torch.randint(0, 2**31 - 1, (1,), generator=generator).item())
    if augment is None:
        augment = SpectrumAugment(**CONTRASTIVE_AUGMENT, random_state=seed)
    else:
        augment = _reseeded(augment, seed)
    views = []
    for _ in range(2):
        view = transform(augment(raw)).to(device)
        mask = random_bin_mask(
            view.shape[0], view.shape[1], mask_ratio, generator=generator, device=device
        )
        views.append(view.masked_fill(mask, 0.0))
    return views[0], views[1]


def nt_xent_loss(
    proj_a: torch.Tensor, proj_b: torch.Tensor, temperature: float
) -> torch.Tensor:
    """Normalised-temperature cross-entropy (NT-Xent, SimCLR) of paired projections.

    Row ``i`` of ``proj_a`` and row ``i`` of ``proj_b`` are the positive
    pair; the other ``2 * (batch - 1)`` projections of the batch are
    the negatives. Computed in ``float32``.

    Parameters
    ----------
    proj_a, proj_b : torch.Tensor of shape (batch, projection_dim)
        Projections of the two views.
    temperature : float
        Softmax temperature (``> 0``).

    Returns
    -------
    torch.Tensor
        Scalar loss (``0`` for a single pair, which has no negative).
    """
    n = int(proj_a.shape[0])
    z = F.normalize(torch.cat([proj_a, proj_b]).float(), dim=1)
    with torch.autocast(device_type=z.device.type, enabled=False):
        sim = (z @ z.T) / float(temperature)
    self_mask = torch.eye(2 * n, dtype=torch.bool, device=sim.device)
    sim = sim.masked_fill(self_mask, float("-inf"))
    idx = torch.arange(n, device=sim.device)
    target = torch.cat([idx + n, idx])
    return F.cross_entropy(sim, target)


def contrastive_loss(
    backbone: nn.Module,
    head: nn.Module,
    view_a: torch.Tensor,
    view_b: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """NT-Xent loss of two views, embedded together in one forward pass.

    Parameters
    ----------
    backbone : nn.Module
        Module exposing ``forward_features`` (a backbone or an
        :class:`SSLModel`).
    head : nn.Module
        Projection head (e.g. :class:`ProjectionHead`).
    view_a, view_b : torch.Tensor of shape (batch, n_bins)
        Outputs of :func:`contrastive_views`.
    temperature : float
        NT-Xent temperature.

    Returns
    -------
    torch.Tensor
        Scalar loss.
    """
    n = int(view_a.shape[0])
    proj = head(backbone.forward_features(torch.cat([view_a, view_b])))
    return nt_xent_loss(proj[:n], proj[n:], temperature)


_PARAM_CHECKS: dict[str, tuple[Callable[[Any], bool], str]] = {
    "mask_ratio": (lambda v: 0.0 < float(v) <= 1.0, "in (0, 1]"),
    "shuffle_fraction": (lambda v: 0.0 <= float(v) <= 1.0, "in [0, 1]"),
    "projection_dim": (lambda v: int(v) >= 1, ">= 1"),
    "temperature": (lambda v: float(v) > 0, "> 0"),
}


def _check_pretext_params(owner: Any, names: Iterable[str]) -> None:
    """Validate the built-in pre-text parameters ``names`` of ``owner``.

    ``owner`` is a :class:`Pretext` or the encoder (its encoder-level
    parameters); :class:`ValueError` names the first invalid one.
    """
    for name in names:
        value = getattr(owner, name)
        valid, constraint = _PARAM_CHECKS[name]
        if not valid(value):
            raise ValueError(f"{name} must be {constraint}; got {value!r}.")


def _check_augment(augment: Any) -> None:
    """Raise :class:`TypeError` unless ``augment`` is ``None`` or a SpectrumAugment."""
    if augment is not None and not isinstance(augment, SpectrumAugment):
        raise TypeError(
            "augment must be a SpectrumAugment instance or None; got "
            f"{type(augment).__name__}."
        )


class Pretext(TrainingHooks, BaseEstimator, metaclass=ABCMeta):  # type: ignore[misc]
    """Base class of the self-supervised pre-text tasks of the encoder.

    A pre-text task builds a head on the backbone embedding and defines
    the self-supervised loss of one batch. Pass an instance as
    ``MaldiSpectrumEncoder(pretext=...)``, or register the class under a
    name with :func:`register_pretext` and pass the name.

    Subclasses follow the scikit-learn estimator conventions: every
    ``__init__`` argument is stored unchanged under its own name (no
    validation or conversion in ``__init__``), so ``get_params``,
    ``set_params``, and :func:`~sklearn.base.clone` work and nested
    ``pretext__<param>`` keys reach the instance through the encoder
    (e.g. in :class:`~sklearn.model_selection.GridSearchCV`). The encoder
    trains with a clone (its ``pretext_`` attribute), never with the
    instance it was given. Parameters are validated in
    :meth:`check_params`, called by the encoder's ``fit``.

    Subclasses implement :meth:`build_head` and :meth:`loss`, and may
    override :meth:`check_params`, the training hooks of
    :class:`~maldideepkit.utils.Callback` (``on_train_start(model,
    encoder)``, ``on_epoch_start(model, epoch)``, ``on_step_end(model,
    step)``, ``on_epoch_end(model, epoch, logs)``, ``on_train_end(model)``,
    no-ops by default) and :attr:`uses_target`. State a hook creates on
    the instance (e.g. a momentum teacher) lives in :attr:`pretext_`
    only: it reaches neither the checkpoint nor early stopping, unless
    it is part of the head. Every random draw of :meth:`loss`
    must be driven by its ``generator`` (a seeded CPU generator),
    directly or through a generator on the batch's device seeded from
    it (as the built-in tasks do on a GPU batch): the encoder re-seeds
    it before each validation pass, so the validation loss uses the
    same corruption every epoch and does not depend on the global
    RNG.

    Attributes
    ----------
    uses_target : bool, default=False
        Class attribute. When ``True``, the ``y`` given to the encoder's
        ``fit`` is required and reaches :meth:`loss` as ``batch.target``
        and ``batch.labeled`` even with ``aux_lambda == 0``: encoded as
        the encoder's ``aux_target`` says (``"classification"``, the
        default, gives class indices in the order of ``aux_classes_``;
        unlabeled rows are ``-1``), so ``y`` can be, e.g., the acquisition
        site. With ``aux_lambda > 0`` the auxiliary head reads the same
        ``y``.

    Examples
    --------
    A denoising pre-text task: reconstruct the clean model input from a
    copy with additive Gaussian noise.

    >>> import torch
    >>> import torch.nn.functional as F
    >>> from torch import nn
    >>> from maldideepkit.pretraining import MaldiSpectrumEncoder, Pretext
    >>> class DenoisePretext(Pretext):
    ...     def __init__(self, noise_std=0.1):
    ...         self.noise_std = noise_std
    ...
    ...     def check_params(self, encoder=None):
    ...         if float(self.noise_std) <= 0:
    ...             raise ValueError("noise_std must be > 0.")
    ...
    ...     def build_head(self, feature_dim, input_dim):
    ...         return nn.Linear(feature_dim, input_dim)
    ...
    ...     def loss(self, model, batch, generator):
    ...         noise = torch.randn(batch.x.shape, generator=generator)
    ...         noisy = batch.x + float(self.noise_std) * noise.to(batch.x.device)
    ...         recon = model.pretext(model.forward_features(noisy))
    ...         return F.mse_loss(recon.float(), batch.x.float())
    >>> enc = MaldiSpectrumEncoder(pretext=DenoisePretext(noise_std=0.2))
    """

    uses_target: bool = False

    def check_params(self, encoder: Any | None = None) -> None:
        """Validate the parameters before fitting (default: no check).

        Raise :class:`ValueError` / :class:`TypeError` on an invalid
        value; a :class:`UserWarning` may flag a poor configuration.

        Parameters
        ----------
        encoder : MaldiSpectrumEncoder or None, default=None
            The encoder being fitted, for checks against its own
            parameters (e.g. ``batch_size``). It must not be modified.
        """

    @abstractmethod
    def build_head(self, feature_dim: int, input_dim: int) -> nn.Module:
        """Return a freshly initialised pre-text head.

        The head is registered as ``model.pretext`` of the
        :class:`SSLModel` (state-dict keys ``pretext.*``); use an
        :class:`~torch.nn.ModuleDict` for several heads. It is
        discarded at fine-tuning time.

        Parameters
        ----------
        feature_dim : int
            Width of the backbone embedding.
        input_dim : int
            Number of input bins.

        Returns
        -------
        nn.Module
            The pre-text head.
        """

    @abstractmethod
    def loss(
        self, model: SSLModel, batch: SSLBatch, generator: torch.Generator
    ) -> torch.Tensor:
        """Return the scalar pre-text loss of one batch.

        Parameters
        ----------
        model : SSLModel
            ``model.forward_features(x)`` embeds a model input,
            ``model.pretext`` is the head of :meth:`build_head`.
        batch : SSLBatch
            Device-placed batch: ``x`` (model input), ``x_raw`` (raw
            spectra, not to be modified in place) and ``transform``
            (the fitted input transform, raw batch to model input on
            the same device).
        generator : torch.Generator
            Seeded CPU generator; every random draw must be driven by
            it.

        Returns
        -------
        torch.Tensor
            Scalar loss.
        """


def _mae_batch_loss(
    model: nn.Module,
    head: nn.Module,
    batch: SSLBatch,
    mask_ratio: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """``"mae"`` loss of one batch with a fresh uniform mask.

    The target is :meth:`SSLBatch.recon_target` (the model input,
    standardized per bin when the input transform does not).
    """
    mask = random_bin_mask(
        batch.x.shape[0],
        batch.x.shape[1],
        float(mask_ratio),
        generator=generator,
        device=batch.x.device,
    )
    return mae_loss(model, head, batch.x, mask, batch.recon_target())


def _shuffle_batch_loss(
    model: nn.Module,
    head: nn.Module,
    batch: SSLBatch,
    mask_ratio: float,
    shuffle_fraction: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """``"shuffle"`` loss of one batch with a fresh intensity-weighted corruption."""
    x_corrupt, sampled, noise = shuffle_corruption(
        batch.x,
        batch.x_raw,
        float(mask_ratio),
        float(shuffle_fraction),
        generator=generator,
    )
    return shuffle_loss(model, head, x_corrupt, sampled, noise)


class MAEPretext(Pretext):
    """Masked-bin reconstruction (``pretext="mae"``).

    ``mask_ratio`` of the bins of every input-transformed spectrum,
    drawn uniformly, are set to ``0``; a per-bin decoder
    ``Linear(feature_dim -> input_dim)`` (:class:`MAEHead`) reconstructs
    them from the embedding (MSE on the masked bins). The target is the
    model input, standardized per bin with the training-split statistics
    when the input transform leaves the bins on the raw intensity scale
    (``"log1p"``, ``"none"``; :meth:`SSLBatch.recon_target
    <maldideepkit.pretraining.SSLBatch.recon_target>`).

    Parameters
    ----------
    mask_ratio : float, default=0.15
        Fraction of bins masked per spectrum, in ``(0, 1]``.
    """

    def __init__(self, mask_ratio: float = 0.15) -> None:
        self.mask_ratio = mask_ratio

    def check_params(self, encoder: Any | None = None) -> None:
        """Validate ``mask_ratio``."""
        _check_pretext_params(self, ("mask_ratio",))

    def build_head(self, feature_dim: int, input_dim: int) -> nn.Module:
        """Return a :class:`MAEHead`.

        Parameters
        ----------
        feature_dim : int
            Width of the backbone embedding.
        input_dim : int
            Number of input bins.

        Returns
        -------
        MAEHead
            The per-bin reconstruction decoder.
        """
        return MAEHead(feature_dim, input_dim)

    def loss(
        self, model: SSLModel, batch: SSLBatch, generator: torch.Generator
    ) -> torch.Tensor:
        """Return the masked-reconstruction loss of one batch.

        Parameters
        ----------
        model : SSLModel
            Model whose ``pretext`` is the head of :meth:`build_head`.
        batch : SSLBatch
            Device-placed batch.
        generator : torch.Generator
            Seeded CPU generator driving the mask.

        Returns
        -------
        torch.Tensor
            Scalar loss.
        """
        return _mae_batch_loss(model, model.pretext, batch, self.mask_ratio, generator)


class ShufflePretext(Pretext):
    """Intensity-weighted bin shuffling (``pretext="shuffle"``).

    ``mask_ratio`` of the bins are sampled without replacement with
    probability proportional to the raw (pre-transform) intensity;
    ``shuffle_fraction`` of them take the value of the same bin in
    another spectrum of the batch (:func:`shuffle_corruption`). A
    per-bin discriminator ``Linear(feature_dim -> input_dim)``
    (:class:`ShuffleHead`) predicts which sampled bins are "noise" (BCE
    on the sampled bins); a swap that leaves the value unchanged is
    labeled "real".

    Parameters
    ----------
    mask_ratio : float, default=0.15
        Fraction of bins sampled per spectrum, in ``(0, 1]``.
    shuffle_fraction : float, default=0.5
        Fraction of the sampled bins that are swapped, in ``[0, 1]``.
    """

    def __init__(self, mask_ratio: float = 0.15, shuffle_fraction: float = 0.5) -> None:
        self.mask_ratio = mask_ratio
        self.shuffle_fraction = shuffle_fraction

    def check_params(self, encoder: Any | None = None) -> None:
        """Validate ``mask_ratio`` and ``shuffle_fraction``."""
        _check_pretext_params(self, ("mask_ratio", "shuffle_fraction"))

    def build_head(self, feature_dim: int, input_dim: int) -> nn.Module:
        """Return a :class:`ShuffleHead`.

        Parameters
        ----------
        feature_dim : int
            Width of the backbone embedding.
        input_dim : int
            Number of input bins.

        Returns
        -------
        ShuffleHead
            The per-bin real / noise discriminator.
        """
        return ShuffleHead(feature_dim, input_dim)

    def loss(
        self, model: SSLModel, batch: SSLBatch, generator: torch.Generator
    ) -> torch.Tensor:
        """Return the real / noise discrimination loss of one batch.

        Parameters
        ----------
        model : SSLModel
            Model whose ``pretext`` is the head of :meth:`build_head`.
        batch : SSLBatch
            Device-placed batch.
        generator : torch.Generator
            Seeded CPU generator driving the corruption.

        Returns
        -------
        torch.Tensor
            Scalar loss.
        """
        return _shuffle_batch_loss(
            model,
            model.pretext,
            batch,
            self.mask_ratio,
            self.shuffle_fraction,
            generator,
        )


class ShuffleMAEPretext(Pretext):
    """Bin shuffling plus masked reconstruction (``pretext="shuffle+mae"``).

    Both corruptions (:class:`ShufflePretext` then :class:`MAEPretext`)
    are drawn independently, each with its own head (an
    :class:`~torch.nn.ModuleDict` with ``"shuffle"`` and ``"mae"``
    heads), and the two losses are summed.

    Parameters
    ----------
    mask_ratio : float, default=0.15
        Fraction of bins sampled (shuffle) and masked (mae) per
        spectrum, in ``(0, 1]``.
    shuffle_fraction : float, default=0.5
        Fraction of the sampled bins that are swapped, in ``[0, 1]``.
    """

    def __init__(self, mask_ratio: float = 0.15, shuffle_fraction: float = 0.5) -> None:
        self.mask_ratio = mask_ratio
        self.shuffle_fraction = shuffle_fraction

    def check_params(self, encoder: Any | None = None) -> None:
        """Validate ``mask_ratio`` and ``shuffle_fraction``."""
        _check_pretext_params(self, ("mask_ratio", "shuffle_fraction"))

    def build_head(self, feature_dim: int, input_dim: int) -> nn.Module:
        """Return a ``ModuleDict`` of a :class:`ShuffleHead` and a :class:`MAEHead`.

        Parameters
        ----------
        feature_dim : int
            Width of the backbone embedding.
        input_dim : int
            Number of input bins.

        Returns
        -------
        torch.nn.ModuleDict
            The heads under ``"shuffle"`` and ``"mae"``.
        """
        return nn.ModuleDict(
            {
                "shuffle": ShuffleHead(feature_dim, input_dim),
                "mae": MAEHead(feature_dim, input_dim),
            }
        )

    def loss(
        self, model: SSLModel, batch: SSLBatch, generator: torch.Generator
    ) -> torch.Tensor:
        """Return the summed shuffle and masked-reconstruction losses of one batch.

        Parameters
        ----------
        model : SSLModel
            Model whose ``pretext`` is the head of :meth:`build_head`.
        batch : SSLBatch
            Device-placed batch.
        generator : torch.Generator
            Seeded CPU generator driving both corruptions.

        Returns
        -------
        torch.Tensor
            Scalar loss.
        """
        heads = model.pretext
        return _shuffle_batch_loss(
            model,
            heads["shuffle"],
            batch,
            self.mask_ratio,
            self.shuffle_fraction,
            generator,
        ) + _mae_batch_loss(model, heads["mae"], batch, self.mask_ratio, generator)


class ContrastivePretext(Pretext):
    """Contrastive views with the NT-Xent loss (``pretext="contrastive"``).

    Two views per spectrum (:func:`contrastive_views`): ``augment`` on
    the raw spectra, then the fitted input transform and ``"mae"``-style
    masking of ``mask_ratio`` of the bins. A projection MLP
    ``feature_dim -> feature_dim -> projection_dim``
    (:class:`ProjectionHead`) feeds the NT-Xent loss with the other
    spectra of the batch as negatives. :meth:`check_params` warns when
    the encoder's ``batch_size < 64`` (few negatives).

    Parameters
    ----------
    mask_ratio : float, default=0.15
        Fraction of bins masked per view, in ``(0, 1]``.
    projection_dim : int, default=128
        Output width of the projection head.
    temperature : float, default=0.1
        NT-Xent temperature (``> 0``).
    augment : SpectrumAugment or None, default=None
        Augmentation of the raw spectra drawing the two views. ``None``
        uses :data:`CONTRASTIVE_AUGMENT` (a per-spectrum multiplicative
        intensity jitter ``1 + U(-0.2, 0.2)`` and a global m/z shift of
        up to 1 bin). Every batch runs on a copy re-seeded from the
        loss generator, so the instance is never modified and its own
        ``random_state`` is ignored.
    """

    def __init__(
        self,
        mask_ratio: float = 0.15,
        projection_dim: int = 128,
        temperature: float = 0.1,
        augment: SpectrumAugment | None = None,
    ) -> None:
        self.mask_ratio = mask_ratio
        self.projection_dim = projection_dim
        self.temperature = temperature
        self.augment = augment

    def check_params(self, encoder: Any | None = None) -> None:
        """Validate the parameters; warn when the encoder's ``batch_size < 64``."""
        _check_pretext_params(self, ("mask_ratio", "projection_dim", "temperature"))
        _check_augment(self.augment)
        if encoder is not None and int(encoder.batch_size) < 64:
            warnings.warn(
                f"pretext='contrastive' with batch_size={encoder.batch_size}: "
                "NT-Xent uses the other spectra of the batch as negatives, so "
                "batch_size >= 64 is recommended.",
                UserWarning,
                stacklevel=4,
            )

    def build_head(self, feature_dim: int, input_dim: int) -> nn.Module:
        """Return a :class:`ProjectionHead`.

        Parameters
        ----------
        feature_dim : int
            Width of the backbone embedding.
        input_dim : int
            Number of input bins (unused).

        Returns
        -------
        ProjectionHead
            The ``feature_dim -> feature_dim -> projection_dim``
            projection MLP.
        """
        return ProjectionHead(feature_dim, int(self.projection_dim))

    def loss(
        self, model: SSLModel, batch: SSLBatch, generator: torch.Generator
    ) -> torch.Tensor:
        """Return the NT-Xent loss of two augmented, masked views of one batch.

        Parameters
        ----------
        model : SSLModel
            Model whose ``pretext`` is the head of :meth:`build_head`.
        batch : SSLBatch
            Device-placed batch.
        generator : torch.Generator
            Seeded CPU generator driving the augmentation and the masks.

        Returns
        -------
        torch.Tensor
            Scalar loss.
        """
        if batch.transform is None:
            raise ValueError(
                "ContrastivePretext needs batch.transform (the fitted input "
                "transform) to transform the augmented raw spectra."
            )
        view_a, view_b = contrastive_views(
            batch.x_raw,
            batch.transform,
            float(self.mask_ratio),
            generator=generator,
            device=batch.x.device,
            augment=self.augment,
        )
        return contrastive_loss(
            model, model.pretext, view_a, view_b, float(self.temperature)
        )


_REGISTRY: dict[str, type[Pretext]] = dict(
    zip(
        PRETEXTS,
        (MAEPretext, ShufflePretext, ShuffleMAEPretext, ContrastivePretext),
        strict=True,
    )
)


def register_pretext(name: str, cls: type[Pretext]) -> type[Pretext]:
    """Register a :class:`Pretext` subclass under ``name``.

    ``MaldiSpectrumEncoder(pretext=name)`` then builds ``cls`` with the
    encoder-level pre-text parameters (``mask_ratio``,
    ``shuffle_fraction``, ``projection_dim``, ``temperature``,
    ``augment``) its ``__init__`` accepts, the others keeping the class
    defaults. Registering a name again replaces its class; the built-in
    names cannot be re-registered. A checkpoint saved with a registered
    name stores the name only, so register it again before
    ``MaldiSpectrumEncoder.load``.

    Parameters
    ----------
    name : str
        Name accepted by ``pretext=``.
    cls : type
        Concrete :class:`Pretext` subclass.

    Returns
    -------
    type
        ``cls`` (unchanged).

    Raises
    ------
    ValueError
        If ``name`` is empty or a built-in pre-text name.
    TypeError
        If ``cls`` is not a :class:`Pretext` subclass.
    """
    if not isinstance(name, str) or not name:
        raise ValueError(f"name must be a non-empty string; got {name!r}.")
    if not (isinstance(cls, type) and issubclass(cls, Pretext)):
        raise TypeError(f"cls must be a Pretext subclass; got {cls!r}.")
    if name in PRETEXTS:
        raise ValueError(
            f"{name!r} is a built-in pre-text and cannot be re-registered; "
            "pass an instance of your class as pretext= instead."
        )
    _REGISTRY[name] = cls
    return cls


def get_pretext_class(name: str) -> type[Pretext]:
    """Return the :class:`Pretext` subclass registered under ``name``.

    Parameters
    ----------
    name : str
        A built-in name (``"mae"``, ``"shuffle"``, ``"shuffle+mae"``,
        ``"contrastive"``) or one added with :func:`register_pretext`.

    Returns
    -------
    type
        The registered class.

    Raises
    ------
    ValueError
        If ``name`` is not registered.
    """
    try:
        return _REGISTRY[name]
    except (KeyError, TypeError):
        raise ValueError(
            f"Unknown pretext={name!r}; expected one of {tuple(_REGISTRY)} "
            "(register_pretext adds a name) or a Pretext instance."
        ) from None


__all__ = [
    "CONTRASTIVE_AUGMENT",
    "ContrastivePretext",
    "MAEHead",
    "MAEPretext",
    "PRETEXTS",
    "Pretext",
    "ProjectionHead",
    "SSLModel",
    "ShuffleHead",
    "ShuffleMAEPretext",
    "ShufflePretext",
    "contrastive_loss",
    "contrastive_views",
    "get_pretext_class",
    "mae_loss",
    "nt_xent_loss",
    "random_bin_mask",
    "register_pretext",
    "shuffle_corruption",
    "shuffle_loss",
]
