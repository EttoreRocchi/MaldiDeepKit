"""Training loop for self-supervised pre-training."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any, NamedTuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from ..base.data import _STD_FLOOR, apply_input_transform, apply_input_transform_torch
from ..utils.callbacks import dispatch
from ..utils.optim import build_scheduler
from ..utils.training import EarlyStopping


class SSLBatch(NamedTuple):
    """One collated, device-placed pre-training batch.

    Attributes
    ----------
    x : torch.Tensor of shape (batch, n_bins)
        Input-transformed spectra (the model input).
    x_raw : torch.Tensor of shape (batch, n_bins)
        Raw (warped, not input-transformed) spectra, e.g. for
        intensity-weighted corruption. Pre-texts must not modify it
        in place.
    target : torch.Tensor of shape (batch,)
        Auxiliary targets (undefined on unlabeled rows).
    labeled : torch.Tensor of shape (batch,)
        Boolean mask of rows carrying a target.
    transform : callable or None
        The fitted input transform: maps a raw ``(batch, n_bins)``
        tensor, e.g. an augmented copy of ``x_raw``, to a new
        ``float32`` model input on the same device (the NumPy transform
        of :func:`apply_batch_transform` for a CPU tensor, its PyTorch
        twin :func:`~maldideepkit.base.data.apply_input_transform_torch`
        on another device). Set by :func:`prepare_batch`; ``None`` on a
        batch built by hand.
    recon_stats : tuple of torch.Tensor or None
        Per-bin ``(mean, std)`` of the model input over the training
        split, on the batch's device (``std`` is ``1`` on the bins without
        variance), used by
        :meth:`recon_target`; ``None`` when the input transform already
        standardizes the bins (see
        :func:`~maldideepkit.pretraining.make_pretraining_loaders`).
    """

    x: torch.Tensor
    x_raw: torch.Tensor
    target: torch.Tensor
    labeled: torch.Tensor
    transform: Callable[[torch.Tensor], torch.Tensor] | None = None
    recon_stats: tuple[torch.Tensor, torch.Tensor] | None = None

    def recon_target(self) -> torch.Tensor:
        """Return the reconstruction target of the masked-bin pre-texts.

        ``x`` standardized per bin with :attr:`recon_stats`, or ``x``
        itself without them. With a non-standardizing input transform
        (``"log1p"``, ``"none"``) the bins of total-ion-current
        normalised spectra are of order ``1e-4``, so a reconstruction
        loss on ``x`` would be of order ``1e-7``: below the
        early-stopping ``min_delta`` and the ``float16`` resolution.
        """
        if self.recon_stats is None:
            return self.x
        mean, std = self.recon_stats
        return (self.x - mean) / std


def apply_batch_transform(
    raw: torch.Tensor, input_transform_state: dict[str, Any]
) -> torch.Tensor:
    """Apply the fitted input transform to a raw batch.

    The transform runs in NumPy (``float32``) through
    :func:`~maldideepkit.base.data.apply_input_transform`, exactly as
    the array-level classifier path does.

    Parameters
    ----------
    raw : torch.Tensor of shape (batch, n_bins)
        Raw (warped, not input-transformed) spectra on any device.
    input_transform_state : dict
        State from :func:`~maldideepkit.base.data.fit_input_transform`.

    Returns
    -------
    torch.Tensor
        New CPU ``float32`` tensor (never sharing memory with ``raw``).
    """
    raw_np = raw.detach().cpu().numpy()
    x_np = apply_input_transform(raw_np, input_transform_state)
    if np.shares_memory(x_np, raw_np):
        x_np = x_np.copy()
    return torch.from_numpy(np.ascontiguousarray(x_np, dtype=np.float32))


class _FittedTransform:
    """The fitted input transform of the pre-training batches (``SSLBatch.transform``).

    A CPU tensor goes through the NumPy transform
    (:func:`apply_batch_transform`); a tensor on another device through
    the PyTorch twin on that device, with the fitted statistics copied to
    each device once. The result never shares memory with the input.

    Parameters
    ----------
    state : dict
        State from :func:`~maldideepkit.base.data.fit_input_transform`.
    recon_stats : dict or None, default=None
        Per-bin ``"mean"`` / ``"std"`` of the model input over the
        training split (``stats["recon_stats"]`` of
        :func:`~maldideepkit.pretraining.make_pretraining_loaders`),
        handed to each batch by :meth:`recon_stats_on`.
    """

    def __init__(
        self, state: dict[str, Any], recon_stats: dict[str, Any] | None = None
    ) -> None:
        self.state = state
        self.recon_stats = recon_stats
        self._device_states: dict[torch.device, dict[str, Any]] = {}
        self._device_recon: dict[torch.device, tuple[torch.Tensor, torch.Tensor]] = {}

    def __call__(self, raw: torch.Tensor) -> torch.Tensor:
        if raw.device.type == "cpu":
            return apply_batch_transform(raw, self.state)
        state = self._device_states.get(raw.device)
        if state is None:
            state = {
                k: v
                if k == "mode"
                else torch.as_tensor(v, dtype=torch.float32, device=raw.device)
                for k, v in self.state.items()
            }
            self._device_states[raw.device] = state
        x = apply_input_transform_torch(raw, state)
        return x.clone() if x is raw else x

    def recon_stats_on(
        self, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Return ``(mean, std)`` on ``device`` (copied once), or ``None``.

        ``std`` is ``1`` on the bins without variance.
        """
        if self.recon_stats is None:
            return None
        device = torch.device(device)
        stats = self._device_recon.get(device)
        if stats is None:
            mean = torch.as_tensor(
                self.recon_stats["mean"], dtype=torch.float32, device=device
            )
            std = torch.as_tensor(
                self.recon_stats["std"], dtype=torch.float32, device=device
            )
            std = torch.where(std < _STD_FLOOR, torch.ones_like(std), std)
            stats = (mean, std)
            self._device_recon[device] = stats
        return stats


def prepare_batch(
    batch: Sequence[torch.Tensor],
    input_transform_state: dict[str, Any] | _FittedTransform,
    device: torch.device,
) -> SSLBatch:
    """Apply the fitted input transform to a collated raw batch.

    A CPU batch is transformed in NumPy (``float32``) through
    :func:`~maldideepkit.base.data.apply_input_transform`, exactly as
    the array-level classifier path does, so per-batch values equal the
    values :func:`~maldideepkit.base.data.make_loaders` produces for the
    same rows; it is then moved to ``device``. A batch already on a
    device (the device-resident pool of
    :func:`~maldideepkit.pretraining.make_pretraining_loaders`) is
    transformed there by the PyTorch twin
    :func:`~maldideepkit.base.data.apply_input_transform_torch` (equal
    to the NumPy values up to ``float32`` rounding).

    Parameters
    ----------
    batch : sequence of torch.Tensor
        ``(x_raw, target, labeled_mask)`` as returned by the loader's
        ``collate_fn``.
    input_transform_state : dict
        State from :func:`~maldideepkit.base.data.fit_input_transform`.
    device : torch.device
        Target device.

    Returns
    -------
    SSLBatch
        Transformed input plus the raw input, targets, mask, the
        fitted transform, and its reconstruction statistics (when a
        :class:`_FittedTransform` carrying them is given).
    """
    transform = (
        input_transform_state
        if isinstance(input_transform_state, _FittedTransform)
        else _FittedTransform(input_transform_state)
    )
    x_raw, target, labeled = batch[0], batch[1], batch[2]
    x = transform(x_raw)
    return SSLBatch(
        x=x.to(device, non_blocking=True),
        x_raw=x_raw.to(device, non_blocking=True),
        target=target.to(device, non_blocking=True),
        labeled=labeled.to(device, non_blocking=True),
        transform=transform,
        recon_stats=transform.recon_stats_on(device),
    )


def _loss_sum_dtype(device: torch.device) -> torch.dtype:
    """Return the dtype of the on-device loss sums of :func:`ssl_train_loop`.

    ``float64`` (the values of the Python-float sum, without a host
    sync per batch), except on MPS, which has no ``float64``.
    """
    return torch.float32 if device.type == "mps" else torch.float64


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: Callable[[nn.Module, SSLBatch, torch.Generator], torch.Tensor],
    input_transform_state: dict[str, Any] | _FittedTransform,
    device: torch.device,
    generator: torch.Generator,
    amp_enabled: bool,
    amp_dtype: torch.dtype = torch.float16,
) -> float:
    """Return the sample-weighted mean loss of ``model`` over ``loader``.

    The sum is accumulated on the device (see :func:`_loss_sum_dtype`).
    """
    model.eval()
    sum_dtype = _loss_sum_dtype(device)
    total = torch.zeros((), dtype=sum_dtype, device=device)
    n_seen = 0
    with torch.no_grad():
        for batch in loader:
            b = prepare_batch(batch, input_transform_state, device)
            if amp_enabled:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    loss = loss_fn(model, b, generator)
            else:
                loss = loss_fn(model, b, generator)
            n = int(b.x.shape[0])
            total = total + loss.detach().to(sum_dtype) * n
            n_seen += n
    return float(total.item()) / n_seen if n_seen else float("nan")


def ssl_train_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn: Callable[[nn.Module, SSLBatch, torch.Generator], torch.Tensor],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    early_stopping: EarlyStopping,
    *,
    input_transform_state: dict[str, Any],
    warmup_epochs: int = 0,
    grad_clip_norm: float | None = None,
    use_amp: bool = False,
    random_state: int = 0,
    verbose: bool = False,
    amp_dtype: torch.dtype = torch.float16,
    recon_stats: dict[str, Any] | None = None,
    scheduler: Any = None,
    callbacks: Sequence[Any] = (),
) -> tuple[nn.Module, dict[str, list[float]]]:
    """Run self-supervised training with early stopping on the validation loss.

    Every collated batch goes through :func:`prepare_batch` (the fitted
    input transform), whatever the loader's ``collate_fn``: in NumPy for
    a host batch, by the PyTorch twin on the device for a batch already
    on it (device-resident pool). The learning
    rate follows the classifiers' schedule: a linear warmup over
    ``warmup_epochs`` epochs, then ``scheduler`` (by default cosine
    annealing to ``1e-6`` over the remaining epochs).

    Parameters
    ----------
    model : nn.Module
        Model already placed on ``device`` (typically an
        :class:`~maldideepkit.pretraining._pretext.SSLModel`).
    train_loader, val_loader : DataLoader
        Loaders from
        :func:`~maldideepkit.pretraining.make_pretraining_loaders`.
    loss_fn : callable
        ``loss_fn(model, batch, generator) -> scalar tensor``. The
        CPU ``generator`` drives every random corruption (directly, or
        by seeding a generator on the batch's device): it is seeded
        once with ``random_state`` for training and re-seeded with
        ``random_state + 1`` before each validation pass, so the
        validation loss uses the same corruption every epoch.
    optimizer : torch.optim.Optimizer
        Optimizer bound to ``model`` parameters.
    device : torch.device
        Training device.
    epochs : int
        Maximum number of epochs.
    early_stopping : EarlyStopping
        Tracks the best validation loss and its weights.
    input_transform_state : dict
        Fitted input-transform state applied to every batch.
    warmup_epochs : int, default=0
        Linear learning-rate warmup length.
    grad_clip_norm : float or None, default=None
        Global gradient-norm clipping threshold.
    use_amp : bool, default=False
        Mixed precision on a GPU (no-op on CPU).
    random_state : int, default=0
        Seed of the corruption generators.
    verbose : bool, default=False
        If ``True``, print one line per epoch.
    amp_dtype : torch.dtype, default=torch.float16
        Autocast dtype with ``use_amp``: ``torch.float16`` with a
        :class:`torch.amp.GradScaler`, ``torch.bfloat16`` without one.
    recon_stats : dict or None, default=None
        ``stats["recon_stats"]`` of
        :func:`~maldideepkit.pretraining.make_pretraining_loaders`,
        carried by every batch (:meth:`SSLBatch.recon_target`).
    scheduler : None, str or callable, default=None
        See :func:`~maldideepkit.utils.optim.build_scheduler`.
    callbacks : sequence, default=()
        Objects with the ``on_epoch_start``, ``on_step_end``, and
        ``on_epoch_end`` hooks of :class:`~maldideepkit.utils.Callback`
        (``on_train_start`` and ``on_train_end`` belong to the caller).

    Returns
    -------
    model : nn.Module
        ``model`` with the best-validation weights loaded, in
        ``eval()`` mode.
    history : dict of list
        Per-epoch ``"epoch"``, ``"train_loss"``, ``"val_loss"``,
        ``"lr"``, and ``"time_s"`` (wall-clock seconds).
    """
    amp_enabled = bool(use_amp) and device.type == "cuda"
    scaler = (
        torch.amp.GradScaler("cuda")
        if amp_enabled and amp_dtype == torch.float16
        else None
    )
    warmup = max(0, int(warmup_epochs))
    lr_scheduler = build_scheduler(
        scheduler, optimizer, epochs=int(epochs), warmup_epochs=warmup
    )
    base_lrs = [pg["lr"] for pg in optimizer.param_groups]
    params = [p for g in optimizer.param_groups for p in g["params"]]
    transform = _FittedTransform(input_transform_state, recon_stats)
    train_gen = torch.Generator().manual_seed(int(random_state))
    val_gen = torch.Generator()
    sum_dtype = _loss_sum_dtype(device)
    history: dict[str, list[float]] = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "lr": [],
        "time_s": [],
    }

    step = 0
    for epoch in range(int(epochs)):
        start = time.perf_counter()
        if warmup > 0 and epoch < warmup:
            scale = (epoch + 1) / warmup
            for pg, base in zip(optimizer.param_groups, base_lrs, strict=True):
                pg["lr"] = base * scale
        lr = float(optimizer.param_groups[0]["lr"])

        model.train()
        dispatch(callbacks, "on_epoch_start", model, epoch)
        loss_sum = torch.zeros((), dtype=sum_dtype, device=device)
        n_batches = 0
        for batch in train_loader:
            b = prepare_batch(batch, transform, device)
            optimizer.zero_grad()
            if amp_enabled:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    loss = loss_fn(model, b, train_gen)
                if scaler is not None:
                    scaler.scale(loss.float()).backward()
                    if grad_clip_norm is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.float().backward()
                    if grad_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip_norm)
                    optimizer.step()
            else:
                loss = loss_fn(model, b, train_gen)
                loss.backward()
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip_norm)
                optimizer.step()
            loss_sum = loss_sum + loss.detach().to(sum_dtype)
            n_batches += 1
            dispatch(callbacks, "on_step_end", model, step)
            step += 1

        val_gen.manual_seed(int(random_state) + 1)
        val_loss = _evaluate(
            model,
            val_loader,
            loss_fn,
            transform,
            device,
            val_gen,
            amp_enabled,
            amp_dtype,
        )
        if lr_scheduler is not None and epoch >= warmup:
            if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                lr_scheduler.step(val_loss)
            else:
                lr_scheduler.step()
        improved = early_stopping.step(val_loss, model)

        train_loss = float(loss_sum.item()) / max(1, n_batches)
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(lr)
        history["time_s"].append(time.perf_counter() - start)
        if verbose:
            marker = " *" if improved else ""
            print(
                f"epoch {epoch + 1}/{epochs}  train_loss={train_loss:.4f}  "
                f"val_loss={val_loss:.4f}{marker}"
            )
        dispatch(
            callbacks,
            "on_epoch_end",
            model,
            epoch,
            {"train_loss": train_loss, "val_loss": val_loss, "lr": lr},
        )
        if early_stopping.should_stop:
            break

    if early_stopping.best_state is not None:
        model.load_state_dict(early_stopping.best_state)
    model.eval()
    return model, history


__all__ = ["SSLBatch", "apply_batch_transform", "prepare_batch", "ssl_train_loop"]
