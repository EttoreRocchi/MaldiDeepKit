"""Generic training primitives used by :class:`~maldideepkit.BaseSpectralClassifier`."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from ..augment.mixing import apply_cutmix, apply_mixup, to_one_hot
from .callbacks import dispatch
from .loss import _gather_cw_per_sample, _weighted_loss_reduce
from .sam import SAMOptimizer

AMP_DTYPES = ("float16", "bfloat16", "auto")


def resolve_amp_dtype(amp_dtype: str, device: torch.device) -> torch.dtype:
    """Return the autocast dtype of an ``amp_dtype`` parameter on ``device``.

    ``"auto"`` resolves to ``bfloat16`` on a GPU that supports it,
    else to ``float16``.

    Parameters
    ----------
    amp_dtype : {"float16", "bfloat16", "auto"}
        Requested mixed-precision dtype.
    device : torch.device
        Training device.

    Returns
    -------
    torch.dtype
        ``torch.float16`` or ``torch.bfloat16``.

    Raises
    ------
    ValueError
        If ``amp_dtype`` is not one of :data:`AMP_DTYPES`.
    """
    if amp_dtype not in AMP_DTYPES:
        raise ValueError(
            f"Unknown amp_dtype={amp_dtype!r}; expected one of {AMP_DTYPES}."
        )
    if amp_dtype == "auto":
        bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
        return torch.bfloat16 if bf16 else torch.float16
    return torch.bfloat16 if amp_dtype == "bfloat16" else torch.float16


class EarlyStopping:
    """Track the best validation loss and signal when to stop training.

    Parameters
    ----------
    patience : int
        Number of consecutive epochs without improvement before
        :attr:`should_stop` flips to ``True``.
    min_delta : float, default=1e-6
        Absolute floor on the improvement counted as progress.
    min_delta_rel : float, default=0.0
        Relative floor: an epoch counts as improvement only if
        ``val_loss < best_loss - max(min_delta, min_delta_rel * |best_loss|)``.
        Useful for losses that asymptote near small values where the
        absolute ``min_delta`` is essentially never the binding
        constraint.

    Attributes
    ----------
    best_loss : float
        Best validation loss observed so far (``inf`` before the first
        update).
    best_state : dict or None
        CPU copy of the model ``state_dict`` at the best epoch.
    best_epoch : int or None
        0-based index of the :meth:`step` call that produced
        :attr:`best_state` (the epoch index in :func:`train_loop`, which
        steps once per epoch); ``None`` before the first improvement.
    should_stop : bool
        ``True`` once ``patience`` epochs have elapsed without an
        improvement.
    """

    def __init__(
        self,
        patience: int,
        min_delta: float = 1e-6,
        min_delta_rel: float = 0.0,
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.min_delta_rel = min_delta_rel
        self.best_loss = float("inf")
        self.best_state: dict[str, torch.Tensor] | None = None
        self.best_epoch: int | None = None
        self.should_stop = False
        self._stale = 0
        self._n_steps = 0

    def step(self, val_loss: float, model: nn.Module) -> bool:
        """Record ``val_loss`` and snapshot ``model`` if it improved.

        Parameters
        ----------
        val_loss : float
            Validation loss for the current epoch.
        model : nn.Module
            Model whose parameters will be cached on improvement.

        Returns
        -------
        bool
            ``True`` if the loss improved this call.
        """
        step = self._n_steps
        self._n_steps += 1
        if math.isfinite(self.best_loss):
            threshold = max(self.min_delta, self.min_delta_rel * abs(self.best_loss))
        else:
            threshold = self.min_delta
        if val_loss < self.best_loss - threshold:
            self.best_loss = val_loss
            self.best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            self.best_epoch = step
            self._stale = 0
            return True
        self._stale += 1
        if self._stale >= self.patience:
            self.should_stop = True
        return False

    def reset_patience(self) -> None:
        """Restart the patience count, keeping the best loss and state.

        Used when the training regime changes mid-run (e.g. the trunk
        is unfrozen after a linear-probing phase), so the new phase
        gets a full ``patience`` window while the best checkpoint of
        the previous phase stays a candidate.
        """
        self._stale = 0
        self.should_stop = False


def train_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_tensors: tuple[torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor | None],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler
    | torch.optim.lr_scheduler.ReduceLROnPlateau
    | None,
    device: torch.device,
    epochs: int,
    early_stopping: EarlyStopping,
    verbose: bool = False,
    on_epoch_end: Callable[[int, float], None] | None = None,
    warmup_epochs: int = 0,
    grad_clip_norm: float | None = None,
    use_amp: bool = False,
    swa_start_epoch: int | None = None,
    use_sam: bool = False,
    metrics_recorder: Callable[[dict[str, float]], None] | None = None,
    augment: Callable[[torch.Tensor], torch.Tensor] | None = None,
    mixup_alpha: float = 0.0,
    cutmix_alpha: float = 0.0,
    n_classes: int | None = None,
    mix_generator: torch.Generator | None = None,
    ema_decay: float | None = None,
    class_weight_t: torch.Tensor | None = None,
    n_augment_cols: int | None = None,
    on_epoch_start: Callable[[int, nn.Module], None] | None = None,
    compiled_model: nn.Module | None = None,
    amp_dtype: torch.dtype = torch.float16,
    callbacks: Sequence[Any] = (),
) -> nn.Module:
    """Run a classic train + validate loop with early stopping.

    Parameters
    ----------
    model : nn.Module
        Already placed on ``device``.
    train_loader : DataLoader
        Iterates over ``(x, y)`` batches of training data.
    val_tensors : tuple of torch.Tensor
        ``(X_val, y_val)`` tensors already on ``device``.
    criterion : nn.Module
        Loss function, e.g. ``nn.CrossEntropyLoss``.
    optimizer : torch.optim.Optimizer
        Optimizer bound to ``model`` parameters.
    scheduler : torch.optim.lr_scheduler.LRScheduler or ReduceLROnPlateau or None
        Optional LR scheduler. :class:`ReduceLROnPlateau` is stepped on
        validation loss; any other scheduler is stepped once per epoch
        with no argument.
    device : torch.device
        Device on which training is carried out.
    epochs : int
        Maximum number of epochs.
    early_stopping : EarlyStopping
        Tracks the best validation loss and stops training when stale.
    verbose : bool, default=False
        If ``True``, print one line per epoch.
    on_epoch_end : callable or None, default=None
        Called as ``on_epoch_end(epoch, val_loss)`` after each epoch.
    warmup_epochs : int, default=0
        If positive, linearly ramp each optimizer param group's learning
        rate from ``0`` to its configured target over the first
        ``warmup_epochs`` epochs. ``scheduler`` is not stepped during
        warmup.
    grad_clip_norm : float or None, default=None
        If set, clip gradient global L2 norm to this value via
        :func:`torch.nn.utils.clip_grad_norm_`.
    use_amp : bool, default=False
        If ``True`` and ``device.type == "cuda"``, run forward + loss
        under :func:`torch.autocast` (``amp_dtype``) and, in
        ``float16``, use :class:`torch.amp.GradScaler` for backward. On
        CPU this is a no-op.
    swa_start_epoch : int or None, default=None
        If set, maintain a :class:`torch.optim.swa_utils.AveragedModel`
        starting at this epoch (0-indexed). At end of training,
        replaces the best-val checkpoint with the SWA average.
    use_sam : bool, default=False
        If ``True``, assume ``optimizer`` is a
        :class:`~maldideepkit.utils.SAMOptimizer` and run the two-step
        SAM update (roughly doubles compute). Grad clipping is applied
        only on the second gradient.
    metrics_recorder : callable or None, default=None
        If provided, called once per epoch with a dict containing
        ``{"epoch", "train_loss", "val_loss", "lr",
        "mean_grad_norm", "n_grad_updates"}``.
    augment : callable or None, default=None
        If provided, called on each training batch's feature tensor
        after it is moved to ``device`` but before the forward pass.
    mixup_alpha : float, default=0.0
        When ``> 0``, apply MixUp on each training batch with a mix
        coefficient drawn from ``Beta(alpha, alpha)``. Requires
        ``n_classes``. Labels become soft probability distributions.
    cutmix_alpha : float, default=0.0
        When ``> 0``, apply CutMix on each training batch. When both
        ``mixup_alpha`` and ``cutmix_alpha`` are positive a fair coin
        picks between the two per batch.
    n_classes : int or None, default=None
        Required when ``mixup_alpha > 0`` or ``cutmix_alpha > 0``.
    mix_generator : torch.Generator or None, default=None
        Optional seeded RNG for MixUp / CutMix draws.
    ema_decay : float or None, default=None
        When set, maintain an exponential moving average of the
        model parameters: ``ema = decay * ema + (1 - decay) * model``.
        Typical values ``0.99``-``0.9999``. At end of training the
        EMA weights overwrite the base model.
    class_weight_t : torch.Tensor or None, default=None
        Per-class weight tensor (already moved to ``device``). When
        provided, it is used solely for the denominator gather in the
        weighted-mean reduction so that the scikit-learn convention
        "doubling sample_weight is equivalent to duplicating the row"
        composes correctly with class_weight. The actual per-class
        scaling of the per-sample loss is expected to be applied by
        ``criterion`` itself (e.g. via ``nn.CrossEntropyLoss(weight=...)``).
        ``criterion`` should be built with ``reduction='none'``; a
        mean- or sum-reduced criterion is accepted and its scalar loss
        used as-is (it has already performed its own class-weighted
        reduction), except when ``sample_weight`` batches are present,
        where per-sample losses are required and a scalar raises a
        :class:`ValueError`.
    n_augment_cols : int or None, default=None
        Number of leading spectrum columns in each batch. When given
        (dual-input models, whose batch is ``[ X_spec | X_side ]``),
        ``augment`` is applied to the spectrum block only and CutMix
        draws its window inside that block, so spectrum-shaped
        perturbations never reach the packed side features. MixUp is
        unaffected: it already convex-combines every column with a
        single shared ``lam``, which stays consistent with the mixed
        label. ``None`` treats every column as spectrum.
    on_epoch_start : callable or None, default=None
        Called as ``on_epoch_start(epoch, model)`` at the start of each
        epoch, right after ``model.train()`` and before the first
        batch, so it can switch sub-modules back to ``eval()`` or
        toggle ``requires_grad`` for the epoch (e.g. trunk freezing).
    compiled_model : nn.Module or None, default=None
        Compiled wrapper of ``model`` (``torch.compile(model)``, sharing
        its parameters) used for the training and validation forward
        passes. Everything else (train / eval modes, early-stopping
        snapshots, EMA / SWA averages, the returned module) uses
        ``model`` itself, so no state dict carries the ``_orig_mod.``
        prefix of the wrapper.
    amp_dtype : torch.dtype, default=torch.float16
        Autocast dtype when ``use_amp`` is active. ``torch.float16``
        scales the loss with :class:`torch.amp.GradScaler`;
        ``torch.bfloat16`` (the exponent range of ``float32``) runs
        without one.
    callbacks : sequence, default=()
        Objects with the ``on_epoch_start``, ``on_step_end``, and
        ``on_epoch_end`` hooks of :class:`~maldideepkit.utils.Callback`
        (``on_train_start`` and ``on_train_end`` belong to the caller).

    Returns
    -------
    nn.Module
        The input ``model`` with the best-validation weights loaded
        (or the EMA / SWA average when those are enabled: precedence
        EMA > SWA > best_val).
    """
    mix_enabled = mixup_alpha > 0.0 or cutmix_alpha > 0.0
    if mix_enabled and n_classes is None:
        raise ValueError(
            "mixup_alpha / cutmix_alpha require n_classes to be specified."
        )
    if bool(use_sam) and bool(use_amp) and device.type == "cuda":
        import warnings as _warnings

        _warnings.warn(
            "use_sam=True with use_amp=True: SAM's two-pass update is not "
            "compatible with AMP's GradScaler, so this run executes SAM "
            "in FP32 (no AMP speedup). Disable one to silence.",
            stacklevel=2,
        )
    forward = model if compiled_model is None else compiled_model
    if len(val_tensors) == 3:
        X_val, y_val, sw_val_t = val_tensors  # type: ignore[misc]
    else:
        X_val, y_val = val_tensors  # type: ignore[misc]
        sw_val_t = None
    base_lrs = [pg["lr"] for pg in optimizer.param_groups]
    amp_enabled = bool(use_amp) and device.type == "cuda"
    scaler = (
        torch.amp.GradScaler("cuda")
        if amp_enabled and amp_dtype == torch.float16
        else None
    )
    swa_model: torch.optim.swa_utils.AveragedModel | None = None
    swa_updated = False
    ema_model: torch.optim.swa_utils.AveragedModel | None = None
    ema_updated = False
    if ema_decay is not None:
        if not 0.0 < float(ema_decay) < 1.0:
            raise ValueError(f"ema_decay must be in (0, 1); got {ema_decay!r}.")
        decay = float(ema_decay)

        def _ema_avg_fn(avg: torch.Tensor, cur: torch.Tensor, _n: int) -> torch.Tensor:
            return decay * avg + (1.0 - decay) * cur

        ema_model = torch.optim.swa_utils.AveragedModel(model, avg_fn=_ema_avg_fn)

    def _compute_grad_norm(params: list[torch.Tensor]) -> float:
        grads = [p.grad for p in params if p.grad is not None]
        if not grads:
            return 0.0
        norms = torch._foreach_norm(grads, 2.0)
        return float(torch.linalg.vector_norm(torch.stack(norms)).item())

    step = 0
    for epoch in range(epochs):
        if warmup_epochs > 0 and epoch < warmup_epochs:
            scale = (epoch + 1) / warmup_epochs
            for pg, base in zip(optimizer.param_groups, base_lrs, strict=True):
                pg["lr"] = base * scale
        epoch_lr = float(optimizer.param_groups[0]["lr"])

        epoch_loss_sum = 0.0
        epoch_grad_norm_sum = 0.0
        epoch_n_updates = 0

        model.train()
        if on_epoch_start is not None:
            on_epoch_start(epoch, model)
        dispatch(callbacks, "on_epoch_start", model, epoch)
        for batch in train_loader:
            if len(batch) == 3:
                xb, yb, sw_b = batch
                sw_b = sw_b.to(device, non_blocking=True)
            else:
                xb, yb = batch
                sw_b = None
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            if augment is not None:
                if n_augment_cols is None or int(n_augment_cols) >= xb.shape[1]:
                    xb = augment(xb)
                else:
                    n_aug = int(n_augment_cols)
                    xb = torch.cat([augment(xb[:, :n_aug]), xb[:, n_aug:]], dim=1)

            if mix_enabled:
                yb_oh = to_one_hot(yb, int(n_classes))
                use_cutmix = cutmix_alpha > 0.0 and (
                    mixup_alpha == 0.0
                    or torch.rand(1, generator=mix_generator).item() < 0.5
                )
                mix_kwargs: dict[str, Any] = {"generator": mix_generator}
                if use_cutmix:
                    mix_fn = apply_cutmix
                    mix_alpha = cutmix_alpha
                    mix_kwargs["n_spec_cols"] = n_augment_cols
                else:
                    mix_fn = apply_mixup
                    mix_alpha = mixup_alpha
                if sw_b is not None:
                    xb, yb_target, sw_b = mix_fn(  # type: ignore[misc]
                        xb, yb_oh, mix_alpha, sample_weight=sw_b, **mix_kwargs
                    )
                else:
                    xb, yb_target = mix_fn(  # type: ignore[misc]
                        xb, yb_oh, mix_alpha, **mix_kwargs
                    )
            else:
                yb_target = yb

            cw_per_sample = _gather_cw_per_sample(yb_target, class_weight_t)

            def _reduce(
                per_sample: torch.Tensor,
                _sw: torch.Tensor | None = sw_b,
                _cw: torch.Tensor | None = cw_per_sample,
            ) -> torch.Tensor:
                return _weighted_loss_reduce(per_sample, _sw, _cw)

            if use_sam:
                if not isinstance(optimizer, SAMOptimizer):
                    raise TypeError(
                        "use_sam=True requires `optimizer` to be a SAMOptimizer; "
                        f"got {type(optimizer).__name__}."
                    )
                optimizer.zero_grad()
                logits = forward(xb)
                loss = _reduce(criterion(logits, yb_target))
                loss.backward()
                optimizer.first_step(zero_grad=True)

                logits = forward(xb)
                loss = _reduce(criterion(logits, yb_target))
                loss.backward()
                if metrics_recorder is not None or grad_clip_norm is not None:
                    params = [p for g in optimizer.param_groups for p in g["params"]]
                    if metrics_recorder is not None:
                        epoch_grad_norm_sum += _compute_grad_norm(params)
                    if grad_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip_norm)
                optimizer.second_step(zero_grad=True)
            elif amp_enabled:
                optimizer.zero_grad()
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    logits = forward(xb)
                    per_sample = criterion(logits, yb_target)
                loss = _reduce(per_sample.float())
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                if grad_clip_norm is not None or metrics_recorder is not None:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                    params = [p for g in optimizer.param_groups for p in g["params"]]
                    if metrics_recorder is not None:
                        epoch_grad_norm_sum += _compute_grad_norm(params)
                    if grad_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip_norm)
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
            else:
                optimizer.zero_grad()
                logits = forward(xb)
                loss = _reduce(criterion(logits, yb_target))
                loss.backward()
                if metrics_recorder is not None or grad_clip_norm is not None:
                    params = [p for g in optimizer.param_groups for p in g["params"]]
                    if metrics_recorder is not None:
                        epoch_grad_norm_sum += _compute_grad_norm(params)
                    if grad_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip_norm)
                optimizer.step()

            if ema_model is not None:
                ema_model.update_parameters(model)
                ema_updated = True

            if metrics_recorder is not None or callbacks:
                epoch_loss_sum += float(loss.detach().item())
                epoch_n_updates += 1
            dispatch(callbacks, "on_step_end", model, step)
            step += 1

        model.eval()
        val_cw_per_sample = _gather_cw_per_sample(y_val, class_weight_t)
        with torch.no_grad():
            if amp_enabled:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    val_logits = forward(X_val)
                    val_per_sample = criterion(val_logits, y_val)
                val_loss = float(
                    _weighted_loss_reduce(
                        val_per_sample.float(), sw_val_t, val_cw_per_sample
                    ).item()
                )
            else:
                val_logits = forward(X_val)
                val_per_sample = criterion(val_logits, y_val)
                val_loss = float(
                    _weighted_loss_reduce(
                        val_per_sample, sw_val_t, val_cw_per_sample
                    ).item()
                )

        if scheduler is not None and epoch >= warmup_epochs:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        if swa_start_epoch is not None and epoch >= swa_start_epoch:
            if swa_model is None:
                swa_model = torch.optim.swa_utils.AveragedModel(model)
            swa_model.update_parameters(model)
            swa_updated = True

        improved = early_stopping.step(val_loss, model)
        if verbose:
            marker = " *" if improved else ""
            print(f"epoch {epoch + 1}/{epochs}  val_loss={val_loss:.4f}{marker}")
        if on_epoch_end is not None:
            on_epoch_end(epoch, val_loss)
        if metrics_recorder is not None:
            n = max(1, epoch_n_updates)
            metrics_recorder(
                {
                    "epoch": int(epoch),
                    "train_loss": epoch_loss_sum / n,
                    "val_loss": float(val_loss),
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    "mean_grad_norm": epoch_grad_norm_sum / n,
                    "n_grad_updates": int(epoch_n_updates),
                }
            )
        dispatch(
            callbacks,
            "on_epoch_end",
            model,
            epoch,
            {
                "train_loss": epoch_loss_sum / max(1, epoch_n_updates),
                "val_loss": float(val_loss),
                "lr": epoch_lr,
            },
        )
        if early_stopping.should_stop:
            break

    if ema_updated and ema_model is not None:
        if any(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in model.modules()):
            torch.optim.swa_utils.update_bn(train_loader, ema_model, device=device)
        model.load_state_dict(ema_model.module.state_dict())
    elif swa_updated and swa_model is not None:
        if any(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in model.modules()):
            torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
        model.load_state_dict(swa_model.module.state_dict())
    elif early_stopping.best_state is not None:
        model.load_state_dict(early_stopping.best_state)
    model.eval()
    return model
