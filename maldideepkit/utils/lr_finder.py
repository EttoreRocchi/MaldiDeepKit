"""Learning-rate finder.

Sweeps the learning rate geometrically over a small training run and
records the smoothed loss at each step. The minimum of the smoothed
loss curve's gradient gives a reasonable starting point for the base
learning rate.

Diagnostic only; not wired into
:meth:`~maldideepkit.BaseSpectralClassifier.fit`.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
from torch import nn

from .optim import build_optimizer
from .reproducibility import resolve_device, seed_everything

if TYPE_CHECKING:
    from ..base.classifier import BaseSpectralClassifier


def find_lr(
    classifier: "BaseSpectralClassifier",
    X: Any,
    y: Any,
    *,
    start_lr: float = 1e-8,
    end_lr: float = 1.0,
    num_iter: int = 200,
    smoothing: float = 0.98,
    divergence_factor: float = 4.0,
    plot: bool = False,
) -> dict[str, Any]:
    """Sweep learning rate geometrically and return the LR / loss curve.

    Parameters
    ----------
    classifier : BaseSpectralClassifier
        An unfitted classifier configured with the desired architecture
        / batch_size / loss. Its ``learning_rate`` is ignored (the sweep
        sets it at every step) and its weights are reset at the start
        of every call. ``init_from`` is not applied: the sweep
        runs a randomly initialised model with preprocessing fitted on
        ``X`` (a :class:`UserWarning` is emitted when it is set).
    X, y : array-like
        Training data. Only enough batches to cover ``num_iter`` steps
        are consumed.
    start_lr, end_lr : float, default=1e-8, 1.0
        Bounds of the geometric LR sweep.
    num_iter : int, default=200
        Number of steps in the sweep.
    smoothing : float, default=0.98
        Exponential-moving-average factor applied to the per-step loss
        (0 = no smoothing, 0.99 = heavy smoothing).
    divergence_factor : float, default=4.0
        Stop early once the smoothed loss exceeds
        ``divergence_factor * min_smoothed_loss``.
    plot : bool, default=False
        If ``True``, render a matplotlib plot. ``matplotlib`` is only
        imported when this is true.

    Returns
    -------
    dict
        ``{"lrs": np.ndarray, "losses": np.ndarray, "suggested_lr": float}``.
        ``suggested_lr`` is the LR at the steepest-descent point of the
        smoothed loss curve.
    """
    from ..base.data import make_loaders

    if getattr(classifier, "init_from", None) is not None:
        warnings.warn(
            "find_lr ignores init_from: the sweep uses a randomly "
            "initialised model with its own preprocessing fitted on X, "
            "not the pre-trained trunk and the encoder's preprocessing.",
            UserWarning,
            stacklevel=2,
        )
    seed_everything(int(classifier.random_state))
    device = resolve_device(classifier.device)

    X_np, y_encoded = classifier._prepare_inputs(X, y)
    train_loader, _, stats = make_loaders(
        X_np,
        y_encoded,
        batch_size=int(classifier.batch_size),
        val_size=float(classifier.val_fraction),
        random_state=int(classifier.random_state),
        standardize=bool(classifier.standardize),
        input_transform=classifier.input_transform,
        n_transform_cols=classifier._n_transform_cols(),
    )
    classifier.feature_mean_ = stats["mean"]
    classifier.feature_std_ = stats["std"]
    model = classifier._build_model().to(device)

    class_weight = classifier._compute_class_weight(y_encoded)
    if class_weight is not None:
        class_weight = class_weight.to(device)
    criterion = classifier._build_criterion(class_weight, reduction="mean")
    if isinstance(criterion, nn.Module):
        criterion = criterion.to(device)
    optimizer = build_optimizer(
        classifier.optimizer,
        model.parameters(),
        learning_rate=start_lr,
        weight_decay=0.0,
    )

    lrs: list[float] = []
    losses: list[float] = []
    best_smoothed = float("inf")
    smoothed = 0.0
    log_step = (np.log(end_lr) - np.log(start_lr)) / max(1, num_iter - 1)

    model.train()
    step = 0
    data_iter = iter(train_loader)
    while step < num_iter:
        try:
            xb, yb = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            xb, yb = next(data_iter)
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        lr = float(np.exp(np.log(start_lr) + log_step * step))
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        if loss.ndim > 0:
            # A custom per-row loss.
            loss = loss.mean()
        loss.backward()
        optimizer.step()

        raw = float(loss.detach().item())
        smoothed = smoothing * smoothed + (1 - smoothing) * raw
        debiased = smoothed / (1 - smoothing ** (step + 1)) if smoothing > 0 else raw

        lrs.append(lr)
        losses.append(debiased)

        best_smoothed = min(best_smoothed, debiased)
        if debiased > divergence_factor * best_smoothed and step > 5:
            break

        step += 1

    lrs_arr = np.asarray(lrs)
    losses_arr = np.asarray(losses)
    if len(lrs_arr) < 3:
        suggested_idx = int(np.argmin(losses_arr))
    else:
        grads = np.gradient(losses_arr, np.log(lrs_arr))
        suggested_idx = int(np.argmin(grads))
    suggested_lr = float(lrs_arr[suggested_idx])

    if plot:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(lrs_arr, losses_arr)
        ax.set_xscale("log")
        ax.set_xlabel("learning rate")
        ax.set_ylabel("smoothed loss")
        ax.axvline(
            suggested_lr,
            color="r",
            linestyle="--",
            label=f"suggested = {suggested_lr:.2e}",
        )
        ax.legend()
        fig.tight_layout()
        plt.show()

    return {"lrs": lrs_arr, "losses": losses_arr, "suggested_lr": suggested_lr}
