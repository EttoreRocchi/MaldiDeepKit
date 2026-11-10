"""Optimizer and learning-rate scheduler of a training run.

The classifiers and :class:`~maldideepkit.pretraining.MaldiSpectrumEncoder`
share these rules for their ``optimizer=`` and ``scheduler=`` parameters.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from typing import Any

import torch

from .sam import SAMOptimizer

SCHEDULERS = ("cosine", "constant")


def _optimizer_classes() -> dict[str, type[torch.optim.Optimizer]]:
    """Map the lower-case name of every :mod:`torch.optim` optimizer to its class."""
    return {
        name.lower(): obj
        for name, obj in vars(torch.optim).items()
        if isinstance(obj, type)
        and issubclass(obj, torch.optim.Optimizer)
        and obj is not torch.optim.Optimizer
    }


def resolve_optimizer(spec: Any, weight_decay: float) -> Callable[..., Any]:
    """Return the factory of the optimizer described by ``spec``.

    Parameters
    ----------
    spec : None, str or callable
        ``None`` (:class:`torch.optim.AdamW` when ``weight_decay > 0``,
        else :class:`torch.optim.Adam`), the case-insensitive name of a
        :mod:`torch.optim` class, or a class or factory called as
        ``spec(params, ...)``.
    weight_decay : float
        The weight decay of the run.

    Returns
    -------
    callable
        The class or factory.

    Raises
    ------
    ValueError
        If ``spec`` is an unknown name.
    TypeError
        If ``spec`` is neither ``None``, a string nor a callable.
    """
    if spec is None:
        return torch.optim.AdamW if float(weight_decay) > 0 else torch.optim.Adam
    if isinstance(spec, str):
        classes = _optimizer_classes()
        if spec.lower() not in classes:
            raise ValueError(
                f"Unknown optimizer={spec!r}; expected one of {sorted(classes)}, a "
                "class, or a factory."
            )
        return classes[spec.lower()]
    if callable(spec):
        return spec
    raise TypeError(
        "optimizer must be None, a torch.optim name, a class, or a factory; got "
        f"{type(spec).__name__}."
    )


def _accepted_kwargs(factory: Callable[..., Any], **kwargs: Any) -> dict[str, Any]:
    """Return the ``kwargs`` that ``factory`` accepts and does not already fix.

    A keyword bound by :func:`functools.partial` wins over the estimator's
    own value, so ``partial(SGD, lr=0.1)`` keeps ``lr=0.1``.
    """
    bound = factory.keywords if isinstance(factory, functools.partial) else {}
    try:
        parameters = inspect.signature(factory).parameters
    except (TypeError, ValueError):
        return {k: v for k, v in kwargs.items() if k not in bound}
    takes_any = any(p.kind is p.VAR_KEYWORD for p in parameters.values())
    return {
        k: v
        for k, v in kwargs.items()
        if k not in bound and (takes_any or k in parameters)
    }


def build_optimizer(
    spec: Any,
    params: Any,
    *,
    learning_rate: float,
    weight_decay: float,
    use_sam: bool = False,
    sam_rho: float = 0.05,
) -> torch.optim.Optimizer:
    """Build the optimizer of a run (see :func:`resolve_optimizer`).

    ``lr`` and ``weight_decay`` reach the factory unless it does not
    accept them or a :func:`functools.partial` already fixes them.

    Parameters
    ----------
    spec : None, str or callable
        See :func:`resolve_optimizer`.
    params : iterable
        Parameters or parameter groups.
    learning_rate, weight_decay : float
        The estimator's values.
    use_sam : bool, default=False
        Wrap the optimizer in :class:`~maldideepkit.utils.SAMOptimizer`.
    sam_rho : float, default=0.05
        SAM neighbourhood radius.

    Returns
    -------
    torch.optim.Optimizer
        The optimizer.
    """
    factory = resolve_optimizer(spec, weight_decay)
    kwargs = _accepted_kwargs(
        factory, lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    if use_sam:
        return SAMOptimizer(
            params, base_optimizer=factory, rho=float(sam_rho), **kwargs
        )
    return factory(params, **kwargs)


def check_scheduler(spec: Any) -> None:
    """Raise unless ``spec`` is a valid ``scheduler=`` value (see :func:`build_scheduler`)."""
    if spec is None or callable(spec):
        return
    if isinstance(spec, str):
        if spec not in SCHEDULERS:
            raise ValueError(
                f"Unknown scheduler={spec!r}; expected one of {SCHEDULERS}, a class, "
                "or a factory."
            )
        return
    raise TypeError(
        f"scheduler must be None, one of {SCHEDULERS}, a class, or a factory; got "
        f"{type(spec).__name__}."
    )


def build_scheduler(
    spec: Any,
    optimizer: torch.optim.Optimizer,
    *,
    epochs: int,
    warmup_epochs: int = 0,
) -> Any | None:
    """Build the learning-rate scheduler of a run.

    Parameters
    ----------
    spec : None, str or callable
        ``None`` or ``"cosine"``: cosine annealing to ``1e-6`` over the
        epochs after the warmup. ``"constant"``: no scheduler. A class or
        factory is called as ``spec(optimizer)``, so one that needs more
        arguments goes through :func:`functools.partial`.
    optimizer : torch.optim.Optimizer
        The optimizer.
    epochs : int
        Maximum number of epochs.
    warmup_epochs : int, default=0
        Epochs of linear warmup, during which the scheduler is not
        stepped.

    Returns
    -------
    scheduler or None
        Stepped once per epoch after the warmup (a
        :class:`~torch.optim.lr_scheduler.ReduceLROnPlateau` with the
        validation loss). A per-batch schedule fits in the
        ``on_step_end`` hook of a callback.
    """
    check_scheduler(spec)
    if spec is None or spec == "cosine":
        warmup = max(0, int(warmup_epochs))
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, int(epochs) - warmup), eta_min=1e-6
        )
    if spec == "constant":
        return None
    return spec(optimizer)


__all__ = ["SCHEDULERS", "build_optimizer", "build_scheduler", "check_scheduler"]
