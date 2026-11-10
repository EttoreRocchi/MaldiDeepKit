"""Training hooks shared by the classifiers and the pre-training encoder.

:class:`Callback` is what ``callbacks=`` takes on every classifier and on
:class:`~maldideepkit.pretraining.MaldiSpectrumEncoder`.
:class:`~maldideepkit.pretraining.Pretext` has the same hooks, so a
pre-text task can act on its own training run.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from torch import nn

_HOOKS = (
    "on_train_start",
    "on_epoch_start",
    "on_step_end",
    "on_epoch_end",
    "on_train_end",
)


class TrainingHooks:
    """The optional hooks of a training run (no-ops by default).

    No hook runs with ``epochs=0``. A classifier retrained by
    ``retry_on_val_auroc_below`` calls ``on_train_start`` and
    ``on_train_end`` once per attempt.
    """

    def on_train_start(self, model: nn.Module, estimator: Any) -> None:
        """Run once, before the first epoch and before any ``torch.compile``.

        Parameters
        ----------
        model : torch.nn.Module
            The freshly built module, on its training device.
        estimator : BaseSpectralClassifier or MaldiSpectrumEncoder
            The estimator being fitted, with its fitted preprocessing
            already set. It must not be modified.
        """

    def on_epoch_start(self, model: nn.Module, epoch: int) -> None:
        """Run after ``model.train()``, before the first batch of ``epoch``."""

    def on_step_end(self, model: nn.Module, step: int) -> None:
        """Run after each optimizer step (``step`` counts from 0 over the run)."""

    def on_epoch_end(
        self, model: nn.Module, epoch: int, logs: dict[str, float]
    ) -> None:
        """Run after validation and early stopping of ``epoch``.

        Parameters
        ----------
        model : torch.nn.Module
            The module, with the weights of the end of the epoch.
        epoch : int
            Epoch index, from 0.
        logs : dict
            ``"train_loss"`` (mean over the batches), ``"val_loss"`` and
            ``"lr"`` (of the first parameter group, during the epoch).
        """

    def on_train_end(self, model: nn.Module) -> None:
        """Run once at the end, with the deployed weights loaded in ``model``."""


class Callback(TrainingHooks):
    """Base class of the objects accepted by ``callbacks=``.

    Subclasses override the hooks they need: logging to an external
    tracker, intermediate checkpoints, parameter schedules, a per-batch
    learning-rate scheduler. Random draws in a hook change the run, as in
    any training code.

    A clone of the estimator (:func:`~sklearn.base.clone`,
    :class:`~sklearn.model_selection.GridSearchCV`) shares the same
    instance, so one logger can see every fit. To store a callback with
    :meth:`~maldideepkit.BaseSpectralClassifier.save`, define its class
    at module level and keep each constructor argument in an attribute
    of the same name; otherwise use ``set_params(callbacks=None)`` before
    saving.

    Examples
    --------
    >>> from maldideepkit import Callback, MaldiMLPClassifier
    >>> class History(Callback):
    ...     def __init__(self):
    ...         self.val_losses = []
    ...
    ...     def on_epoch_end(self, model, epoch, logs):
    ...         self.val_losses.append(logs["val_loss"])
    >>> clf = MaldiMLPClassifier(callbacks=[History()])
    """

    def __deepcopy__(self, memo: dict[int, Any]) -> Callback:
        return self


def callback_list(callbacks: Any) -> list[Any]:
    """Return ``callbacks`` (``None``, one callback, or a sequence) as a list.

    Raises
    ------
    TypeError
        If an element defines none of the hooks of :class:`Callback`.
    """
    if callbacks is None:
        return []
    items = list(callbacks) if isinstance(callbacks, Sequence) else [callbacks]
    for item in items:
        if not any(callable(getattr(item, hook, None)) for hook in _HOOKS):
            raise TypeError(
                "callbacks must be Callback instances (or objects defining at "
                f"least one of {_HOOKS}); got {type(item).__name__}."
            )
    return items


def dispatch(callbacks: Sequence[Any], hook: str, *args: Any) -> None:
    """Call ``hook`` on every element of ``callbacks`` that defines it."""
    for callback in callbacks:
        method = getattr(callback, hook, None)
        if method is not None:
            method(*args)


__all__ = ["Callback", "TrainingHooks", "callback_list", "dispatch"]
