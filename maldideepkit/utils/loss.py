"""Loss functions used by :class:`BaseSpectralClassifier`."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class FocalLoss(nn.Module):
    r"""Multi-class focal loss with optional class weighting and label smoothing.

    Implements

    .. math::

        L = - (1 - p_t)^\gamma \log p_t

    where :math:`p_t` is the predicted probability of the true class.
    At :math:`\gamma = 0` and ``label_smoothing=0`` this reduces to
    :class:`~torch.nn.CrossEntropyLoss`.

    Parameters
    ----------
    weight : torch.Tensor or None, default=None
        Per-class weight tensor of shape ``(n_classes,)``. Applied to
        every sample by gathering at its target index (matches the
        :class:`CrossEntropyLoss` convention for ``weight``).
    gamma : float, default=2.0
        Focusing parameter. ``0`` degrades to cross-entropy; ``2`` is
        the value used in Lin et al. 2017.
    label_smoothing : float, default=0.0
        Target smoothing in ``[0, 1)``. At ``0.0`` the target is a
        one-hot vector; otherwise the target distribution becomes
        ``(1 - eps) * one_hot + eps / n_classes``.
    reduction : {"mean", "sum", "none"}, default="mean"
        How to reduce the per-sample loss tensor.
    """

    def __init__(
        self,
        weight: torch.Tensor | None = None,
        gamma: float = 2.0,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if gamma < 0:
            raise ValueError(f"gamma must be >= 0; got {gamma!r}.")
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError(
                f"label_smoothing must be in [0, 1); got {label_smoothing!r}."
            )
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(
                f"reduction must be 'mean', 'sum', or 'none'; got {reduction!r}."
            )
        self.register_buffer(
            "class_weight",
            weight.detach().clone() if weight is not None else None,
            persistent=False,
        )
        self.gamma = float(gamma)
        self.label_smoothing = float(label_smoothing)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        r"""Compute focal loss for ``(N, C)`` logits.

        Accepts either integer targets of shape ``(N,)`` or a soft
        probability distribution of shape ``(N, C)`` (as produced by
        MixUp / CutMix). When soft targets are passed the loss
        becomes

        .. math::

            L = - \sum_c t_c \, (1 - p_c)^\gamma \log p_c

        ``label_smoothing`` is ignored on the soft-target path.

        Class weighting follows the :class:`~torch.nn.CrossEntropyLoss`
        convention with ``reduction="mean"``: the per-sample weight is
        ``weight[y_i]`` (or ``Σ_c t_c · weight_c`` for soft targets),
        and the mean reduction divides by ``Σ_i sample_weight_i``
        rather than ``N``.
        """
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        n_classes = logits.shape[-1]
        sample_weight: torch.Tensor | None = None

        if target.dim() == 2:
            smooth = target.to(dtype=log_probs.dtype)
            focal_per_class = (1.0 - probs).clamp_min(1e-12).pow(self.gamma)
            per_class_loss = -smooth * focal_per_class * log_probs
            loss = per_class_loss.sum(dim=-1)
            if self.class_weight is not None:
                w = self.class_weight.to(loss.device)
                sample_weight = (smooth * w).sum(dim=-1)
                loss = loss * sample_weight
        elif self.label_smoothing == 0.0:
            logpt = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
            pt = probs.gather(1, target.unsqueeze(1)).squeeze(1)
            focal_term = (1.0 - pt).clamp_min(1e-12).pow(self.gamma)
            loss = -focal_term * logpt
            if self.class_weight is not None:
                sample_weight = self.class_weight.to(loss.device).gather(0, target)
                loss = loss * sample_weight
        else:
            eps = self.label_smoothing
            smooth = torch.full_like(probs, eps / n_classes)
            smooth.scatter_(
                1,
                target.unsqueeze(1),
                smooth.gather(1, target.unsqueeze(1)) + (1.0 - eps),
            )
            focal_per_class = (1.0 - probs).clamp_min(1e-12).pow(self.gamma)
            per_class_loss = -smooth * focal_per_class * log_probs
            loss = per_class_loss.sum(dim=-1)
            if self.class_weight is not None:
                sample_weight = self.class_weight.to(loss.device).gather(0, target)
                loss = loss * sample_weight

        if self.reduction == "mean":
            if sample_weight is not None:
                denom = sample_weight.sum().clamp_min(1e-12)
                return loss.sum() / denom
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

    def extra_repr(self) -> str:
        """Return a string with the focal-loss hyperparameters for ``repr``."""
        return (
            f"gamma={self.gamma}, label_smoothing={self.label_smoothing}, "
            f"reduction={self.reduction!r}"
        )
