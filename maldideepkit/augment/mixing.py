"""MixUp and CutMix for 1-D binned MALDI-TOF spectra.

Both transforms operate on a batch of features ``x`` with shape
``(batch, n_bins)`` and one-hot targets ``y_oh`` with shape
``(batch, n_classes)``, and return a mixed ``(x, y_soft)`` pair.

- MixUp: ``x = lam * x_i + (1 - lam) * x_j``.
- CutMix: splice a contiguous m/z window from a shuffled sample
  into the original; labels mixed by window fraction.

Both draw the mix coefficient from ``Beta(alpha, alpha)``.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def to_one_hot(y: torch.Tensor, n_classes: int) -> torch.Tensor:
    """Return ``y`` as a float one-hot tensor of shape ``(batch, n_classes)``."""
    return F.one_hot(y.long(), num_classes=n_classes).to(dtype=torch.float32)


def _sample_beta(alpha: float, generator: torch.Generator | None = None) -> float:
    """Draw a single ``Beta(alpha, alpha)`` sample."""
    if generator is None:
        return float(np.random.beta(alpha, alpha))
    a = torch.tensor([float(alpha)], dtype=torch.float64)
    x = torch._standard_gamma(a, generator=generator)
    y = torch._standard_gamma(a, generator=generator)
    return float((x / (x + y)).item())


def apply_mixup(
    x: torch.Tensor,
    y_oh: torch.Tensor,
    alpha: float,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mixup: convex-combine two random permutations of the batch.

    Parameters
    ----------
    x : torch.Tensor
        Feature tensor of shape ``(batch, n_bins)``.
    y_oh : torch.Tensor
        One-hot (or soft) target tensor of shape ``(batch, n_classes)``.
    alpha : float
        Beta-distribution parameter (``Beta(alpha, alpha)``). Typical
        values 0.1-0.4 for tabular-ish inputs. Must be ``> 0``.
    generator : torch.Generator or None, default=None
        Seeded RNG for reproducibility.

    Returns
    -------
    tuple of torch.Tensor
        ``(x_mixed, y_mixed)`` with the same shapes as the inputs.
    """
    if alpha <= 0:
        raise ValueError(f"mixup alpha must be > 0; got {alpha!r}.")
    lam = _sample_beta(alpha, generator)
    perm = torch.randperm(x.shape[0], generator=generator).to(x.device)
    x_mixed = lam * x + (1.0 - lam) * x[perm]
    y_mixed = lam * y_oh + (1.0 - lam) * y_oh[perm]
    return x_mixed, y_mixed


def apply_cutmix(
    x: torch.Tensor,
    y_oh: torch.Tensor,
    alpha: float,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CutMix on 1-D spectra: splice a contiguous m/z window.

    A window of length ``w = round(n_bins * (1 - lam))`` is drawn
    uniformly along the m/z axis and copied from the shuffled sample
    into the original. Labels are mixed by the window fraction.

    Parameters
    ----------
    x : torch.Tensor
        Feature tensor of shape ``(batch, n_bins)``.
    y_oh : torch.Tensor
        One-hot (or soft) target tensor of shape ``(batch, n_classes)``.
    alpha : float
        Beta-distribution parameter (``Beta(alpha, alpha)``). Typical
        value 1.0 (uniform over window fractions). Must be ``> 0``.
    generator : torch.Generator or None, default=None
        Seeded RNG for reproducibility.

    Returns
    -------
    tuple of torch.Tensor
        ``(x_mixed, y_mixed)`` with the same shapes as the inputs.
    """
    if alpha <= 0:
        raise ValueError(f"cutmix alpha must be > 0; got {alpha!r}.")
    batch, n_bins = x.shape
    lam = _sample_beta(alpha, generator)
    window = int(round(n_bins * (1.0 - lam)))
    window = max(0, min(window, n_bins))
    if window == 0:
        return x.clone(), y_oh.clone()
    start = int(torch.randint(0, n_bins - window + 1, (1,), generator=generator).item())
    perm = torch.randperm(batch, generator=generator).to(x.device)
    x_mixed = x.clone()
    x_mixed[:, start : start + window] = x[perm][:, start : start + window]
    effective_lam = 1.0 - window / n_bins
    y_mixed = effective_lam * y_oh + (1.0 - effective_lam) * y_oh[perm]
    return x_mixed, y_mixed
