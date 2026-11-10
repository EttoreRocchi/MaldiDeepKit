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
    """Return ``y`` as a float one-hot tensor of shape ``(batch, n_classes)``.

    Parameters
    ----------
    y : torch.Tensor of shape (batch,)
        Integer class indices.
    n_classes : int
        Number of classes.

    Returns
    -------
    torch.Tensor of shape (batch, n_classes)
        ``float32`` one-hot encoding of ``y``.
    """
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
    *,
    sample_weight: torch.Tensor | None = None,
) -> (
    tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
):
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
    sample_weight : torch.Tensor or None, keyword-only, default=None
        Optional per-sample weight tensor of shape ``(batch,)``. When
        given, the function returns a 3-tuple whose third element is
        ``lam * sample_weight + (1 - lam) * sample_weight[perm]`` (the
        same convex combination applied to ``y_oh``). When ``None``
        the function returns a 2-tuple for backward compatibility.

    Returns
    -------
    tuple of torch.Tensor
        ``(x_mixed, y_mixed)`` or ``(x_mixed, y_mixed, sw_mixed)``
        when ``sample_weight`` is provided. ``x_mixed`` and ``y_mixed``
        have the same shape as the inputs.
    """
    if alpha <= 0:
        raise ValueError(f"mixup alpha must be > 0; got {alpha!r}.")
    lam = _sample_beta(alpha, generator)
    perm = torch.randperm(x.shape[0], generator=generator).to(x.device)
    x_mixed = lam * x + (1.0 - lam) * x[perm]
    y_mixed = lam * y_oh + (1.0 - lam) * y_oh[perm]
    if sample_weight is None:
        return x_mixed, y_mixed
    sw = sample_weight.to(device=x.device)
    sw_mixed = lam * sw + (1.0 - lam) * sw[perm]
    return x_mixed, y_mixed, sw_mixed


def apply_cutmix(
    x: torch.Tensor,
    y_oh: torch.Tensor,
    alpha: float,
    generator: torch.Generator | None = None,
    *,
    sample_weight: torch.Tensor | None = None,
    n_spec_cols: int | None = None,
) -> (
    tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
):
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
    n_spec_cols : int or None, keyword-only, default=None
        Number of leading spectrum columns. When given (dual-input
        models, where ``x`` is ``[ X_spec | X_side ]``), the spliced
        window is drawn **inside the spectrum block only** and the
        window fraction is taken relative to ``n_spec_cols`` rather
        than the full width. The trailing side columns are instead
        convex-combined with the same ``effective_lam`` and the same
        permutation, so a contiguous m/z window never straddles the
        spectrum / side boundary and ``(x, y)`` stay consistent.
        ``None`` treats every column as spectrum.
    sample_weight : torch.Tensor or None, keyword-only, default=None
        Optional per-sample weight tensor of shape ``(batch,)``. When
        given, the function returns a 3-tuple whose third element is
        ``effective_lam * sample_weight + (1 - effective_lam) *
        sample_weight[perm]`` (matching how labels are mixed). When
        ``None`` the function returns a 2-tuple for backward
        compatibility.

    Returns
    -------
    tuple of torch.Tensor
        ``(x_mixed, y_mixed)`` or ``(x_mixed, y_mixed, sw_mixed)``
        when ``sample_weight`` is provided.
    """
    if alpha <= 0:
        raise ValueError(f"cutmix alpha must be > 0; got {alpha!r}.")
    batch, n_bins = x.shape
    n_cut = n_bins if n_spec_cols is None else min(int(n_spec_cols), n_bins)
    if n_cut <= 0:
        raise ValueError(f"n_spec_cols must be > 0; got {n_spec_cols!r}.")
    lam = _sample_beta(alpha, generator)
    window = int(round(n_cut * (1.0 - lam)))
    window = max(0, min(window, n_cut))
    if window == 0:
        if sample_weight is None:
            return x.clone(), y_oh.clone()
        return x.clone(), y_oh.clone(), sample_weight.to(device=x.device).clone()
    start = int(torch.randint(0, n_cut - window + 1, (1,), generator=generator).item())
    perm = torch.randperm(batch, generator=generator).to(x.device)
    x_mixed = x.clone()
    x_mixed[:, start : start + window] = x[perm][:, start : start + window]
    effective_lam = 1.0 - window / n_cut
    if n_cut < n_bins:
        x_mixed[:, n_cut:] = (
            effective_lam * x[:, n_cut:] + (1.0 - effective_lam) * x[perm][:, n_cut:]
        )
    y_mixed = effective_lam * y_oh + (1.0 - effective_lam) * y_oh[perm]
    if sample_weight is None:
        return x_mixed, y_mixed
    sw = sample_weight.to(device=x.device)
    sw_mixed = effective_lam * sw + (1.0 - effective_lam) * sw[perm]
    return x_mixed, y_mixed, sw_mixed
