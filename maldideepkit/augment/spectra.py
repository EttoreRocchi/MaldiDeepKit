"""Composable per-batch augmentations for MALDI-TOF binned spectra.

All transforms operate on a ``torch.Tensor`` of shape ``(batch, n_bins)``.
Each augmentation step is gated by a parameter and is a no-op at its
default value. Applied in order:

1. Additive Gaussian noise (``noise_std``).
2. Per-sample intensity jitter (``intensity_jitter``).
3. Random peak dropout (``peak_dropout_rate``).
4. Per-sample m/z shift (``mz_shift_max_bins``).
5. Spline-based m/z warp (``mz_warp_max_bins`` + ``mz_warp_n_knots``).
6. Gaussian blur (``blur_sigma``).

Only invoked on training batches. All m/z-axis parameters are specified
in *bins*, not Daltons.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from scipy.interpolate import CubicSpline


class SpectrumAugment:
    """Composable per-batch spectrum augmentation.

    Parameters
    ----------
    noise_std : float, default=0.0
        Standard deviation of additive Gaussian noise.
    intensity_jitter : float, default=0.0
        Half-range of the per-sample multiplicative jitter: every
        sample is scaled by ``1 + U(-jitter, jitter)``. Must be in
        ``[0, 1)``.
    peak_dropout_rate : float, default=0.0
        Per-bin Bernoulli zero-out probability. Must be in ``[0, 1)``.
    mz_shift_max_bins : int, default=0
        Per-sample global m/z shift, drawn uniformly in
        ``[-mz_shift_max_bins, +mz_shift_max_bins]`` and applied with
        :func:`torch.roll`. Units are bins. Must be non-negative.
    mz_warp_max_bins : int, default=0
        Peak amplitude (in bins) of a smooth cubic-spline warp of the
        m/z axis. ``0`` disables the warp. Runs on CPU. A warning is
        emitted when the amplitude exceeds 5 % of ``n_bins``, since
        beyond that the boundary clipping starts to dominate the
        augmentation distribution. Must be non-negative.
    mz_warp_n_knots : int, default=10
        Number of interior spline control points for the m/z warp.
        Only used when ``mz_warp_max_bins > 0``. Must be non-negative.
    blur_sigma : float, default=0.0
        Standard deviation (in bins) of a 1-D Gaussian blur along the
        m/z axis. Zero disables the blur.
    random_state : int, optional
        If provided, the transform is seeded for deterministic batches.
        When ``None`` (default), PyTorch's global RNG is used.
    """

    def __init__(
        self,
        noise_std: float = 0.0,
        intensity_jitter: float = 0.0,
        peak_dropout_rate: float = 0.0,
        mz_shift_max_bins: int = 0,
        mz_warp_max_bins: int = 0,
        mz_warp_n_knots: int = 10,
        blur_sigma: float = 0.0,
        random_state: int | None = None,
    ) -> None:
        if noise_std < 0:
            raise ValueError(f"noise_std must be >= 0; got {noise_std!r}.")
        if not 0.0 <= intensity_jitter < 1.0:
            raise ValueError(
                f"intensity_jitter must be in [0, 1); got {intensity_jitter!r}."
            )
        if not 0.0 <= peak_dropout_rate < 1.0:
            raise ValueError(
                f"peak_dropout_rate must be in [0, 1); got {peak_dropout_rate!r}."
            )
        if mz_shift_max_bins < 0:
            raise ValueError(
                f"mz_shift_max_bins must be >= 0; got {mz_shift_max_bins!r}."
            )
        if mz_warp_max_bins < 0:
            raise ValueError(
                f"mz_warp_max_bins must be >= 0; got {mz_warp_max_bins!r}."
            )
        if mz_warp_n_knots < 0:
            raise ValueError(f"mz_warp_n_knots must be >= 0; got {mz_warp_n_knots!r}.")
        if blur_sigma < 0:
            raise ValueError(f"blur_sigma must be >= 0; got {blur_sigma!r}.")
        self.noise_std = float(noise_std)
        self.intensity_jitter = float(intensity_jitter)
        self.peak_dropout_rate = float(peak_dropout_rate)
        self.mz_shift_max_bins = int(mz_shift_max_bins)
        self.mz_warp_max_bins = int(mz_warp_max_bins)
        self.mz_warp_n_knots = int(mz_warp_n_knots)
        self.blur_sigma = float(blur_sigma)
        self.random_state = random_state
        self._generator: torch.Generator | None = None
        self._np_rng: np.random.Generator | None = None
        self._blur_kernel: torch.Tensor | None = None

    def _generator_for(self, device: torch.device) -> torch.Generator | None:
        if self.random_state is None:
            return None
        if self._generator is None or self._generator.device != device:
            self._generator = torch.Generator(device=device)
            self._generator.manual_seed(int(self.random_state))
        return self._generator

    def _numpy_generator(self) -> np.random.Generator:
        if self._np_rng is None:
            self._np_rng = np.random.default_rng(self.random_state)
        return self._np_rng

    def _is_identity(self) -> bool:
        return (
            self.noise_std == 0.0
            and self.intensity_jitter == 0.0
            and self.peak_dropout_rate == 0.0
            and self.mz_shift_max_bins == 0
            and self.mz_warp_max_bins == 0
            and self.blur_sigma == 0.0
        )

    def _apply_mz_shift(
        self, X: torch.Tensor, gen: torch.Generator | None
    ) -> torch.Tensor:
        k = self.mz_shift_max_bins
        shifts = torch.randint(
            low=-k,
            high=k + 1,
            size=(X.shape[0],),
            generator=gen,
            device=X.device,
        )
        out = torch.empty_like(X)
        for i in range(X.shape[0]):
            out[i] = torch.roll(X[i], shifts=int(shifts[i].item()), dims=0)
        return out

    def _apply_spline_warp(self, X: torch.Tensor) -> torch.Tensor:
        rng = self._numpy_generator()
        device = X.device
        x_np = X.detach().cpu().numpy().copy()
        n_samples, n_bins = x_np.shape
        if self.mz_warp_max_bins > 0.05 * n_bins:
            import warnings as _warnings

            _warnings.warn(
                f"mz_warp_max_bins={self.mz_warp_max_bins} exceeds 5% of "
                f"n_bins={n_bins}; warped indices outside the support are "
                "clipped before interpolation, which produces asymmetric "
                "edge flattening. Consider reducing mz_warp_max_bins.",
                stacklevel=2,
            )
        original_indices = np.arange(n_bins, dtype=np.float64)
        n_knots = self.mz_warp_n_knots
        knot_positions = np.linspace(0, n_bins - 1, n_knots + 2)
        for i in range(n_samples):
            knot_shifts = np.zeros(n_knots + 2)
            if n_knots > 0:
                knot_shifts[1:-1] = rng.uniform(
                    -self.mz_warp_max_bins,
                    self.mz_warp_max_bins,
                    size=n_knots,
                )
            spline = CubicSpline(knot_positions, knot_shifts, bc_type="clamped")
            smooth_shifts = spline(original_indices)
            warped = np.clip(original_indices + smooth_shifts, 0, n_bins - 1)
            x_np[i] = np.interp(original_indices, warped, x_np[i])
        return torch.from_numpy(x_np).to(device=device, dtype=X.dtype)

    def _build_blur_kernel(
        self, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        if (
            self._blur_kernel is not None
            and self._blur_kernel.device == device
            and self._blur_kernel.dtype == dtype
        ):
            return self._blur_kernel
        radius = int(math.ceil(3.0 * self.blur_sigma))
        xs = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
        kernel = torch.exp(-0.5 * (xs / self.blur_sigma) ** 2)
        kernel = kernel / kernel.sum()
        self._blur_kernel = kernel.view(1, 1, -1)
        return self._blur_kernel

    def _apply_blur(self, X: torch.Tensor) -> torch.Tensor:
        kernel = self._build_blur_kernel(X.device, X.dtype)
        padding = kernel.shape[-1] // 2
        x3 = X.unsqueeze(1)
        out = torch.nn.functional.conv1d(x3, kernel, padding=padding)
        return out.squeeze(1)

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        """Apply the enabled augmentations to ``X``.

        Returns the input unchanged if no augmentation is enabled.
        """
        if self._is_identity():
            return X
        gen = self._generator_for(X.device)
        out = X

        if self.noise_std > 0.0:
            noise = (
                torch.randn(out.shape, generator=gen, device=out.device)
                * self.noise_std
            )
            out = out + noise

        if self.intensity_jitter > 0.0:
            jitter = (
                torch.rand((out.shape[0], 1), generator=gen, device=out.device) * 2.0
                - 1.0
            ) * self.intensity_jitter
            out = out * (1.0 + jitter)

        if self.peak_dropout_rate > 0.0:
            keep = 1.0 - self.peak_dropout_rate
            mask = torch.empty(out.shape, device=out.device).bernoulli_(
                keep, generator=gen
            )
            out = out * mask

        if self.mz_shift_max_bins > 0:
            out = self._apply_mz_shift(out, gen)

        if self.mz_warp_max_bins > 0:
            out = self._apply_spline_warp(out)

        if self.blur_sigma > 0.0:
            out = self._apply_blur(out)

        return out

    def __repr__(self) -> str:
        return (
            f"SpectrumAugment(noise_std={self.noise_std}, "
            f"intensity_jitter={self.intensity_jitter}, "
            f"peak_dropout_rate={self.peak_dropout_rate}, "
            f"mz_shift_max_bins={self.mz_shift_max_bins}, "
            f"mz_warp_max_bins={self.mz_warp_max_bins}, "
            f"mz_warp_n_knots={self.mz_warp_n_knots}, "
            f"blur_sigma={self.blur_sigma})"
        )
