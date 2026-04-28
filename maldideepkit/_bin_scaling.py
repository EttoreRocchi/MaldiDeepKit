"""Helpers to scale architectural hyperparameters with spectrum layout.

The package defaults are calibrated for ``bin_width=3`` over the
2000-20000 Da range (``input_dim=6000``). For other bin widths,
:func:`scale_odd_kernel` adjusts the conv kernel size inversely so
the receptive field in Daltons stays comparable.

See ``docs/source/spectrum_scaling.rst`` for the per-species
peak-width statistics underpinning the defaults.
"""

from __future__ import annotations

REFERENCE_BIN_WIDTH = 3
REFERENCE_CONV_KERNEL = 7


def scale_odd_kernel(
    bin_width: int,
    *,
    reference_kernel: int = REFERENCE_CONV_KERNEL,
    reference_bin_width: int = REFERENCE_BIN_WIDTH,
    min_kernel: int = 3,
) -> int:
    """Return an odd conv kernel scaled inversely with ``bin_width``.

    Used by :class:`~maldideepkit.MaldiCNNClassifier` and
    :class:`~maldideepkit.MaldiResNetClassifier` to adjust the first
    convolutional layer's receptive field for non-default bin widths.

    Parameters
    ----------
    bin_width : int
        The target bin width in Daltons.
    reference_kernel : int, default=7
        Kernel size used at ``reference_bin_width`` (the package default).
    reference_bin_width : int, default=3
        Reference bin width at which ``reference_kernel`` was chosen.
    min_kernel : int, default=3
        Lower bound on the returned kernel size.

    Returns
    -------
    int
        Odd kernel size ``>= min_kernel`` closest to
        ``reference_kernel * reference_bin_width / bin_width``.
    """
    target = reference_kernel * reference_bin_width / bin_width
    k = max(min_kernel, round(target))
    if k % 2 == 0:
        k += 1
    return k
