Architectural scaling
=====================

MaldiDeepKit's default hyperparameters are calibrated for the MaldiAMRKit
default layout: ``bin_width = 3`` over the 2000-20000 Da range, giving
``input_dim = 6000``. Two things can change that layout:

- the user picks a different **bin width** (wider or finer bins), and
- the user **trims** the spectrum to a narrower m/z range, shrinking
  ``input_dim`` independently of ``bin_width``.

These two axes affect different architectural concerns, so MaldiDeepKit
scales each knob with whichever quantity it actually depends on. This
page documents that semantic split and the :meth:`from_spectrum`
factory that applies it.

Why the defaults look the way they do
-------------------------------------

Default conv-kernel width (CNN first layer, ResNet stem) and patch size
(Transformer) balance two forces:

- **Local feature extraction.** A small kernel or patch lets early
  layers capture short-range correlations between adjacent bins without
  collapsing distinct local patterns into a single value.
- **Global feature integration.** Keeping the kernel / patch small
  leaves enough depth (for conv stacks) or token-sequence length (for
  attention and SSM) for deeper components to integrate long-range
  structure.

The specific reference values (``kernel_size=7``, ``patch_size=4``) are
inherited from well-tested image-domain defaults (ResNet-18 stem,
ViT patch).

Empirically, these values also align cleanly with the **peak width
distribution** on DRIAMS spectra at ``bin_width=3``. We measured the
full width at half maximum (FWHM) of all detected peaks on 50 random
spectra per species-drug pair (``scipy.signal.find_peaks`` with
prominence threshold, ``scipy.signal.peak_widths`` at ``rel_height=0.5``):

.. list-table:: Peak FWHM on DRIAMS (Da, ``bin_width=3``)
   :header-rows: 1
   :widths: 30 12 10 10 10 10 10

   * - Pair
     - Spectra
     - P10
     - P25
     - Median
     - P75
     - P90
   * - *S. aureus* / oxacillin
     - 4874
     - 6.6
     - 9.0
     - **11.8**
     - 15.1
     - **21.3**
   * - *E. coli* / ceftriaxone
     - 8080
     - 6.3
     - 8.7
     - **11.8**
     - 15.4
     - **21.5**
   * - *K. pneumoniae* / ceftriaxone
     - 5512
     - 6.3
     - 8.6
     - **11.8**
     - 15.2
     - **21.1**

The distribution is strikingly consistent across the three species: the
median peak is **~12 Da** wide and the 90th-percentile peak is **~21 Da**.
This is *why* the image-domain defaults happen to be a natural fit:

- ``patch_size=4`` (12 Da at ``bin_width=3``) ≈ one median peak per
  token, so each Transformer token captures roughly one peak's worth
  of shape.
- ``kernel_size=7`` (21 Da) ≈ 90th-percentile peak FWHM, so the first
  CNN / ResNet conv fully contains ~90% of peaks along with one or two
  bins of shoulder on either side.

This is **empirical observation, not a priori design**. The defaults
were picked for the local-vs-global balance first; the alignment with
DRIAMS peak widths emerged from the measurement above. Different
instruments, different preprocessing, or different bin widths will
shift the distribution - use :meth:`from_spectrum` to scale.

The semantic split
------------------

When the spectrum layout changes, each architectural knob is driven
by the quantity it physically depends on:

.. list-table::
   :header-rows: 1
   :widths: 28 26 46

   * - Knob
     - Driven by
     - Why
   * - CNN ``kernel_size``
     - ``bin_width``
     - Kernels aggregate per-bin information density. A finer bin grid
       carries less information per bin, so a wider kernel in bins is
       needed to gather the same local structure; a coarser grid calls
       for a smaller kernel.
   * - ResNet ``stem_kernel_size``
     - ``bin_width``
     - Same argument as for the CNN: the first layer's job is to
       aggregate adjacent bin values, which is a per-bin-density
       concern.
   * - Transformer ``patch_size``
     - (none: scale-agnostic)
     - The Transformer uses a learned positional embedding sized to
       whatever token count the patch embedding produces, so any
       ``input_dim`` works with the default ``patch_size=4`` without
       tuning. ``from_spectrum`` just forwards ``input_dim``.

The MLP has no spectral-layout-dependent knob either: its first layer
is a single ``Linear(input_dim, hidden_dim)`` that handles any input
size identically.

Auto-scaling: ``from_spectrum``
-------------------------------

Each classifier exposes a ``from_spectrum`` classmethod that applies
the semantic split automatically:

.. code-block:: python

   Classifier.from_spectrum(bin_width: int, input_dim: int, **overrides)

Both parameters describe the spectrum layout. Each classifier uses the
relevant one internally:

- ``MaldiCNNClassifier`` and ``MaldiResNetClassifier`` scale their conv
  kernel using ``bin_width``.
- ``MaldiTransformerClassifier`` and ``MaldiMLPClassifier`` are
  architecturally scale-agnostic and just record ``input_dim`` for
  shape validation; ``from_spectrum`` forwards overrides untouched.

Any keyword in ``**overrides`` wins over the auto-scaled default.

.. code-block:: python

   from maldideepkit import (
       MaldiCNNClassifier,
       MaldiTransformerClassifier,
   )

   # Reference layout: kernel_size=7, patch_size=4
   cnn = MaldiCNNClassifier.from_spectrum(bin_width=3, input_dim=6000)
   tr = MaldiTransformerClassifier.from_spectrum(bin_width=3, input_dim=6000)

   # Trim the spectrum (bin_width unchanged, input_dim halved).
   # CNN kernel stays at 7 (bin density unchanged).
   # Transformer unchanged (scale-agnostic).
   cnn_trim = MaldiCNNClassifier.from_spectrum(bin_width=3, input_dim=3000)
   tr_trim = MaldiTransformerClassifier.from_spectrum(bin_width=3, input_dim=3000)

   # Coarser binning over the full range.
   # CNN kernel drops to 5 (fewer bins carry more info per bin).
   cnn_coarse = MaldiCNNClassifier.from_spectrum(bin_width=6, input_dim=3000)

   # Fine bins, trimmed.
   # CNN kernel up to 21 (fine bins -> larger kernel to aggregate).
   cnn_fine = MaldiCNNClassifier.from_spectrum(bin_width=1, input_dim=4000)

   # Override the auto-choice explicitly.
   cnn_custom = MaldiCNNClassifier.from_spectrum(
       bin_width=3, input_dim=6000, kernel_size=(11, 7, 5, 3),
   )

Scaling helpers
---------------

The underlying helper is public and lives under
``maldideepkit._bin_scaling``:

- :func:`maldideepkit._bin_scaling.scale_odd_kernel` returns an odd
  kernel size closest to ``reference_kernel * reference_bin_width /
  bin_width``, clamped ``>= 3``. Reference: kernel 7 at
  ``bin_width=3``. Used by CNN and ResNet kernels.

The Transformer's ``patch_size`` is deliberately *not* scaled by
layout: a learned positional embedding sizes itself to whatever token
count the patch embedding produces, so the same ``patch_size=4`` works
across bin widths and trimmed m/z ranges.

Per-block configurability
-------------------------

For reviewer-driven ablations, :class:`~maldideepkit.MaldiCNNClassifier`
accepts a scalar *or* a per-block tuple for both ``kernel_size`` and
``pool_size``:

.. code-block:: python

   # Scalar: broadcast to every block
   MaldiCNNClassifier(channels=(32, 64, 128, 128), kernel_size=7)

   # Tuple: per-block progression
   MaldiCNNClassifier(
       channels=(32, 64, 128, 128),
       kernel_size=(11, 7, 5, 3),
       pool_size=(2, 2, 2, 4),
   )

:class:`~maldideepkit.MaldiResNetClassifier` exposes
``stem_kernel_size``, ``stem_stride``, ``block_kernel_size``, and
``use_stem_pool`` as explicit keyword arguments. Defaults are
peak-friendly for MALDI-TOF (``stem_stride=1``, ``block_kernel_size=7``,
``use_stem_pool=False``) and deviate from the literal ResNet-18
backbone. To reproduce the literal configuration, set
``stem_stride=2``, ``block_kernel_size=3``, ``use_stem_pool=True``.

Input-dim sensitivity of the flat dense head
--------------------------------------------

Only :class:`~maldideepkit.MaldiCNNClassifier` has a flat dense head
whose input width scales linearly with ``input_dim``. Given four
``pool_size=2`` blocks and ``channels[-1]=128``, the flat head width is
``128 * input_dim / 16``. For ``input_dim=6000`` this is already
48,000 units; at ``input_dim=18000`` it grows to 144,000. If this
matters, prefer :class:`~maldideepkit.MaldiResNetClassifier` or
:class:`~maldideepkit.MaldiTransformerClassifier` - both use a pooled
head whose width is independent of ``input_dim``.
