Dual-branch module
==================

Dual-input classifier that conditions on both the spectrum and a
per-row **side feature** (e.g. an antibiotic fingerprint, a culture
medium descriptor, an instrument ID). The side feature is packed into
``X`` by column concatenation, so the class behaves like any other
:class:`~maldideepkit.BaseSpectralClassifier` for scikit-learn (it
works in ``Pipeline``, ``GridSearchCV``, and ``cross_val_score``).

Preprocessing and the split point
---------------------------------

Every parameter that assumes its input is a spectrum is confined to the
first ``n_spec_bins`` columns; the side block is passed through
untouched.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Parameter
     - Applies to
   * - ``warping``, ``standardize``, ``input_transform``
     - Spectrum columns only. Fitted on the training split of the
       spectrum block, so ``feature_mean_`` / ``feature_std_`` have
       length ``n_spec_bins`` and ``input_transform="log1p"`` never
       rewrites binary fingerprint bits.
   * - ``augment``
     - Spectrum columns only.
   * - ``cutmix_alpha``
     - Window drawn inside the spectrum block, so it cannot straddle
       the boundary. The side block is convex-combined with the same
       ``effective_lam`` and permutation, keeping ``(x, y)``
       consistent.
   * - ``mixup_alpha``
     - Both blocks, with one shared ``lam``. This is deliberate:
       blending the spectra while leaving the side features unblended
       would contradict the blended label.

Anything you want applied to the side block (scaling a continuous
descriptor, for instance) belongs in a ``Pipeline`` step upstream of
the estimator.

MaldiDualBranchClassifier
-------------------------

.. autoclass:: maldideepkit.MaldiDualBranchClassifier
   :members:
   :undoc-members:
   :show-inheritance:

smiles_to_fingerprint
---------------------

Helper that converts SMILES strings to ECFP4 (Morgan radius=2,
2048-bit) fingerprints, suitable as side features for the dual-branch
classifier. Requires the optional ``rdkit`` extra (see
:doc:`../installation`).

.. autofunction:: maldideepkit.smiles_to_fingerprint

Examples
--------

Pack a 64-bin spectrum next to a 16-dim side vector and fit:

.. code-block:: python

   import numpy as np
   from maldideepkit import MaldiDualBranchClassifier

   rng = np.random.default_rng(0)
   X_spec = rng.standard_normal((40, 64)).astype("float32")
   X_side = rng.standard_normal((40, 16)).astype("float32")
   X = np.hstack([X_spec, X_side])
   y = rng.integers(0, 2, size=40)

   clf = MaldiDualBranchClassifier(
       n_spec_bins=64,
       spec_hidden_dims=(32,),
       fp_hidden_dims=(16,),
       random_state=0,
   ).fit(X, y)

Plug a backbone (an instance, or a name such as ``"cnn"``) in as the
spectral branch (its final ``nn.Linear`` is replaced with ``Identity``,
so the module exposes features rather than logits):

.. code-block:: python

   from maldideepkit import MaldiCNNClassifier, MaldiDualBranchClassifier

   clf = MaldiDualBranchClassifier(
       n_spec_bins=64,
       backbone=MaldiCNNClassifier(channels=(8, 16), head_dim=32),
       fp_hidden_dims=(16,),
   )

Start the spectral branch from a pre-trained encoder (without
``backbone=``, it takes the architecture of the encoder), and optionally
freeze it for the first epochs while the side branch and the fusion head
train:

.. code-block:: python

   clf = MaldiDualBranchClassifier(
       n_spec_bins=64,
       init_from=encoder,
       freeze_trunk_epochs=10,
   )

Convert SMILES into ECFP4 fingerprints for the side input (requires
the ``rdkit`` extra):

.. code-block:: python

   from maldideepkit import smiles_to_fingerprint

   fps = smiles_to_fingerprint(
       ["CCO", "c1ccccc1"], fp_dim=2048, radius=2,
   )
   # fps.shape == (2, 2048), dtype float32
