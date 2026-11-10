Pre-training module
===================

Self-supervised pre-training of the MaldiDeepKit backbones on unlabelled
(or partially labelled) binned spectra.
:class:`~maldideepkit.pretraining.MaldiSpectrumEncoder` wraps the
module of a classifier, trains it with a pre-text task on a spectra
pool that needs no labels, and exposes the learned embeddings. A
classifier then starts its trunk from the encoder through
``init_from=`` and is fine-tuned on the labelled data, optionally with
a linear-probing phase (``freeze_trunk_epochs=``).

- :class:`~maldideepkit.pretraining.MaldiSpectrumEncoder` - scikit-learn
  transformer (``fit`` / ``transform`` / ``save`` / ``load``) that
  pre-trains a backbone and returns ``(n_samples, feature_dim_)``
  embeddings. Also importable as ``maldideepkit.MaldiSpectrumEncoder``.
- :class:`~maldideepkit.pretraining.Pretext` - base class of the
  pre-text tasks, with the built-in
  :class:`~maldideepkit.pretraining.MAEPretext`,
  :class:`~maldideepkit.pretraining.ShufflePretext`,
  :class:`~maldideepkit.pretraining.ShuffleMAEPretext`, and
  :class:`~maldideepkit.pretraining.ContrastivePretext`;
  :func:`~maldideepkit.pretraining.register_pretext` and
  :func:`~maldideepkit.pretraining.get_pretext_class` map names to
  classes.
- :func:`~maldideepkit.pretraining.make_pretraining_loaders` - lazy
  train / validation loaders over a (possibly memory-mapped) spectra
  pool, held on the host or on a GPU, with a ``collate_fn`` seam for
  custom batching.

Trunk and embeddings
--------------------

Every built-in backbone module exposes ``forward_features(x)`` (the
``(batch, feature_dim)`` embedding), an integer ``feature_dim``, and a
``final_layer`` property (the ``nn.Linear`` mapping the embedding to the
logits, so ``forward(x) == final_layer(forward_features(x))``). The
**trunk** is every parameter and buffer of the module except
``final_layer``: it is what ``init_from`` copies and what
``freeze_trunk_epochs`` freezes. Module attributes and ``state_dict``
keys are unchanged from v0.3.0, so existing checkpoints load as before.
A fitted classifier returns the embedding with
:meth:`~maldideepkit.BaseSpectralClassifier.get_embeddings`, as
:meth:`~maldideepkit.pretraining.MaldiSpectrumEncoder.transform` does
for an encoder (for the dual-branch classifier, the fused features of
the two branches).

.. list-table::
   :header-rows: 1
   :widths: 30 25 45

   * - Backbone
     - ``feature_dim`` (defaults)
     - ``final_layer``
   * - ``SpectralAttentionMLP``
     - ``head_dims[-1]`` = 128 (``hidden_dim`` when ``head_dims=()``)
     - Last layer of ``head``
   * - ``SpectralCNN1D``
     - ``head_dim`` = 128
     - Last layer of ``head``
   * - ``SpectralResNet1D``
     - ``stage_channels[-1]`` = 512
     - ``fc``
   * - ``SpectralTransformer1D``
     - ``head_dim`` = 128
     - Last layer of ``head``

**Custom backbones.** The backbone module of a custom classifier (a
:class:`~maldideepkit.BaseSpectralClassifier` subclass whose
``_build_model()`` returns, say, a plain ``nn.Sequential``) does not
need ``forward_features`` / ``feature_dim`` / ``final_layer``. It
follows the convention of the dual-branch spectral branch: the final
layer is the ``nn.Linear`` that **produces the output**, found with a
forward pass on zeros whatever the order in which the layers are
registered (the last registered ``nn.Linear``, in ``named_modules()``
order, when that pass fails), the trunk is everything else,
``feature_dim`` is that layer's
``in_features``, and the embedding is the module's output with that
layer replaced by ``nn.Identity`` (on a parameter-sharing copy, so the
module and its ``state_dict`` are unchanged). The encoder, ``init_from``
(trunk transfer and ``freeze_trunk_epochs``), and
:class:`~maldideepkit.MaldiDualBranchClassifier` all use this rule, so
a custom template can be pre-trained and fine-tuned like a built-in
one. The module's forward must end with that layer (anything applied
after it is kept in the embedding) and reach it through its registered
attribute. :meth:`~maldideepkit.pretraining.MaldiSpectrumEncoder.save`
records the template class by import path (see
:ref:`the loading rule <checkpoint-trust>`).

A custom classifier can take the base parameters through ``**kwargs``
(``def __init__(self, hidden=16, **kwargs): super().__init__(**kwargs)``):
they are kept by ``clone``, ``GridSearchCV``, and ``save`` / ``load``, as
long as ``kwargs`` is forwarded unchanged.
:func:`~maldideepkit.pretraining.register_backbone` gives the class a name
accepted wherever the package takes an architecture
(``MaldiSpectrumEncoder(backbone=)``, ``MaldiDualBranchClassifier(backbone=)``,
``MaldiSelfTrainingClassifier(base_estimator=)``, and ``SpectralEnsemble``); a
name uses the class defaults, and must be registered again before ``load``.
:func:`~maldideepkit.pretraining.get_backbone_class` returns the class
registered under a name.

.. code-block:: python

   from torch import nn
   from maldideepkit import BaseSpectralClassifier, MaldiSpectrumEncoder

   class SequentialClassifier(BaseSpectralClassifier):
       def __init__(self, hidden_dim=256, *, input_dim=None, n_classes=2,
                    random_state=0, init_from=None, freeze_trunk_epochs=0):
           super().__init__(
               input_dim=input_dim, n_classes=n_classes,
               random_state=random_state, init_from=init_from,
               freeze_trunk_epochs=freeze_trunk_epochs,
           )
           self.hidden_dim = hidden_dim

       def _build_model(self):
           return nn.Sequential(
               nn.Linear(self.input_dim_, self.hidden_dim),
               nn.BatchNorm1d(self.hidden_dim),
               nn.ReLU(),
               nn.Linear(self.hidden_dim, self.n_classes_),  # final layer
           )

   encoder = MaldiSpectrumEncoder(backbone=SequentialClassifier()).fit(X_pool)
   clf = SequentialClassifier(init_from=encoder).fit(X, y)

Pre-text tasks
--------------

The binned backbones emit one embedding per spectrum, so every pre-text
task is a small head on ``forward_features``. The heads are dropped at
fine-tuning time: only the backbone transfers. Corruptions act on the
input-transformed batch, except where raw intensities are needed. They
are drawn on the batch's device (see `GPU throughput`_), driven by the
encoder's seeded corruption generator.

.. list-table::
   :header-rows: 1
   :widths: 18 50 32

   * - ``pretext``
     - Corruption
     - Head and loss
   * - ``"mae"`` (default, ``MAEPretext``)
     - ``mask_ratio`` of the bins, drawn uniformly, are set to ``0``.
     - Per-bin decoder ``Linear(feature_dim -> input_dim)``, MSE on the
       masked bins. With an input transform that leaves the bins on the
       raw intensity scale (``"log1p"``, the ResNet default, or
       ``"none"``) the target is the model input standardised per bin
       with its training-split mean and standard deviation
       (``recon_mean_``, ``recon_std_``): on total-ion-current
       normalised spectra the raw bins are of order ``1e-4``.
   * - ``"shuffle"`` (``ShufflePretext``)
     - ``mask_ratio`` of the bins are sampled without replacement with
       probability proportional to the raw (pre-transform) intensity;
       ``shuffle_fraction`` of them take the value of the same bin in
       another spectrum of the batch.
     - Per-bin discriminator ``Linear(feature_dim -> input_dim)``, BCE
       on the sampled bins. A swap that leaves the value unchanged
       (e.g. a zero for a zero) is labelled "real".
   * - ``"shuffle+mae"`` (``ShuffleMAEPretext``)
     - Both corruptions, drawn independently.
     - Both heads, losses summed.
   * - ``"contrastive"`` (``ContrastivePretext``)
     - Two views per spectrum: ``augment`` on the raw spectra (by
       default a multiplicative intensity jitter ``1 + U(-0.2, 0.2)``
       per spectrum and a global m/z shift of up to 1 bin), then the
       input transform and ``"mae"``-style masking.
     - Projection MLP (``feature_dim -> feature_dim -> projection_dim``)
       and the NT-Xent loss (``temperature``) with the other spectra of
       the batch as negatives.

NT-Xent draws its negatives from the batch, so a contrastive pre-text
with ``batch_size < 64`` emits a ``UserWarning``.

**Contrastive augmentations.** ``augment=`` (an encoder parameter, and
a :class:`~maldideepkit.pretraining.ContrastivePretext` parameter)
takes a :class:`~maldideepkit.augment.SpectrumAugment`, as the
classifiers' ``augment`` does; ``None`` keeps the default above. Every
batch draws its views with a copy of it re-seeded from the encoder's
corruption generator (the instance is never modified and its own
``random_state`` is ignored), so the validation loss uses the same
views every epoch and the result depends on ``random_state`` only.
``augment`` set with a non-contrastive pre-text emits a
``UserWarning``.

.. code-block:: python

   from maldideepkit.augment import SpectrumAugment

   encoder = MaldiSpectrumEncoder(
       backbone=template, pretext="contrastive",
       augment=SpectrumAugment(intensity_jitter=0.3, mz_shift_max_bins=2,
                               peak_dropout_rate=0.05),
   )

Pluggable pre-text tasks
------------------------

``pretext=`` takes a name or a :class:`~maldideepkit.pretraining.Pretext`
instance:

- **A name** (``"mae"``, ``"shuffle"``, ``"shuffle+mae"``,
  ``"contrastive"``, or one added with
  :func:`~maldideepkit.pretraining.register_pretext`) is resolved with
  :func:`~maldideepkit.pretraining.get_pretext_class` and built from the
  encoder-level ``mask_ratio`` / ``shuffle_fraction`` /
  ``projection_dim`` / ``temperature`` / ``augment`` that its class
  accepts.
- **An instance** holds its own parameters. It follows the
  scikit-learn conventions (``get_params`` / ``set_params`` /
  :func:`~sklearn.base.clone`), so its parameters are searchable as
  ``pretext__<param>`` (e.g.
  ``GridSearchCV(pipe, {"enc__pretext__mask_ratio": [0.1, 0.3]})``).
  The encoder trains with a clone (exposed as ``pretext_``), and the
  encoder-level pre-text parameters are ignored, with a ``UserWarning``
  when one differs from its default.

A custom task subclasses :class:`~maldideepkit.pretraining.Pretext`,
stores its ``__init__`` arguments unchanged under their own names, and
implements:

- ``build_head(feature_dim, input_dim)``: a freshly initialised head
  (an ``nn.Module``, or an ``nn.ModuleDict`` for several heads),
  registered as ``model.pretext`` (state-dict keys ``pretext.*``) and
  discarded at fine-tuning time.
- ``loss(model, batch, generator)``: the scalar loss of one batch.
  ``model`` is the :class:`~maldideepkit.pretraining.SSLModel`
  (``model.forward_features(x)`` embeds a model input,
  ``model.pretext`` is the head); ``batch`` is an
  :class:`~maldideepkit.pretraining.SSLBatch` (``x``: model input,
  ``x_raw``: raw spectra, ``transform``: the fitted input transform
  for raw spectra, on their device, ``target`` / ``labeled``:
  auxiliary targets, ``recon_target()``: the reconstruction target of
  the masked-bin tasks). Every random draw must be driven by
  ``generator`` (a seeded CPU generator re-seeded before each
  validation pass): drawn from it directly, or from a generator on the
  batch's device seeded with one draw of it, as the built-in tasks do
  on a GPU batch.
- ``check_params(encoder=None)`` (optional): validates the parameters
  at ``fit`` (raise, or warn against the encoder's own parameters such
  as ``batch_size``).
- ``uses_target = True`` (optional class attribute): the ``y`` given to
  ``fit`` is required and reaches ``batch.target`` / ``batch.labeled``
  without an auxiliary head, encoded as ``aux_target`` says. With the
  default ``"classification"`` it holds class indices in the order of
  ``aux_classes_``, ``-1`` on unlabelled rows, so ``y`` can be, e.g., the
  acquisition site. With ``aux_lambda > 0`` the auxiliary head reads the
  same ``y``.
- ``on_train_start(model, encoder)``, ``on_epoch_start(model, epoch)``,
  ``on_step_end(model, step)``, ``on_epoch_end(model, epoch, logs)``, and
  ``on_train_end(model)`` (optional): the hooks of
  :class:`~maldideepkit.utils.Callback`. For example, a momentum teacher
  can be copied from ``model`` in ``on_train_start`` and updated in
  ``on_step_end``. State a hook keeps on the instance reaches neither the
  checkpoint nor early stopping. No hook runs with ``epochs=0``.

.. code-block:: python

   import torch
   import torch.nn.functional as F
   from torch import nn
   from maldideepkit.pretraining import (
       MaldiSpectrumEncoder, Pretext, register_pretext,
   )

   class DenoisePretext(Pretext):
       """Reconstruct the model input from a noisy copy."""

       def __init__(self, noise_std=0.1):
           self.noise_std = noise_std

       def check_params(self, encoder=None):
           if float(self.noise_std) <= 0:
               raise ValueError("noise_std must be > 0.")

       def build_head(self, feature_dim, input_dim):
           return nn.Linear(feature_dim, input_dim)

       def loss(self, model, batch, generator):
           noise = torch.randn(batch.x.shape, generator=generator)
           noisy = batch.x + self.noise_std * noise.to(batch.x.device)
           recon = model.pretext(model.forward_features(noisy))
           return F.mse_loss(recon.float(), batch.x.float())

   encoder = MaldiSpectrumEncoder(backbone=template,
                                  pretext=DenoisePretext(noise_std=0.2))

   register_pretext("denoise", DenoisePretext)  # optional: by name
   encoder = MaldiSpectrumEncoder(backbone=template, pretext="denoise")

``save`` stores a pre-text instance as its class import path plus its
parameters (a :class:`~maldideepkit.augment.SpectrumAugment` parameter
likewise), and a registered name as the name, so register it again
before ``load``. What ``load`` then imports is described in
:ref:`the loading rule <checkpoint-trust>`.

Optimiser, scheduler, and callbacks
-----------------------------------

The encoder takes the ``optimizer``, ``scheduler``, and ``callbacks``
parameters of the classifiers
(:class:`~maldideepkit.BaseSpectralClassifier`), with the same rules.
Callback hooks run after those of the pre-text task.

Semi-supervised auxiliary head
------------------------------

With ``aux_lambda > 0`` the encoder adds an auxiliary head on the
embedding and minimises ``L_pretext + aux_lambda * L_aux``, using the
labels passed as ``y`` to :meth:`~maldideepkit.pretraining.MaldiSpectrumEncoder.fit`:

- ``aux_target="classification"``: ``Linear(feature_dim -> n_classes)``
  and cross-entropy. Labels of any dtype (e.g. species names) are
  encoded in ``aux_classes_``.
- ``aux_target="regression"``: ``Linear(feature_dim -> 1)`` and MSE on
  ``float32`` targets (e.g. log2-MIC values).

Rows that are missing (``np.nan`` / ``None``) or equal to
``unlabeled_marker`` (default ``-1``) are unlabelled and excluded from
``L_aux``; a batch without labelled rows contributes ``0``. Since ``-1``
can be a valid regression target, a numeric marker with
``aux_target="regression"`` emits a ``UserWarning``; use
``unlabeled_marker=np.nan`` there. The auxiliary head reads the
embedding of the clean (uncorrupted) spectra, not of the corrupted
pre-text input. With ``aux_lambda=0`` (default) ``y`` is ignored, so
the encoder fits inside a ``Pipeline``.

Preprocessing
-------------

The encoder has no ``standardize`` / ``input_transform`` / ``warping``
parameters of its own: it takes them **from the backbone template**. A
string backbone uses the class defaults (``"resnet"`` implies
``input_transform="log1p"``, the other three no transform); a
classifier instance uses its own settings. Pass the template of the
classifier you will fine-tune, and the encoder's preprocessing matches
it by construction.

:meth:`~maldideepkit.pretraining.MaldiSpectrumEncoder.fit` holds out a
random ``val_fraction`` of the pool for the early-stopping loss. The
warper (if any) is fitted on the training rows and the whole pool is
warped once; the input transform is fitted on the (warped) training rows
and applied to every collated batch inside the training loop, so a
custom ``collate_fn`` cannot bypass it. The fitted state is exposed as
``warper_``, ``input_transform_state_``, ``feature_mean_``, and
``feature_std_``; :meth:`~maldideepkit.pretraining.MaldiSpectrumEncoder.transform`
applies it through the same code path as the classifiers' ``predict``.
``epochs=0`` fits this state and builds a randomly initialised backbone
without any self-supervised training (a control with the same
preprocessing).

**Small pools.** ``early_stopping_patience`` and ``warmup_epochs``
count epochs, and an epoch is ``n_train / batch_size`` optimiser steps,
so on a pool of a few hundred spectra the default ``batch_size=256``
leaves only a handful of steps per epoch. On the 741 spectra of the
tutorial dataset, the default MLP encoder stops after 11 epochs with
its best validation loss at epoch 0 and an almost constant embedding,
while ``batch_size=32`` keeps improving for 100 to 200 epochs,
depending on the number of threads and the device. On a small pool,
lower ``batch_size`` and check ``history_["val_loss"]`` (a best epoch
of 0 means nothing was learned); the contrastive pre-text still needs a
batch of at least 64 (see above).

Fine-tuning with ``init_from``
------------------------------

Every single-input classifier accepts ``init_from=`` (the base path of
an encoder checkpoint, or a fitted encoder) and
``freeze_trunk_epochs=``. At :meth:`~maldideepkit.BaseSpectralClassifier.fit`:

- **Weights.** Every trunk tensor is copied from the encoder; the
  ``final_layer`` keeps its fresh initialisation. A trunk tensor missing
  on either side, or with a different shape, raises :class:`ValueError`
  naming both architectures (no partial load). Every rebuilt model
  (including ``retry_on_val_auroc_below`` retries) starts from the encoder;
  ``fit(warm_start=True)`` with an existing ``model_`` keeps its own
  weights.
- **Preprocessing.** The classifier inherits the encoder's fitted
  warper and input-transform state instead of fitting its own on the
  (possibly small) labelled training split. Its ``standardize`` /
  ``input_transform`` / ``warping`` parameters become **assertions**:
  the resolved transform mode (``standardize=True`` and
  ``input_transform="standardize"`` are equivalent), the warping
  configuration, and the number of input bins must match the encoder's,
  else :class:`ValueError`.
- **Linear probing then fine-tuning (LP-FT).** For the first
  ``freeze_trunk_epochs`` epochs only ``final_layer`` is trained (trunk
  parameters frozen, trunk modules in ``eval()`` mode, so BatchNorm
  statistics and dropout are frozen too); the trunk is then unfrozen
  and the early-stopping patience restarts. A value ``>= epochs`` gives
  a pure linear probe. After :meth:`~maldideepkit.BaseSpectralClassifier.fit`,
  ``trunk_unfroze_`` reports whether the unfrozen phase was reached and
  ``best_epoch_phase_`` (``"frozen"`` or ``"unfrozen"``) the phase of the
  deployed weights. When the trunk never unfroze it is restored exactly
  to the encoder's weights (undoing EMA / SWA drift).

**Clone semantics.** A checkpoint path is the clone-safe form: every
:func:`sklearn.base.clone` (``GridSearchCV``, ``cross_val_score``,
:class:`~maldideepkit.MaldiSelfTrainingClassifier`) loads it at ``fit``.
A fitted encoder instance is kept **by reference** across clones, so it
is shared and must be treated as read-only:
``set_params(init_from__<name>=...)`` with a changed value raises
:class:`ValueError`. A search **over** ``init_from`` itself must list
checkpoint paths (``GridSearchCV(clf, {"init_from": ["enc_a",
"enc_b"]})``): the search clones every candidate value, so a list of
encoder instances reaches ``fit`` as unfitted copies and raises
:class:`ValueError`. ``save()`` records the encoder checkpoint in
``meta["init_from_provenance"]`` (absolute path and sha256 of its
``.pt``, ``None`` for an in-memory encoder); ``load()`` restores the
fine-tuned weights and never re-applies ``init_from``.

**Dual-branch classifier.** :class:`~maldideepkit.MaldiDualBranchClassifier`
takes ``init_from`` on itself or on its ``backbone=`` template (setting
both raises :class:`ValueError`) and loads the trunk into the spectral
branch (without ``backbone=``, the branch takes the architecture of the
encoder); ``freeze_trunk_epochs`` freezes that branch while the side branch
and the fusion head train. The outer dual classifier owns the
preprocessing of the spectrum columns, so **its** ``standardize`` /
``input_transform`` / ``warping`` and ``n_spec_bins`` are checked
against the encoder (with a ResNet template, set
``input_transform="log1p"`` on the dual classifier).

GPU throughput
--------------

Two encoder parameters speed up pre-training on a GPU without changing
what is learned (the checkpoint ``.json`` stores them, but neither is
part of the fitted module or of its state dict):

- ``data_device="auto"`` (default) keeps the pool **on the training
  GPU** when it takes at most half of the GPU memory free at the start
  of :meth:`~maldideepkit.pretraining.MaldiSpectrumEncoder.fit` (the
  pool is ``n_samples * n_bins`` ``float32`` values; the targets and
  the transform statistics are counted too), and on the host otherwise.
  ``"cuda"`` forces the GPU pool, ``"cpu"`` the host pool; the choice
  is recorded in ``data_device_``.

  - **GPU pool.** The (warped) pool is copied to the GPU once, in row
    chunks (a memory-mapped pool is never loaded whole into host
    memory); every batch is gathered there and input-transformed by the
    PyTorch twin of the transform
    (:func:`~maldideepkit.base.data.apply_input_transform_torch`, equal
    to the NumPy values up to ``float32`` rounding), so only the batch
    row indices cross the host-device bus.
  - **Host pool.** Every batch is read from the pool with one indexing
    call (a memory map stays on disk), transformed in NumPy, and copied
    to the device from pinned memory.

  Both pools use the same ``DataLoader`` sampler, so the batch
  composition of every epoch is the same for the same ``random_state``.
  Warping, if any, is applied once to the whole pool on the host before
  the upload. With ``epochs=0`` nothing is uploaded.
  :meth:`~maldideepkit.pretraining.MaldiSpectrumEncoder.transform`
  likewise streams raw batches to the GPU and transforms them there
  (unless ``data_device="cpu"``).
- **Device-side corruption.** On a GPU batch the corruptions (masks,
  intensity-weighted sampling, donor permutations, contrastive
  augmentations) are drawn on the GPU, from a GPU generator seeded with
  one draw of the CPU corruption generator; the masked losses are
  computed without host synchronisation. The draws are still a
  deterministic function of ``random_state`` and the validation
  corruption is the same every epoch, but they differ from the draws of
  a CPU run.
- ``torch_compile=True`` runs the backbone's feature extractor through
  ``torch.compile`` during ``fit`` (static shapes: one compilation per
  batch shape and train / eval mode, done on the first batches, so it
  pays off on long runs). ``model_``, the ``save`` checkpoint (no
  ``_orig_mod.`` key prefix), the ``init_from`` transfer, and
  ``transform`` use the plain module, so pickling and
  :func:`~sklearn.base.clone` never see a compiled object. The
  classifiers take the same parameter (``torch.compile`` of the training
  module).

.. code-block:: python

   encoder = MaldiSpectrumEncoder(
       backbone=template, pretext="mae", use_amp=True,
       data_device="auto", torch_compile=True,
   ).fit(X_pool)
   print(encoder.data_device_)          # cuda:0, or cpu for a too large pool

**Mixed precision.** ``amp_dtype`` sets the autocast dtype of
``use_amp=True``. The encoder default ``"auto"`` uses ``bfloat16`` on
GPUs that support it (else ``float16``): it has the exponent range of
``float32``, so no activation overflows and no ``GradScaler`` is needed.
In ``float16`` a scale-free layer whose weights keep growing during
pre-training (a wide ``Linear`` before a ``LayerNorm``, as in the CNN
head under the contrastive loss) can overflow on some spectra and turn
the loss into ``NaN``. The classifiers take the same parameter (default
``"float16"``, as in v0.3.0); fine-tune a pre-trained trunk with
``amp_dtype="bfloat16"`` too.

Persistence
-----------

:meth:`~maldideepkit.pretraining.MaldiSpectrumEncoder.save` writes
``<path>.pt`` (the full state dict: ``backbone.*`` with its final layer,
``pretext.*``, and ``aux.*``), ``<path>.json`` (``meta["kind"] =
"encoder"``, the encoder parameters, the backbone template, and the
fitted preprocessing), and ``<path>.warper.pkl`` when a warper was
fitted. :meth:`~maldideepkit.pretraining.MaldiSpectrumEncoder.load`
rebuilds a fitted encoder from those files; the same base path is what
``init_from`` expects. Only a ``.pt`` / ``.pth`` / ``.json`` /
``.warper.pkl`` suffix is stripped from that base path, so dotted names
such as ``"enc_mask0.05"`` and ``"enc_mask0.10"`` stay distinct (as for
the classifiers' ``save`` / ``load``). The reloaded encoder's backbone
template has ``warping=None`` (the fitted warper is restored into
``warper_``), so refitting a loaded encoder does not re-apply warping.

.. _checkpoint-trust:

``save`` (of an encoder or a classifier) stores a class, a function, or
a :func:`functools.partial` by import path; any other object is rebuilt
at ``load`` from its class and the attributes named after its
constructor arguments. ``save`` checks that every stored class and
function re-imports from its path, so a lambda, or a function or class
defined inside a function, raises :class:`ValueError`, and it writes
nothing when anything fails. A class defined in a script or notebook is
stored under ``__main__``, so define it again before loading (or keep
it in an importable module). A pre-text saved by name comes from the
:func:`~maldideepkit.pretraining.register_pretext` registry.

With the default ``trusted=False``, ``load`` imports no new module:
every class named by the checkpoint must come from MaldiDeepKit or from
a module you have already imported. On top of that, an object given as
``optimizer``, ``scheduler``, ``loss`` or ``callbacks`` must be a torch
module, optimiser or scheduler, or a ``Callback``, and a function must
be defined in MaldiDeepKit or in ``torch.nn.functional`` (a few
functions exposed there, such as ``F.gelu``, are defined elsewhere).
Anything else needs ``load(path, trusted=True)``. ``init_from=<path>``
always uses the default: for a trusted checkpoint, pass
``MaldiSpectrumEncoder.load(path, trusted=True)``. A ``.warper.pkl`` is
a pickle, read whatever ``trusted`` is: load checkpoints with a warper
only from trusted sources.

MaldiSpectrumEncoder
--------------------

.. autoclass:: maldideepkit.pretraining.MaldiSpectrumEncoder
   :members:
   :undoc-members:
   :show-inheritance:

Pre-text tasks API
------------------

.. autoclass:: maldideepkit.pretraining.Pretext
   :members:
   :show-inheritance:

.. autoclass:: maldideepkit.pretraining.MAEPretext
   :members:
   :show-inheritance:

.. autoclass:: maldideepkit.pretraining.ShufflePretext
   :members:
   :show-inheritance:

.. autoclass:: maldideepkit.pretraining.ShuffleMAEPretext
   :members:
   :show-inheritance:

.. autoclass:: maldideepkit.pretraining.ContrastivePretext
   :members:
   :show-inheritance:

.. autofunction:: maldideepkit.pretraining.register_pretext

.. autofunction:: maldideepkit.pretraining.get_pretext_class

.. autofunction:: maldideepkit.pretraining.register_backbone

.. autofunction:: maldideepkit.pretraining.get_backbone_class

.. autoclass:: maldideepkit.pretraining.SSLModel
   :members: forward_features, forward

.. autoclass:: maldideepkit.pretraining.SSLBatch
   :no-members:

make_pretraining_loaders
------------------------

.. autofunction:: maldideepkit.pretraining.make_pretraining_loaders

Examples
--------

Pre-train the backbone of the classifier you will fine-tune, then
fine-tune it (full fine-tuning and LP-FT):

.. code-block:: python

   import numpy as np
   from sklearn.base import clone
   from maldideepkit import MaldiCNNClassifier, MaldiSpectrumEncoder

   rng = np.random.default_rng(0)
   X_pool = np.abs(rng.standard_normal((2000, 6000))).astype("float32")
   X = np.abs(rng.standard_normal((200, 6000))).astype("float32")
   y = rng.integers(0, 2, size=200)

   template = MaldiCNNClassifier(random_state=0)
   encoder = MaldiSpectrumEncoder(backbone=template, pretext="mae", random_state=0)
   encoder.fit(X_pool)                  # no labels needed
   encoder.save("cnn_encoder")          # cnn_encoder.pt + cnn_encoder.json
   Z = encoder.transform(X)             # (200, encoder.feature_dim_)

   ft = clone(template).set_params(init_from="cnn_encoder").fit(X, y)
   lpft = clone(template).set_params(
       init_from="cnn_encoder", freeze_trunk_epochs=10,
   ).fit(X, y)
   print(lpft.trunk_unfroze_, lpft.best_epoch_phase_)

Model selection over the fine-tuning parameters (a checkpoint path is
loaded by every clone):

.. code-block:: python

   from sklearn.model_selection import GridSearchCV

   search = GridSearchCV(
       clone(template).set_params(init_from="cnn_encoder"),
       {"learning_rate": [1e-4, 1e-3], "freeze_trunk_epochs": [0, 10]},
       cv=3, scoring="roc_auc",
   ).fit(X, y)

Semi-supervised pre-training with a partially labelled pool
(classification labels, then a regression target):

.. code-block:: python

   y_pool = np.full(len(X_pool), -1)
   y_pool[:300] = rng.integers(0, 2, size=300)
   encoder = MaldiSpectrumEncoder(
       backbone=template, pretext="shuffle+mae", aux_lambda=0.5,
   ).fit(X_pool, y_pool)

   log2_mic = np.full(len(X_pool), np.nan)
   log2_mic[:300] = rng.normal(size=300)
   encoder = MaldiSpectrumEncoder(
       backbone=template, aux_lambda=0.5,
       aux_target="regression", unlabeled_marker=np.nan,
   ).fit(X_pool, log2_mic)

Dual-branch fine-tuning through the ``backbone=`` template:

.. code-block:: python

   from maldideepkit import MaldiDualBranchClassifier

   X_side = rng.standard_normal((200, 16)).astype("float32")
   clf = MaldiDualBranchClassifier(
       n_spec_bins=6000,
       backbone=clone(template).set_params(init_from="cnn_encoder"),
       fp_hidden_dims=(16,),
       random_state=0,
   ).fit(np.hstack([X, X_side]), y)
