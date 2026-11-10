:html_theme.sidebar_secondary.remove: true

.. image:: _static/maldideepkit_logo.png
   :align: center
   :width: 280px
   :class: only-light

.. image:: _static/maldideepkit_logo_dark.png
   :align: center
   :width: 280px
   :class: only-dark

.. rst-class:: hero-section

MaldiDeepKit documentation
==========================

A catalogue of sklearn-compatible deep learning classifiers for MALDI-TOF
mass spectrometry: four PyTorch backbones (MLP with optional
sigmoid-gated attention, 1-D CNN, 1-D ResNet, and 1-D Vision
Transformer), a dual-branch classifier for a spectrum plus side
features, a self-training classifier for partly labelled data, and a
self-supervised encoder that pre-trains the backbones on unlabelled
spectra. All of them share one estimator API, with defaults set for
~6000-bin MALDI-TOF spectra.

.. container:: sd-d-flex-row sd-flex-justify-content-center sd-gap-2 sd-mb-4

   .. button-link:: installation.html
      :color: primary
      :shadow:

      Installation

   .. button-link:: api/index.html
      :color: primary
      :outline:
      :shadow:

      API reference

   .. button-link:: quickstart.html
      :color: primary
      :outline:
      :shadow:

      Quickstart guide

----

Key features
------------

.. grid:: 2 2 3 3
   :gutter: 3
   :class-container: feature-grid

   .. grid-item-card:: Four architectures
      :link: api/index.html
      :link-type: url

      MLP with optional attention, 1-D CNN, 1-D ResNet, and 1-D Vision
      Transformer (all with defaults set for ~6000-bin MALDI-TOF
      spectra).

   .. grid-item-card:: Compatible with scikit-learn
      :link: quickstart.html#fitting-a-classifier
      :link-type: url

      Every classifier implements ``fit`` / ``predict`` / ``predict_proba`` /
      ``score`` / ``get_params`` / ``set_params`` and plugs into
      ``Pipeline``, ``cross_val_score``, and ``GridSearchCV``.

   .. grid-item-card:: MaldiSet integration
      :link: quickstart.html#integration-with-maldiamrkit
      :link-type: url

      Pass a :class:`maldiamrkit.MaldiSet` directly to ``fit`` / ``predict``;
      MaldiDeepKit reads its DataFrame-like ``.X`` attribute.

   .. grid-item-card:: Attention inspection
      :link: quickstart.html#inspecting-attention-weights
      :link-type: url

      ``MaldiMLPClassifier`` exposes per-sample sigmoid-gated attention
      through ``attention_weights_`` and ``get_attention_weights(X)``.

   .. grid-item-card:: Built-in training recipes
      :link: quickstart.html#training-recipe-and-lr-schedule
      :link-type: url

      Linear warmup and cosine annealing, AdamW when ``weight_decay > 0``,
      gradient clipping, AMP, SWA, focal loss, and threshold tuning. The
      ResNet and Transformer defaults include the recipe they need to
      converge.

   .. grid-item-card:: Leak-safe spectral warping
      :link: quickstart.html#spectral-warping-pre-scaling
      :link-type: url

      Pass a ``Warping`` (or any scikit-learn transformer) as
      ``warping=``; it is fitted on the training split only and applied
      before per-feature standardisation during training *and*
      inference.

   .. grid-item-card:: Auto-scaling
      :link: spectrum_scaling.html
      :link-type: url

      ``Classifier.from_spectrum(bin_width, input_dim)`` rescales the
      conv kernels when the spectrum layout deviates from the reference
      6000-bin / 3 Da default.

   .. grid-item-card:: Calibration and threshold tuning
      :link: quickstart.html#probability-calibration
      :link-type: url

      Post-hoc temperature scaling and balanced-accuracy / F1 / Youden
      threshold tuning on the validation split, both switched on
      through classifier parameters.

   .. grid-item-card:: Uncertainty quantification
      :link: api/uncertainty.html
      :link-type: url

      Three drop-in estimators on a shared
      ``predict_with_uncertainty`` interface: Monte Carlo Dropout,
      Laplace approximation, and split conformal prediction (LAC).

   .. grid-item-card:: Self-supervised pre-training
      :link: api/pretraining.html
      :link-type: url

      ``MaldiSpectrumEncoder`` pre-trains any backbone on unlabelled
      spectra (masked reconstruction, intensity-weighted bin shuffling,
      or contrastive views); classifiers start from it with ``init_from=`` and
      optional linear probing (``freeze_trunk_epochs=``).

   .. grid-item-card:: Reproducible training
      :link: api/base.html
      :link-type: url

      Shared ``BaseSpectralClassifier`` seeds Python, NumPy, and PyTorch
      from ``random_state``, so on CPU, with the same number of threads,
      identical configurations produce identical weights and predictions.

   .. grid-item-card:: Strict persistence
      :link: quickstart.html#persistence
      :link-type: url

      ``save()`` writes a state-dict ``.pt`` + hyperparameter ``.json``
      pair (and a ``.warper.pkl`` when a warper was fitted); ``load()``
      raises on a class mismatch, and ``predict`` on a wrong number of
      bins.

   .. grid-item-card:: GPU or CPU
      :link: installation.html
      :link-type: url

      With the default ``device="auto"``, every classifier trains on a
      CUDA GPU when one is available and on CPU otherwise.

   .. grid-item-card:: MaldiSuite ecosystem
      :link: papers.html
      :link-type: url

      Part of MaldiSuite with ``MaldiAMRKit`` and ``MaldiBatchKit``: three
      packages that share the ``MaldiSet`` data model and can be used on
      their own or together.

----

Quick example
-------------

.. code-block:: python

   import numpy as np
   from maldideepkit import MaldiMLPClassifier

   rng = np.random.default_rng(0)
   X = rng.standard_normal((200, 6000)).astype("float32")
   y = rng.integers(0, 2, size=200)

   clf = MaldiMLPClassifier(random_state=0)
   clf.fit(X, y)

   proba = clf.predict_proba(X)
   weights = clf.get_attention_weights(X[:10])   # (10, hidden_dim)

Train/test without leakage (the optional spectral warper is fitted
on the training split only and applied to held-out samples with the
same fitted parameters at ``predict`` time):

.. code-block:: python

   from sklearn.model_selection import train_test_split
   from maldiamrkit.alignment import Warping
   from maldideepkit import MaldiCNNClassifier

   X_train, X_test, y_train, y_test = train_test_split(
       X, y, stratify=y, random_state=0,
   )

   clf = MaldiCNNClassifier(
       warping=Warping(method="shift", n_jobs=-1),
       standardize=True,
       random_state=0,
   )
   clf.fit(X_train, y_train)            # warper fit on train only
   acc = clf.score(X_test, y_test)      # warper reused on test

----

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Get started

   installation
   quickstart
   spectrum_scaling

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Reference

   api/index

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Resources

   tutorials/index
   contributing
   papers
   changelog
