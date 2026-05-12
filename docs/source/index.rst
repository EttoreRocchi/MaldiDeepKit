:html_theme.sidebar_secondary.remove: true

.. image:: _static/maldideepkit_logo.png
   :align: center
   :width: 280px
   :class: only-light

.. image:: _static/maldideepkit_logo.png
   :align: center
   :width: 280px
   :class: only-dark

.. rst-class:: hero-section

MaldiDeepKit Documentation
==========================

A catalog of sklearn-compatible deep learning classifiers for MALDI-TOF
mass spectrometry. Four PyTorch architectures - MLP (with optional
sigmoid-gated attention), 1-D CNN, 1-D ResNet, and 1-D Vision
Transformer - wrapped in a unified estimator API, with defaults
calibrated for ~6000-bin MALDI-TOF input.

.. container:: sd-d-flex-row sd-flex-justify-content-center sd-gap-2 sd-mb-4

   .. button-link:: installation.html
      :color: primary
      :shadow:

      Installation

   .. button-link:: api/index.html
      :color: primary
      :outline:
      :shadow:

      API Reference

   .. button-link:: quickstart.html
      :color: primary
      :outline:
      :shadow:

      Quickstart Guide

----

Key Features
------------

.. grid:: 2 2 3 3
   :gutter: 3
   :class-container: feature-grid

   .. grid-item-card:: Four Architectures
      :link: api/index.html
      :link-type: url

      MLP with optional attention, 1-D CNN, 1-D ResNet, and 1-D Vision
      Transformer - all calibrated for ~6000-bin MALDI-TOF spectra
      out of the box.

   .. grid-item-card:: Sklearn-Compatible
      :link: quickstart.html#fitting-a-classifier
      :link-type: url

      Every classifier implements ``fit`` / ``predict`` / ``predict_proba`` /
      ``score`` / ``get_params`` / ``set_params`` and plugs into
      ``Pipeline``, ``cross_val_score``, and ``GridSearchCV``.

   .. grid-item-card:: MaldiSet Integration
      :link: quickstart.html#integration-with-maldiamrkit
      :link-type: url

      Pass a :class:`maldiamrkit.MaldiSet` directly to ``fit`` / ``predict``;
      MaldiDeepKit duck-types on the DataFrame-like ``.X`` attribute, so
      MaldiSuite's data model flows end-to-end.

   .. grid-item-card:: Attention Inspection
      :link: quickstart.html#inspecting-attention-weights
      :link-type: url

      ``MaldiMLPClassifier`` exposes per-sample sigmoid-gated attention
      via ``attention_weights_`` and ``get_attention_weights(X)``.

   .. grid-item-card:: Training Recipes Built-in
      :link: quickstart.html#training-recipe-and-lr-schedule
      :link-type: url

      Linear warmup + cosine annealing, AdamW-on-weight-decay dispatch,
      gradient clipping, AMP, SWA, focal-loss, and threshold tuning. Deep
      models (ResNet / Transformer) ship with the recipes they need to
      converge out of the box.

   .. grid-item-card:: Leak-Safe Spectral Warping
      :link: quickstart.html#spectral-warping-pre-scaling
      :link-type: url

      Pass a ``Warping`` (or any sklearn transformer) via ``warping=``;
      it's fitted on the training fold only and applied before
      per-feature standardization during training *and* inference.

   .. grid-item-card:: Auto-Scaling
      :link: spectrum_scaling.html
      :link-type: url

      ``Classifier.from_spectrum(bin_width, input_dim)`` rescales
      conv kernels and patches when the spectrum layout deviates from
      the reference 6000-bin / 3 Da default.

   .. grid-item-card:: Calibration & Threshold Tuning
      :link: quickstart.html#probability-calibration
      :link-type: url

      Post-hoc temperature scaling and balanced-accuracy / F1 / Youden
      threshold tuning on the validation split, all togglable via
      classifier kwargs.

   .. grid-item-card:: Uncertainty Quantification
      :link: api/uncertainty.html
      :link-type: url

      Three drop-in estimators on a shared
      ``predict_with_uncertainty`` interface: Monte Carlo Dropout,
      Laplace approximation, and split conformal prediction (LAC).

   .. grid-item-card:: Reproducible Training
      :link: api/base.html
      :link-type: url

      Shared ``BaseSpectralClassifier`` seeds Python, NumPy, and PyTorch
      from ``random_state`` so identical configs produce identical
      weights and predictions.

   .. grid-item-card:: Strict Persistence
      :link: quickstart.html#persistence
      :link-type: url

      ``save()`` writes a state-dict ``.pt`` + hyperparameter ``.json``
      pair (and a sibling ``.warper.pkl`` if a warper was fitted);
      ``load()`` fails fast on class or ``input_dim`` mismatches.

   .. grid-item-card:: CPU-Friendly
      :link: installation.html
      :link-type: url

      CPU fallback is fully supported and is what the project's CI runs
      against; CUDA significantly speeds up training across all four
      architectures.

   .. grid-item-card:: MaldiSuite Ecosystem
      :link: papers.html
      :link-type: url

      Sibling of ``MaldiAMRKit`` (preprocessing) and ``MaldiBatchKit``
      (batch correction) - three packages sharing the same data model.

----

Quick Example
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

Train/test without leakage - the optional spectral warper is fit on
the training fold only and applied to held-out samples via the same
fitted parameters at ``predict`` time:

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
   :caption: Get Started

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
