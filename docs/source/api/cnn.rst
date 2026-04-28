CNN Module
==========

Stacked 1-D convolutional blocks (``Conv1d`` + ``BatchNorm1d`` + ``ReLU``
+ ``MaxPool1d`` + ``Dropout``) with a flatten + dense classification
head. Engineering adaptation of the standard 1-D CNN template to
binned MALDI-TOF spectra; not a novel architecture.

MaldiCNNClassifier
------------------

.. autoclass:: maldideepkit.MaldiCNNClassifier
   :members:
   :undoc-members:
   :show-inheritance:

SpectralCNN1D
-------------

.. autoclass:: maldideepkit.cnn.cnn.SpectralCNN1D
   :members:
   :undoc-members:
   :show-inheritance:

Examples
--------

.. code-block:: python

    import numpy as np
    from maldideepkit import MaldiCNNClassifier

    rng = np.random.default_rng(0)
    X = rng.standard_normal((400, 6000)).astype("float32")
    y = rng.integers(0, 2, size=400)

    clf = MaldiCNNClassifier(
        channels=(32, 64, 128, 128),
        kernel_size=7,             # scalar broadcasts to every block
        pool_size=2,
        random_state=0,
    ).fit(X, y)

Per-block kernel / pool progression:

.. code-block:: python

    clf = MaldiCNNClassifier(
        channels=(32, 64, 128, 128),
        kernel_size=(11, 7, 5, 3),  # wider early, narrower late
        pool_size=(2, 2, 2, 4),     # more aggressive pooling on last block
    )

Auto-scale for a different spectrum layout:

.. code-block:: python

    clf = MaldiCNNClassifier.from_spectrum(
        bin_width=6, input_dim=3000, random_state=0,
    )  # picks kernel_size=5 (scaled from 7 at the bin_width=3 reference)
