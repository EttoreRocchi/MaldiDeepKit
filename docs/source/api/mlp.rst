MLP module
==========

Multilayer perceptron classifier with an optional sigmoid-gated
attention layer: a standard MLP with a per-unit gate on the first
hidden layer.

MaldiMLPClassifier
------------------

.. autoclass:: maldideepkit.MaldiMLPClassifier
   :members:
   :undoc-members:
   :show-inheritance:

SpectralAttentionMLP
--------------------

Low-level ``nn.Module`` wrapped by :class:`~maldideepkit.MaldiMLPClassifier`.
Exposed for users embedding the architecture into a larger network.

.. autoclass:: maldideepkit.attention.mlp.SpectralAttentionMLP
   :members:
   :undoc-members:
   :show-inheritance:

Examples
--------

.. code-block:: python

   import numpy as np
   from maldideepkit import MaldiMLPClassifier

   rng = np.random.default_rng(0)
   X = rng.standard_normal((200, 6000)).astype("float32")
   y = rng.integers(0, 2, size=200)

   clf = MaldiMLPClassifier(random_state=0).fit(X, y)

   # Gates of the last forward pass (after fit, at most 64 training rows):
   cached = clf.attention_weights_                 # (min(N, 64), hidden_dim)

   # Recompute for arbitrary inputs:
   weights = clf.get_attention_weights(X[:10])     # (10, hidden_dim)

   # Disable attention to get a plain MLP baseline:
   plain = MaldiMLPClassifier(use_attention=False, random_state=0).fit(X, y)
