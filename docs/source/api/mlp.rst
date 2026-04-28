MLP Module
==========

Multilayer perceptron classifier with an optional sigmoid-gated
attention layer. Engineering adaptation of a standard MLP with a per-unit
gate on the first hidden layer; not a novel architecture.

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

Attention Inspection Example
----------------------------

.. code-block:: python

    import numpy as np
    from maldideepkit import MaldiMLPClassifier

    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 6000)).astype("float32")
    y = rng.integers(0, 2, size=200)

    clf = MaldiMLPClassifier(random_state=0).fit(X, y)

    # Per-sample attention gates cached at the end of fit:
    cached = clf.attention_weights_                 # (N, hidden_dim)

    # Recompute for arbitrary inputs:
    weights = clf.get_attention_weights(X[:10])     # (10, hidden_dim)

    # Disable attention to get a plain MLP baseline:
    plain = MaldiMLPClassifier(use_attention=False, random_state=0).fit(X, y)
