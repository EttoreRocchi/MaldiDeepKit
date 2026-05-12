"""Tests for the ``warm_start=True`` mode of :meth:`BaseSpectralClassifier.fit`.

``warm_start=True`` instructs ``fit()`` to reuse the existing
``self.model_`` as the starting point of training instead of rebuilding
the network from scratch via ``_build_model()``. This is required for
federated-learning, continual-learning, and fine-tuning use cases where
the caller has written specific weights into ``self.model_.state_dict()``
just before invoking ``fit``.
"""

from __future__ import annotations

import numpy as np
import pytest

from maldideepkit import MaldiCNNClassifier, MaldiMLPClassifier


def _tiny_mlp() -> MaldiMLPClassifier:
    return MaldiMLPClassifier(
        input_dim=128,
        hidden_dim=8,
        epochs=1,
        batch_size=8,
        val_fraction=0.2,
        warmup_epochs=0,
        early_stopping_patience=1,
        random_state=0,
        device="cpu",
    )


def _tiny_cnn() -> MaldiCNNClassifier:
    return MaldiCNNClassifier(
        input_dim=128,
        epochs=1,
        batch_size=8,
        val_fraction=0.2,
        warmup_epochs=0,
        early_stopping_patience=1,
        random_state=0,
        device="cpu",
    )


def test_default_fit_rebuilds_model(synthetic_binary):
    """``warm_start`` defaults to False, so each ``fit()`` rebuilds the module."""
    X, y = synthetic_binary
    clf = _tiny_mlp()
    clf.fit(X, y)
    clf.model_._mfk_sentinel = "rebuilt-if-missing"  # type: ignore[attr-defined]

    clf.fit(X, y)  # warm_start defaults to False
    assert not hasattr(clf.model_, "_mfk_sentinel"), (
        "Default fit() should rebuild clf.model_ and drop attached attributes."
    )


def test_warm_start_true_reuses_existing_model_instance(synthetic_binary):
    """warm_start=True must reuse clf.model_, not call _build_model()."""
    X, y = synthetic_binary
    clf = _tiny_mlp()
    clf.fit(X, y)
    # Tag the existing module; if fit() rebuilds, the tag will be gone.
    clf.model_._mfk_sentinel = "warm-started"  # type: ignore[attr-defined]

    clf.fit(X, y, warm_start=True)
    assert getattr(clf.model_, "_mfk_sentinel", None) == "warm-started", (
        "warm_start=True must keep the previously-fitted clf.model_ instance "
        "alive instead of rebuilding via _build_model()."
    )


def test_warm_start_true_preserves_pre_fit_weights_as_starting_point(
    synthetic_binary,
):
    """The starting weights for a warm-started fit must be the existing state_dict.

    Compare two trajectories that share the same training data + seed but
    differ in pre-fit weights. If warm_start=True actually uses the
    pre-fit weights, the resulting models must differ.
    """
    X, y = synthetic_binary

    clf_a = _tiny_mlp()
    clf_a.fit(X, y)
    # Zero every parameter to give the second fit() a degenerate starting point.
    import torch

    with torch.no_grad():
        for p in clf_a.model_.parameters():
            p.zero_()

    clf_b = _tiny_mlp()
    clf_b.fit(X, y)
    # clf_b keeps its real trained weights as the pre-fit starting point.

    # Now run a second warm-started fit() on each.
    clf_a.fit(X, y, warm_start=True)
    clf_b.fit(X, y, warm_start=True)

    # The two trajectories started from different weights, so they should
    # diverge. If warm_start was a no-op (silently rebuilding), both would
    # converge to the same point (same seed, same data, fresh init).
    p_a = next(clf_a.model_.parameters()).detach().cpu().numpy()
    p_b = next(clf_b.model_.parameters()).detach().cpu().numpy()
    assert not np.allclose(p_a, p_b), (
        "Two warm-started fits with different pre-fit weights must diverge."
    )


def test_warm_start_false_yields_deterministic_trajectory(synthetic_binary):
    """Confirms that the divergence above is genuinely from warm_start: with
    warm_start=False both rebuild from the same seed and converge identically.
    """
    X, y = synthetic_binary

    clf_a = _tiny_mlp()
    clf_a.fit(X, y)
    import torch

    with torch.no_grad():
        for p in clf_a.model_.parameters():
            p.zero_()  # pollute pre-fit weights

    clf_b = _tiny_mlp()
    clf_b.fit(X, y)
    # clf_b's pre-fit weights are real, not zeroed.

    clf_a.fit(X, y)  # warm_start=False by default -> rebuild from seed.
    clf_b.fit(X, y)

    p_a = next(clf_a.model_.parameters()).detach().cpu().numpy()
    p_b = next(clf_b.model_.parameters()).detach().cpu().numpy()
    np.testing.assert_allclose(p_a, p_b, rtol=0, atol=1e-6)


def test_warm_start_true_without_prior_fit_falls_back_to_build(synthetic_binary):
    """warm_start=True on an unfitted estimator must silently build a fresh model."""
    X, y = synthetic_binary
    clf = _tiny_mlp()
    assert not hasattr(clf, "model_")
    clf.fit(X, y, warm_start=True)
    assert hasattr(clf, "model_")
    assert clf.model_ is not None


def test_mlp_override_forwards_warm_start_kwarg(synthetic_binary):
    """MaldiMLPClassifier overrides fit() and must pass warm_start through."""
    X, y = synthetic_binary
    clf = _tiny_mlp()
    clf.fit(X, y)
    clf.model_._mfk_sentinel = "tagged"  # type: ignore[attr-defined]
    clf.fit(X, y, warm_start=True)
    assert getattr(clf.model_, "_mfk_sentinel", None) == "tagged"


def test_cnn_inherits_warm_start_from_base(synthetic_binary):
    """CNN does not override fit(), so it gets warm_start for free from the base."""
    X, y = synthetic_binary
    clf = _tiny_cnn()
    clf.fit(X, y)
    clf.model_._mfk_sentinel = "cnn-warm-start"  # type: ignore[attr-defined]
    clf.fit(X, y, warm_start=True)
    assert getattr(clf.model_, "_mfk_sentinel", None) == "cnn-warm-start"


def test_fit_remains_callable_with_two_positional_args(synthetic_binary):
    """Existing callers using fit(X, y) without keyword args still work."""
    X, y = synthetic_binary
    clf = _tiny_mlp()
    clf.fit(X, y)
    assert hasattr(clf, "model_")


def test_fit_rejects_warm_start_as_positional(synthetic_binary):
    """warm_start is keyword-only; passing it positionally should error."""
    X, y = synthetic_binary
    clf = _tiny_mlp()
    with pytest.raises(TypeError):
        clf.fit(X, y, True)  # noqa: FBT003 - intentional misuse
