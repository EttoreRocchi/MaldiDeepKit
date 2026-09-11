"""Iterative pseudo-labeling meta-classifier (Self-Training)."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.validation import check_is_fitted

__all__ = ["MaldiSelfTrainingClassifier"]


class MaldiSelfTrainingClassifier(ClassifierMixin, BaseEstimator):
    """Iterative pseudo-labeling wrapper around a base classifier.

    Matches the :class:`sklearn.semi_supervised.SelfTrainingClassifier`
    API: rows of ``y`` equal to ``unlabeled_marker`` (default ``-1``)
    are treated as unlabeled. On each iteration:

    1. Predict pseudo-labels on the current unlabeled pool via the
       base estimator's ``predict`` (so a decision threshold tuned
       with ``tune_threshold=True`` is respected).
    2. Accept pseudo-labels whose predicted-class probability is
       at least ``confidence_threshold``.
    3. Move accepted rows from the pool into the training set
       (**sticky** pseudo-labels: once added, they stay).
    4. Refit ``base_estimator`` on the expanded training set.

    The loop stops after ``n_iter`` iterations or as soon as a step
    accepts zero new pseudo-labels.

    Parameters
    ----------
    base_estimator : sklearn classifier or None, default=None
        Classifier with ``fit`` / ``predict`` / ``predict_proba`` and
        ``classes_``. When ``None``, defaults to
        :class:`maldideepkit.MaldiMLPClassifier` with
        ``random_state=self.random_state``.
    n_iter : int, default=3
        Maximum number of pseudo-labeling iterations.
    confidence_threshold : float, default=0.9
        Minimum predicted-class probability for a pseudo-label to be
        accepted.
    unlabeled_marker : object, default=-1
        Sentinel value in ``y`` that marks an unlabeled row. Any value
        comparable to ``y`` element-wise works, so string label spaces
        can use e.g. ``unlabeled_marker="?"``.
    random_state : int, default=0
        Seed propagated to ``base_estimator`` via
        ``set_params(random_state=...)`` when supported.
    verbose : bool, default=False
        Print one line per iteration with the accepted count.

    Attributes
    ----------
    estimator_ : sklearn classifier
        Final ``base_estimator`` clone after the iterative refit.
    classes_ : ndarray of shape (n_classes,)
        Class labels learned by ``estimator_``.
    history_ : dict[str, list]
        Per-iteration ``n_pseudolabels`` (count accepted),
        ``confidence_mean`` (mean predicted-class probability over the
        pool **before** filtering), ``n_remaining_unlabeled``.

    Notes
    -----
    *No unlabeled rows.* When ``y`` contains no ``unlabeled_marker``
    sentinels, the class falls back to plain supervised training on
    ``X`` / ``y`` and emits a :class:`UserWarning`.

    *sklearn compatibility.* The class inherits from
    :class:`sklearn.base.BaseEstimator` + :class:`ClassifierMixin`
    and exposes only standard ``fit`` / ``predict`` /
    ``predict_proba`` / ``score`` (so it works with ``Pipeline``,
    ``GridSearchCV``, and ``cross_val_score`` out of the box).

    Examples
    --------
    >>> import numpy as np
    >>> from maldideepkit import MaldiMLPClassifier, MaldiSelfTrainingClassifier
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((100, 32)).astype("float32")
    >>> y_full = rng.integers(0, 2, size=100)
    >>> y = y_full.copy()
    >>> y[20:] = -1  # only first 20 rows are labeled
    >>> clf = MaldiSelfTrainingClassifier(
    ...     base_estimator=MaldiMLPClassifier(
    ...         epochs=1, batch_size=8, random_state=0
    ...     ),
    ...     n_iter=2,
    ...     confidence_threshold=0.7,
    ... ).fit(X, y)
    >>> clf.predict(X).shape
    (100,)
    """

    def __init__(
        self,
        base_estimator: Any = None,
        *,
        n_iter: int = 3,
        confidence_threshold: float = 0.9,
        unlabeled_marker: Any = -1,
        random_state: int = 0,
        verbose: bool = False,
    ) -> None:
        self.base_estimator = base_estimator
        self.n_iter = n_iter
        self.confidence_threshold = confidence_threshold
        self.unlabeled_marker = unlabeled_marker
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X: Any, y: Any) -> MaldiSelfTrainingClassifier:
        """Fit the iterative pseudo-labeling loop.

        Parameters
        ----------
        X : array-like or MaldiSet of shape (n_samples, n_features)
            Spectra (or any feature matrix). Accepts the same input
            types as the wrapped estimator, including
            ``maldiamrkit.MaldiSet`` objects and DataFrames.
        y : array-like of shape (n_samples,)
            Labels. Rows whose value equals ``self.unlabeled_marker``
            are treated as unlabeled and pseudo-labeled iteratively.
            Integer, string and object labels are all supported; the
            dtype is preserved, so ``classes_`` comes back in the
            caller's own label space. Note that mixing string labels
            with the default numeric ``unlabeled_marker=-1`` needs an
            object array (a plain ``"<U"`` array can never equal
            ``-1``, so every row would count as labeled).

        Returns
        -------
        self : MaldiSelfTrainingClassifier
        """
        from ..base.data import _to_numpy

        X_arr = _to_numpy(X)
        y_arr = np.asarray(y)
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError(
                f"X has {X_arr.shape[0]} rows but y has "
                f"{y_arr.shape[0]} - they must match."
            )

        base = self._resolve_base_estimator()
        labeled_mask = y_arr != self.unlabeled_marker
        if not labeled_mask.any():
            raise ValueError(
                "MaldiSelfTrainingClassifier needs at least one labeled "
                f"sample (got y entirely == unlabeled_marker="
                f"{self.unlabeled_marker!r})."
            )
        if labeled_mask.all():
            warnings.warn(
                "MaldiSelfTrainingClassifier.fit() received no "
                "unlabeled rows; the iterative loop is a no-op and "
                "the base estimator will be fitted on the labeled "
                "set directly. Pass rows with "
                f"y == {self.unlabeled_marker!r} to enable "
                "pseudo-labeling.",
                UserWarning,
                stacklevel=2,
            )
            base.fit(X_arr, self._as_labels(y_arr))
            self.estimator_ = base
            self.classes_ = np.asarray(base.classes_)
            self.history_ = {
                "n_pseudolabels": [],
                "confidence_mean": [],
                "n_remaining_unlabeled": [],
            }
            return self

        X_train = X_arr[labeled_mask]
        y_train = self._as_labels(y_arr[labeled_mask])
        X_pool = X_arr[~labeled_mask]

        self.history_ = {
            "n_pseudolabels": [],
            "confidence_mean": [],
            "n_remaining_unlabeled": [],
        }

        base.fit(X_train, y_train)

        for it in range(int(self.n_iter)):
            if X_pool.shape[0] == 0:
                break
            proba = np.asarray(base.predict_proba(X_pool))
            pseudo = np.asarray(base.predict(X_pool))
            classes = np.asarray(base.classes_)
            pred_idx = np.searchsorted(classes, pseudo)
            conf = proba[np.arange(proba.shape[0]), pred_idx]
            accept = conf >= float(self.confidence_threshold)
            n_accept = int(accept.sum())

            self.history_["n_pseudolabels"].append(n_accept)
            self.history_["confidence_mean"].append(float(conf.mean()))
            self.history_["n_remaining_unlabeled"].append(
                int(X_pool.shape[0] - n_accept)
            )
            if self.verbose:
                print(
                    f"[MaldiSelfTrainingClassifier] iter={it + 1} "
                    f"accepted={n_accept}/{X_pool.shape[0]} "
                    f"mean_conf={conf.mean():.4f}"
                )

            if n_accept == 0:
                break

            X_train = np.concatenate([X_train, X_pool[accept]], axis=0)
            y_train = np.concatenate([y_train, self._as_labels(pseudo[accept])], axis=0)
            X_pool = X_pool[~accept]

            base = clone(base)
            self._maybe_seed(base)
            base.fit(X_train, y_train)

        self.estimator_ = base
        self.classes_ = np.asarray(base.classes_)
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict labels via the final fitted base estimator."""
        check_is_fitted(self, "estimator_")
        return np.asarray(self.estimator_.predict(X))

    def predict_proba(self, X: Any) -> np.ndarray:
        """Predict probabilities via the final fitted base estimator."""
        check_is_fitted(self, "estimator_")
        return np.asarray(self.estimator_.predict_proba(X))

    def score(self, X: Any, y: Any) -> float:
        """Mean accuracy on ``(X, y)``."""
        check_is_fitted(self, "estimator_")
        return float(self.estimator_.score(X, y))

    def save(self, path: Any) -> None:
        """Persist the fitted inner estimator.

        Forwards to :meth:`BaseSpectralClassifier.save` on
        :attr:`estimator_`, so the final model produced by the
        self-training loop is saved in the same ``.pt`` + ``.json``
        format as a plain :class:`BaseSpectralClassifier`.
        """
        check_is_fitted(self, "estimator_")
        if not hasattr(self.estimator_, "save"):
            raise AttributeError(
                f"Base estimator {type(self.estimator_).__name__} does "
                "not implement save(); cannot persist this self-training "
                "wrapper."
            )
        self.estimator_.save(path)

    @staticmethod
    def _as_labels(y: np.ndarray) -> np.ndarray:
        """Normalise integer-like labels to ``int64``, pass anything else through.

        String and object labels reach the base estimator untouched, so
        :attr:`classes_` keeps the caller's dtype exactly as
        :class:`~maldideepkit.BaseSpectralClassifier` does. Float labels
        are left alone rather than truncated.
        """
        if y.dtype.kind in "iub":
            return y.astype(np.int64, copy=False)
        return y

    def _resolve_base_estimator(self) -> Any:
        """Clone the user-supplied base estimator or build a default MLP."""
        if self.base_estimator is None:
            from ..attention.mlp import MaldiMLPClassifier

            base = MaldiMLPClassifier(random_state=int(self.random_state))
        else:
            base = clone(self.base_estimator)
        self._maybe_seed(base)
        return base

    def _maybe_seed(self, estimator: Any) -> None:
        """Propagate self.random_state to the estimator if supported."""
        try:
            estimator.set_params(random_state=int(self.random_state))
        except (TypeError, ValueError):
            pass
