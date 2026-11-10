"""Iterative pseudo-labeling meta-classifier (Self-Training)."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.validation import check_is_fitted

from ..base._registry import get_backbone_class
from ..base._serialization import _decode_param, _encode_param
from ..base.classifier import _checkpoint_base, _checkpoint_file

__all__ = ["MaldiSelfTrainingClassifier"]


class MaldiSelfTrainingClassifier(ClassifierMixin, BaseEstimator):
    """Iterative pseudo-labeling wrapper around a base classifier.

    Matches the :class:`sklearn.semi_supervised.SelfTrainingClassifier`
    API: rows of ``y`` equal to ``unlabeled_marker`` are treated as
    unlabeled. On each iteration:

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
    base_estimator : scikit-learn classifier, str or None, default=None
        Classifier with ``fit`` / ``predict`` / ``predict_proba`` and
        ``classes_``, or the name of a backbone (``"mlp"``, ``"cnn"``,
        ``"resnet"``, ``"transformer"``, or a name added with
        :func:`~maldideepkit.pretraining.register_backbone`), built with the
        defaults of its class. ``None`` uses
        :class:`~maldideepkit.MaldiMLPClassifier` with
        ``random_state=self.random_state``.
    n_iter : int, default=3
        Maximum number of pseudo-labeling iterations.
    confidence_threshold : float, default=0.9
        Minimum predicted-class probability for a pseudo-label to be
        accepted.
    unlabeled_marker : object, default=-1
        Value of ``y`` that marks an unlabeled row. Any value comparable
        to ``y`` element-wise works, so string label spaces can use e.g.
        ``unlabeled_marker="?"``.
    random_state : int, default=0
        Seed propagated to ``base_estimator`` via
        ``set_params(random_state=...)`` when supported.
    verbose : bool, default=False
        If ``True``, print one line per iteration (with the number of
        accepted pseudo-labels).

    Attributes
    ----------
    estimator_ : scikit-learn classifier
        Final ``base_estimator`` clone after the iterative refit.
    classes_ : ndarray of shape (n_classes,)
        Class labels learned by ``estimator_``.
    history_ : dict[str, list]
        Per-iteration ``n_pseudolabels`` (count accepted),
        ``confidence_mean`` (mean predicted-class probability over the
        pool **before** filtering), and ``n_remaining_unlabeled``.

    Notes
    -----
    *No unlabeled rows.* When ``y`` contains no ``unlabeled_marker``
    sentinels, the class falls back to plain supervised training on
    ``X`` / ``y`` and emits a :class:`UserWarning`.

    *scikit-learn compatibility.* The class inherits from
    :class:`sklearn.base.BaseEstimator` and
    :class:`sklearn.base.ClassifierMixin`, with the standard ``fit`` /
    ``predict`` / ``predict_proba`` / ``score``, so ``Pipeline``,
    ``GridSearchCV``, and ``cross_val_score`` accept it. It also has
    :meth:`get_embeddings`, :meth:`save`, and :meth:`load`.

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

    def fit(
        self, X: Any, y: Any, *, sample_weight: Any | None = None
    ) -> MaldiSelfTrainingClassifier:
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
            Integer, string, and object labels are all supported; the
            dtype is preserved, so ``classes_`` comes back in the
            caller's own label space. Note that mixing string labels
            with the default numeric ``unlabeled_marker=-1`` needs an
            object array (a plain ``"<U"`` array can never equal
            ``-1``, so every row would count as labeled).
        sample_weight : array-like of shape (n_samples,) or None, default=None
            Per-row weights passed to every fit of the base estimator. A
            pseudo-labeled row keeps the weight of its row.

        Returns
        -------
        self : MaldiSelfTrainingClassifier
            The fitted estimator.
        """
        from ..base.data import _to_numpy

        X_arr = _to_numpy(X)
        y_arr = np.asarray(y)
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError(
                f"X has {X_arr.shape[0]} rows but y has "
                f"{y_arr.shape[0]}; they must match."
            )
        weights = None if sample_weight is None else np.asarray(sample_weight)
        if weights is not None and weights.shape[0] != X_arr.shape[0]:
            raise ValueError(
                f"X has {X_arr.shape[0]} rows but sample_weight has "
                f"{weights.shape[0]}; they must match."
            )

        def fit_base(estimator: Any, X_fit: Any, y_fit: Any, w_fit: Any) -> None:
            if w_fit is None:
                estimator.fit(X_fit, y_fit)
            else:
                estimator.fit(X_fit, y_fit, sample_weight=w_fit)

        base = self._resolve_base_estimator()
        labeled_mask = y_arr != self.unlabeled_marker
        if not labeled_mask.any():
            raise ValueError(
                "MaldiSelfTrainingClassifier needs at least one labeled "
                "sample; got y entirely equal to unlabeled_marker="
                f"{self.unlabeled_marker!r}."
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
            fit_base(base, X_arr, self._as_labels(y_arr), weights)
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
        w_train = None if weights is None else weights[labeled_mask]
        w_pool = None if weights is None else weights[~labeled_mask]

        self.history_ = {
            "n_pseudolabels": [],
            "confidence_mean": [],
            "n_remaining_unlabeled": [],
        }

        fit_base(base, X_train, y_train, w_train)

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
            if weights is not None:
                w_train = np.concatenate([w_train, w_pool[accept]], axis=0)
                w_pool = w_pool[~accept]

            base = clone(base)
            self._maybe_seed(base)
            fit_base(base, X_train, y_train, w_train)

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

    def get_embeddings(self, X: Any) -> np.ndarray:
        """Return the embeddings of the final fitted base estimator."""
        check_is_fitted(self, "estimator_")
        return np.asarray(self.estimator_.get_embeddings(X))

    def save(self, path: str | Path) -> None:
        """Persist the fitted estimator (see :meth:`load`).

        The final model of the self-training loop goes through
        :meth:`~maldideepkit.BaseSpectralClassifier.save` on :attr:`estimator_`, so it
        is saved in the same ``.pt`` + ``.json`` format as a plain
        :class:`~maldideepkit.BaseSpectralClassifier`; the parameters of the loop and
        its history go to ``<path>.self_training.json``.
        """
        check_is_fitted(self, "estimator_")
        if not hasattr(self.estimator_, "save"):
            raise AttributeError(
                f"Base estimator {type(self.estimator_).__name__} does "
                "not implement save(); cannot persist this self-training "
                "wrapper."
            )
        self.estimator_.save(path)
        base = _checkpoint_base(path)
        params = self.get_params(deep=False)
        params["base_estimator"] = (
            params["base_estimator"]
            if isinstance(params["base_estimator"], str)
            else None
        )
        meta = {
            "kind": "self_training",
            "version": 1,
            "params": {k: _encode_param(k, v) for k, v in params.items()},
            "classes_": np.asarray(self.classes_).tolist(),
            "history_": self.history_,
        }
        with open(_checkpoint_file(base, ".self_training.json"), "w") as fh:
            json.dump(meta, fh, indent=2)

    @classmethod
    def load(
        cls, path: str | Path, *, trusted: bool = False
    ) -> MaldiSelfTrainingClassifier:
        """Inverse of :meth:`save`.

        The final model is loaded with
        :meth:`~maldideepkit.BaseSpectralClassifier.load` and the
        parameters of the loop from ``<path>.self_training.json``, both
        under the same ``trusted`` rule. A model saved by v0.3.0 has no
        ``.self_training.json``: it loads with the default parameters of
        the loop and an empty history.

        Parameters
        ----------
        path : str or Path
            Base path given to :meth:`save`.
        trusted : bool, default=False
            With the default, no new module is imported: every class
            named by the checkpoint must come from MaldiDeepKit or from
            an imported module. An object given as ``optimizer``,
            ``scheduler``, ``loss`` or ``callbacks`` must also be a torch
            module, optimizer or scheduler, or a ``Callback``, and a
            function must be defined in MaldiDeepKit or in
            ``torch.nn.functional``. ``True`` lifts these limits, for
            checkpoints from a trusted source.

        Returns
        -------
        MaldiSelfTrainingClassifier
            The fitted estimator. Its ``base_estimator`` is an unfitted
            clone of the loaded model, unless it was given by name.

        Raises
        ------
        ValueError
            If the checkpoint names a class or a function that the
            ``trusted`` rule refuses or that cannot be imported.
        FileNotFoundError
            If a file of the saved model is missing.

        Notes
        -----
        A ``.warper.pkl`` is a pickle, read whatever ``trusted`` is:
        load checkpoints with a warper only from trusted sources.
        """
        from ..base.classifier import BaseSpectralClassifier

        estimator = BaseSpectralClassifier.load(path, trusted=trusted)
        meta_path = _checkpoint_file(_checkpoint_base(path), ".self_training.json")
        params: dict[str, Any] = {}
        history: dict[str, list[Any]] = {
            "n_pseudolabels": [],
            "confidence_mean": [],
            "n_remaining_unlabeled": [],
        }
        if meta_path.exists():
            with open(meta_path) as fh:
                meta = json.load(fh)
            params = {
                k: _decode_param(v, trusted=trusted) for k, v in meta["params"].items()
            }
            history = meta["history_"]
        if params.get("base_estimator") is None:
            params["base_estimator"] = clone(estimator)
        instance = cls(**params)
        instance.estimator_ = estimator
        instance.classes_ = np.asarray(estimator.classes_)
        instance.history_ = history
        return instance

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
        """Clone the base estimator, build it from its name, or use a default MLP."""
        if self.base_estimator is None:
            from ..attention.mlp import MaldiMLPClassifier

            base = MaldiMLPClassifier(random_state=int(self.random_state))
        elif isinstance(self.base_estimator, str):
            base = get_backbone_class(self.base_estimator)()
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
