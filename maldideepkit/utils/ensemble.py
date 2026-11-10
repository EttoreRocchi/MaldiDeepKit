"""Simple mean-of-``predict_proba`` ensemble for spectral classifiers.

A thin wrapper that fits each member independently and averages
``predict_proba`` across them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

# Version of the ``.ensemble.json`` index: 1 up to v0.3.0, 2 since v0.4.0
# (dotted base names kept whole).
_INDEX_VERSION = 2


class SpectralEnsemble(ClassifierMixin, BaseEstimator):
    """Ensemble N fitted or unfitted spectral classifiers.

    When every member is already fitted, the ensemble predicts without
    :meth:`fit`. It is a scikit-learn estimator:
    :func:`~sklearn.base.clone` clones the members, and the ensemble
    works in :class:`~sklearn.model_selection.GridSearchCV`.

    Parameters
    ----------
    classifiers : sequence of BaseSpectralClassifier or str
        Classifier instances (fitted or not), or names (see
        :func:`~maldideepkit.pretraining.register_backbone`). :meth:`fit` calls the
        ``fit`` of each member in order (an instance is fitted in place).

    Attributes
    ----------
    classes_ : np.ndarray
        Class labels shared by the members, set by :meth:`fit` or by
        :meth:`load`. The members must have the same ``classes_``,
        else :meth:`fit` and :meth:`predict_proba` raise
        :class:`ValueError`.
    classifiers_ : list of BaseSpectralClassifier
        The members fitted by :meth:`fit` (or restored by :meth:`load`).
    """

    def __init__(self, classifiers: list[Any]) -> None:
        if not len(classifiers):
            raise ValueError("SpectralEnsemble needs at least one classifier.")
        self.classifiers = classifiers

    def fit(
        self, X: Any, y: Any, *, sample_weight: Any | None = None
    ) -> SpectralEnsemble:
        """Fit every member on the same ``(X, y)``.

        Parameters
        ----------
        X : array-like or MaldiSet of shape (n_samples, n_bins)
            Training spectra.
        y : array-like of shape (n_samples,)
            Labels.
        sample_weight : array-like of shape (n_samples,) or None, default=None
            Per-sample weights passed to every member.

        Returns
        -------
        SpectralEnsemble
            The fitted ensemble.

        Raises
        ------
        ValueError
            If the fitted members do not have the same ``classes_``.
        """
        from ..base._registry import get_backbone_class

        members = [
            get_backbone_class(clf)() if isinstance(clf, str) else clf
            for clf in self.classifiers
        ]
        first_classes: np.ndarray | None = None
        for i, clf in enumerate(members):
            if sample_weight is None:
                clf.fit(X, y)
            else:
                clf.fit(X, y, sample_weight=sample_weight)
            if first_classes is None:
                first_classes = np.asarray(clf.classes_)
            elif not np.array_equal(clf.classes_, first_classes):
                raise ValueError(
                    f"Ensemble member {i} produced classes_={clf.classes_!r}; "
                    f"expected {first_classes!r}. All members must see the same labels."
                )
        self.classes_ = first_classes
        self.classifiers_ = members
        return self

    def _fitted_members(self) -> list[Any]:
        """Return the members of :meth:`fit`, else the given ones when all are fitted."""
        if hasattr(self, "classifiers_"):
            return self.classifiers_
        if any(isinstance(clf, str) for clf in self.classifiers):
            raise NotFittedError(
                "This SpectralEnsemble is built from names: call fit first."
            )
        for clf in self.classifiers:
            check_is_fitted(clf)
        return list(self.classifiers)

    def __sklearn_is_fitted__(self) -> bool:
        try:
            self._fitted_members()
        except NotFittedError:
            return False
        return True

    def predict_proba(self, X: Any) -> np.ndarray:
        """Return the mean of member ``predict_proba`` outputs.

        Parameters
        ----------
        X : array-like or MaldiSet of shape (n_samples, n_bins)
            Spectra to score.

        Returns
        -------
        ndarray of shape (n_samples, n_classes)
            Mean of the member probabilities.

        Raises
        ------
        ValueError
            If the members do not have the same ``classes_``.
        """
        members = self._fitted_members()
        first_classes = np.asarray(members[0].classes_)
        for i, clf in enumerate(members[1:], start=1):
            if not np.array_equal(clf.classes_, first_classes):
                raise ValueError(
                    f"Ensemble member {i} has classes_={clf.classes_!r}; expected "
                    f"{first_classes!r}. All members must see the same labels."
                )
        probas = [clf.predict_proba(X) for clf in members]
        stacked = np.stack(probas, axis=0)
        return stacked.mean(axis=0)

    def predict(self, X: Any) -> np.ndarray:
        """Argmax of the averaged probabilities.

        Per-member post-hoc calibration / thresholds are intentionally
        not averaged.
        """
        proba = self.predict_proba(X)
        idx = np.argmax(proba, axis=1)
        classes = getattr(self, "classes_", None)
        if classes is None:
            classes = self._fitted_members()[0].classes_
        return np.asarray(classes)[idx]

    def score(self, X: Any, y: Any) -> float:
        """Mean accuracy against ``y``."""
        preds = self.predict(X)
        if hasattr(y, "to_numpy"):
            y = y.to_numpy()
        return float(np.mean(preds == np.asarray(y).ravel()))

    def save(self, path: str | Path) -> None:
        """Save each member under ``<path>_<i>``.

        Example: ``SpectralEnsemble.save("my_ens")`` writes
        ``my_ens_0.pt`` / ``my_ens_0.json`` / ... plus an index file
        ``my_ens.ensemble.json`` recording the per-member classes.
        File names follow :meth:`~maldideepkit.BaseSpectralClassifier.save`:
        a single ``.pt``, ``.pth``, ``.json``, or ``.warper.pkl`` suffix
        on ``path`` is stripped and any other dot is kept, so
        ``save("ens0.05")`` writes ``ens0.05.ensemble.json`` and
        ``ens0.05_0.pt`` (distinct from the files of ``"ens0.10"``).

        Every file is first written to a temporary directory next to the
        target and renamed at the end (the index last), so a failing
        member leaves an existing ensemble untouched.

        Parameters
        ----------
        path : str or Path
            Base path without extension.
        """
        import json
        import os
        import tempfile

        from ..base.classifier import _checkpoint_base, _checkpoint_file

        base = _checkpoint_base(path)
        base.parent.mkdir(parents=True, exist_ok=True)
        members = self._fitted_members()
        with tempfile.TemporaryDirectory(
            dir=base.parent, prefix=f".{base.name}."
        ) as tmp_dir:
            tmp_base = Path(tmp_dir) / base.name
            member_paths = []
            for i, clf in enumerate(members):
                member_path = _checkpoint_file(tmp_base, f"_{i}")
                clf.save(member_path)
                member_paths.append(member_path.name)
            index_name = _checkpoint_file(base, ".ensemble.json").name
            (Path(tmp_dir) / index_name).write_text(
                json.dumps(
                    {
                        "version": _INDEX_VERSION,
                        "n_members": len(members),
                        "member_files": member_paths,
                        "classes_": (
                            np.asarray(self.classes_).tolist()
                            if hasattr(self, "classes_") and self.classes_ is not None
                            else None
                        ),
                        "member_class_names": [type(c).__name__ for c in members],
                    },
                    indent=2,
                )
            )
            written = sorted(p.name for p in Path(tmp_dir).iterdir())
            for name in [*(n for n in written if n != index_name), index_name]:
                os.replace(Path(tmp_dir) / name, base.parent / name)
            for name in member_paths:
                if f"{name}.warper.pkl" not in written:
                    (base.parent / f"{name}.warper.pkl").unlink(missing_ok=True)

    @classmethod
    def load(cls, path: str | Path, *, trusted: bool = False) -> SpectralEnsemble:
        """Inverse of :meth:`save`.

        Ensembles written by v0.3.0 under a dotted base name
        (``save("ens0.05")`` wrote ``ens0.ensemble.json`` there) still
        load: when ``<path>.ensemble.json`` does not exist but the
        v0.3.0 index name does and that index was written before
        v0.4.0 (index ``"version"`` 1), the latter is used and a
        ``UserWarning`` naming it is emitted. A v0.4.0 index under the
        v0.3.0 name is another ensemble, so the requested name raises
        :class:`FileNotFoundError`. Every member is loaded with
        :meth:`~maldideepkit.BaseSpectralClassifier.load`.

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
        SpectralEnsemble
            The ensemble of fitted members.

        Raises
        ------
        ValueError
            If a member checkpoint names a class or a function that the
            ``trusted`` rule refuses or that cannot be imported.
        FileNotFoundError
            If the index or a file of a member checkpoint is missing.

        Notes
        -----
        A ``.warper.pkl`` is a pickle, read whatever ``trusted`` is:
        load checkpoints with a warper only from trusted sources.
        """
        import json
        import warnings

        from ..base.classifier import (
            BaseSpectralClassifier,
            _checkpoint_base,
            _checkpoint_file,
        )

        index_path = _checkpoint_file(_checkpoint_base(path), ".ensemble.json")
        legacy_base = Path(path)
        if legacy_base.suffix:
            legacy_base = legacy_base.with_suffix("")
        legacy_index = legacy_base.with_suffix(".ensemble.json")
        meta: dict[str, Any] | None = None
        if not index_path.exists() and legacy_index.exists():
            with open(legacy_index) as fh:
                meta = json.load(fh)
            if int(meta.get("version", 1)) >= _INDEX_VERSION:
                raise FileNotFoundError(index_path)
            warnings.warn(
                f"{index_path} does not exist; loading the v0.3.0 ensemble "
                f"{legacy_index} instead (v0.3.0 dropped the last dotted part "
                "of the name). Save it again to store it under the requested "
                "name.",
                UserWarning,
                stacklevel=2,
            )
            index_path = legacy_index
        if not index_path.exists():
            raise FileNotFoundError(index_path)
        if meta is None:
            with open(index_path) as fh:
                meta = json.load(fh)
        members: list[Any] = []
        for name in meta["member_files"]:
            member_path = index_path.parent / name
            members.append(BaseSpectralClassifier.load(member_path, trusted=trusted))
        ens = cls(members)
        ens.classifiers_ = members
        if meta.get("classes_") is not None:
            ens.classes_ = np.asarray(meta["classes_"])
        return ens
