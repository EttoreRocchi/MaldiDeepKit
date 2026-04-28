"""Simple mean-of-``predict_proba`` ensemble for spectral classifiers.

A thin wrapper that fits each member independently and averages
``predict_proba`` across them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class SpectralEnsemble:
    """Ensemble N fitted or unfitted spectral classifiers.

    Parameters
    ----------
    classifiers : sequence of BaseSpectralClassifier
        Unfitted classifier instances. :meth:`fit` calls each
        member's own ``fit`` in order.

    Attributes
    ----------
    classes_ : np.ndarray
        Union of class labels reported by the members. Members must
        agree on the label set after fitting.
    """

    def __init__(self, classifiers: list[Any]) -> None:
        if not classifiers:
            raise ValueError("SpectralEnsemble needs at least one classifier.")
        self.classifiers = list(classifiers)

    def fit(self, X: Any, y: Any) -> SpectralEnsemble:
        """Fit every member on the same ``(X, y)``."""
        first_classes: np.ndarray | None = None
        for i, clf in enumerate(self.classifiers):
            clf.fit(X, y)
            if first_classes is None:
                first_classes = np.asarray(clf.classes_)
            elif not np.array_equal(clf.classes_, first_classes):
                raise ValueError(
                    f"Ensemble member {i} produced classes_={clf.classes_!r}; "
                    f"expected {first_classes!r}. All members must see the same labels."
                )
        self.classes_ = first_classes
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        """Return the mean of member ``predict_proba`` outputs."""
        probas = [clf.predict_proba(X) for clf in self.classifiers]
        stacked = np.stack(probas, axis=0)
        return stacked.mean(axis=0)

    def predict(self, X: Any) -> np.ndarray:
        """Argmax of the averaged probabilities.

        Per-member post-hoc calibration / thresholds are intentionally
        not averaged.
        """
        proba = self.predict_proba(X)
        idx = np.argmax(proba, axis=1)
        return np.asarray(self.classes_)[idx]

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
        """
        import json

        base = Path(path)
        if base.suffix:
            base = base.with_suffix("")
        base.parent.mkdir(parents=True, exist_ok=True)
        member_paths = []
        for i, clf in enumerate(self.classifiers):
            member_path = base.parent / f"{base.name}_{i}"
            clf.save(member_path)
            member_paths.append(str(member_path.name))
        index_path = base.with_suffix(".ensemble.json")
        with open(index_path, "w") as fh:
            json.dump(
                {
                    "version": 1,
                    "n_members": len(self.classifiers),
                    "member_files": member_paths,
                    "classes_": (
                        np.asarray(self.classes_).tolist()
                        if hasattr(self, "classes_") and self.classes_ is not None
                        else None
                    ),
                    "member_class_names": [type(c).__name__ for c in self.classifiers],
                },
                fh,
                indent=2,
            )

    @classmethod
    def load(cls, path: str | Path) -> SpectralEnsemble:
        """Inverse of :meth:`save`."""
        import json

        from ..base.classifier import BaseSpectralClassifier

        base = Path(path)
        if base.suffix:
            base = base.with_suffix("")
        index_path = base.with_suffix(".ensemble.json")
        if not index_path.exists():
            raise FileNotFoundError(index_path)
        with open(index_path) as fh:
            meta = json.load(fh)
        members: list[Any] = []
        for name in meta["member_files"]:
            member_path = base.parent / name
            members.append(BaseSpectralClassifier.load(member_path))
        ens = cls(members)
        if meta.get("classes_") is not None:
            ens.classes_ = np.asarray(meta["classes_"])
        return ens
