"""Dataset and DataLoader helpers for MALDI-TOF spectra."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

_STD_FLOOR = 1e-7


def _to_numpy(X: Any) -> np.ndarray:
    """Return ``X`` as a float32 2-D ndarray.

    Accepts NumPy arrays, pandas DataFrames/Series, and any object
    exposing a DataFrame-like ``.X`` attribute (e.g. :class:`maldiamrkit.MaldiSet`).
    """
    if hasattr(X, "X") and not isinstance(X, np.ndarray):
        X = X.X
    if hasattr(X, "to_numpy"):
        X = X.to_numpy()
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if X.ndim != 2:
        raise ValueError(f"Expected 2-D feature matrix, got shape {X.shape} instead.")
    return X


class SpectralDataset(Dataset):
    """PyTorch ``Dataset`` wrapping a binned MALDI-TOF feature matrix.

    The dataset stores its spectra as a single float32 tensor in memory
    and optionally standardizes each feature on the fly using
    statistics computed once at construction time.

    Parameters
    ----------
    X : array-like or MaldiSet
        Feature matrix of shape ``(n_samples, n_bins)``. A NumPy array,
        a pandas DataFrame, or any object with a DataFrame-like ``.X``
        attribute is accepted.
    y : array-like, optional
        Integer class labels of shape ``(n_samples,)``. When ``None``
        (inference usage) the dataset yields only features.
    standardize : bool, default=False
        If ``True``, subtract the per-column mean and divide by the
        per-column standard deviation computed from ``X``. Columns with
        zero variance are left untouched.
    mean : array-like, optional
        Pre-computed per-feature means. Used together with ``std`` to
        apply an external standardization (e.g. one fitted on a training
        fold). Ignored when ``standardize=False``.
    std : array-like, optional
        Pre-computed per-feature standard deviations. Ignored when
        ``standardize=False``.

    Attributes
    ----------
    X : torch.Tensor
        Stored features as a float32 tensor.
    y : torch.Tensor or None
        Stored labels as a long tensor, or ``None`` for inference.
    mean : torch.Tensor or None
        Feature-wise mean used for standardization.
    std : torch.Tensor or None
        Feature-wise standard deviation used for standardization.
    """

    def __init__(
        self,
        X: Any,
        y: Any | None = None,
        *,
        standardize: bool = False,
        mean: np.ndarray | None = None,
        std: np.ndarray | None = None,
    ) -> None:
        X_np = _to_numpy(X)

        if standardize:
            if mean is None or std is None:
                mean = X_np.mean(axis=0)
                std = X_np.std(axis=0)
            std = np.asarray(std, dtype=np.float32)
            mean = np.asarray(mean, dtype=np.float32)
            safe_std = np.maximum(std, _STD_FLOOR).astype(np.float32)
            X_np = (X_np - mean) / safe_std
            self.mean: torch.Tensor | None = torch.from_numpy(
                np.asarray(mean, dtype=np.float32)
            )
            self.std: torch.Tensor | None = torch.from_numpy(safe_std)
        else:
            self.mean = None
            self.std = None

        self.X = torch.from_numpy(X_np)

        if y is not None:
            if hasattr(y, "to_numpy"):
                y = y.to_numpy()
            y_np = np.asarray(y).ravel()
            self.y: torch.Tensor | None = torch.from_numpy(y_np.astype(np.int64))
        else:
            self.y = None

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self.y is None:
            return self.X[idx]
        return self.X[idx], self.y[idx]


INPUT_TRANSFORMS = ("standardize", "log1p", "robust", "log1p+standardize", "none")


def fit_input_transform(X_tr: np.ndarray, mode: str) -> dict[str, Any]:
    """Compute per-bin statistics for the requested input-transform mode.

    Fitted on the training split only to keep the pipeline leak-safe.
    Returns a state dict that :func:`apply_input_transform` consumes
    at training and inference time.

    Supported modes:

    - ``"none"``: identity; empty state.
    - ``"standardize"``: per-bin ``(X - mean) / std`` from the train split.
    - ``"log1p"``: element-wise ``log1p(clip(X, 0, None))``; stateless.
    - ``"robust"``: per-bin ``(X - median) / IQR`` from the train split.
      A zero IQR bin is treated as unit scale.
    - ``"log1p+standardize"``: ``log1p`` first, then standardize.
    """
    if mode not in INPUT_TRANSFORMS:
        raise ValueError(
            f"Unknown input_transform={mode!r}; expected one of {INPUT_TRANSFORMS}."
        )
    state: dict[str, Any] = {"mode": mode}
    if mode == "log1p+standardize":
        X_fit = np.log1p(np.clip(X_tr, 0, None))
    else:
        X_fit = X_tr

    if mode in {"standardize", "log1p+standardize"}:
        state["mean"] = X_fit.mean(axis=0).astype(np.float32)
        state["std"] = X_fit.std(axis=0).astype(np.float32)
    elif mode == "robust":
        state["median"] = np.median(X_fit, axis=0).astype(np.float32)
        q75, q25 = np.percentile(X_fit, [75, 25], axis=0)
        iqr = (q75 - q25).astype(np.float32)
        state["iqr"] = np.maximum(iqr, _STD_FLOOR).astype(np.float32)
    return state


def apply_input_transform(X: np.ndarray, state: dict[str, Any]) -> np.ndarray:
    """Apply a fitted :func:`fit_input_transform` state to ``X``."""
    mode = state.get("mode", "none")
    if mode == "none":
        return X.astype(np.float32, copy=False)
    if mode == "log1p":
        return np.log1p(np.clip(X, 0, None)).astype(np.float32, copy=False)
    if mode == "standardize":
        mean = np.asarray(state["mean"], dtype=np.float32)
        std = np.asarray(state["std"], dtype=np.float32)
        safe_std = np.maximum(std, _STD_FLOOR).astype(np.float32)
        return ((X - mean) / safe_std).astype(np.float32, copy=False)
    if mode == "log1p+standardize":
        logged = np.log1p(np.clip(X, 0, None))
        mean = np.asarray(state["mean"], dtype=np.float32)
        std = np.asarray(state["std"], dtype=np.float32)
        safe_std = np.maximum(std, _STD_FLOOR).astype(np.float32)
        return ((logged - mean) / safe_std).astype(np.float32, copy=False)
    if mode == "robust":
        median = np.asarray(state["median"], dtype=np.float32)
        iqr = np.asarray(state["iqr"], dtype=np.float32)
        return ((X - median) / iqr).astype(np.float32, copy=False)
    raise ValueError(f"Unknown input_transform state mode={mode!r}.")


def _warp_numpy(warper: Any, X_np: np.ndarray) -> np.ndarray:
    """Apply a fitted warper to a numpy matrix, returning a numpy matrix."""
    df = pd.DataFrame(X_np)
    out = warper.transform(df)
    if hasattr(out, "to_numpy"):
        out = out.to_numpy()
    return np.asarray(out, dtype=np.float32)


def make_loaders(
    X: Any,
    y: Any,
    *,
    batch_size: int = 32,
    val_size: float = 0.1,
    random_state: int | None = 0,
    standardize: bool = False,
    input_transform: str | None = None,
    stratify: bool = True,
    num_workers: int = 0,
    warper: Any | None = None,
) -> tuple[DataLoader, DataLoader, dict[str, Any]]:
    """Build stratified train / validation :class:`DataLoader` pairs.

    Pipeline order, applied **after** the train/val split so nothing
    from the validation split leaks into training statistics:

    1. Spectral warping / alignment (if ``warper`` is given): fit on
       the training split, then transform both splits.
    2. Per-feature standardization (if ``standardize=True``): fit
       mean/std on the (warped) training split, then apply to both
       splits.

    Parameters
    ----------
    X : array-like or MaldiSet
        Feature matrix of shape ``(n_samples, n_bins)``.
    y : array-like
        Integer class labels of shape ``(n_samples,)``.
    batch_size : int, default=32
        Mini-batch size for the training loader.
    val_size : float, default=0.1
        Fraction of the input held out for validation.
    random_state : int or None, default=0
        Seed for the split.
    standardize : bool, default=False
        Shorthand for ``input_transform="standardize"`` (when True) or
        ``input_transform="none"`` (when False). Kept for backwards
        compatibility; the modern interface is ``input_transform``.
        Ignored whenever ``input_transform`` is given explicitly.
    input_transform : str, optional
        One of ``{"none", "standardize", "log1p", "robust",
        "log1p+standardize"}``. Fitted on the (warped) training split
        only and applied to both splits. Overrides ``standardize``
        when both are given.
    stratify : bool, default=True
        If ``True`` and all classes have at least two samples, stratify
        the split on ``y``. Falls back to random split otherwise.
    num_workers : int, default=0
        ``DataLoader`` worker count.
    warper : sklearn-style transformer, optional
        Unfitted spectral-alignment transformer with ``fit(X) ->
        self`` + ``transform(X) -> X``. Fitted on the training split
        only and used to transform both splits. The fitted object is
        returned in ``stats["warper"]``.

    Returns
    -------
    train_loader : DataLoader
        Shuffling training loader. Drops the last batch when it would
        contain a single sample (avoids ``BatchNorm`` issues).
    val_loader : DataLoader
        Non-shuffling validation loader.
    stats : dict
        ``{"mean": array or None, "std": array or None, "warper":
        fitted warper or None, "input_transform_state": dict}``.
    """
    X_np = _to_numpy(X)
    if hasattr(y, "to_numpy"):
        y = y.to_numpy()
    y_np = np.asarray(y).ravel()

    _, counts = np.unique(y_np, return_counts=True)
    can_stratify = stratify and counts.min() >= 2

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_np,
        y_np,
        test_size=val_size,
        random_state=random_state,
        stratify=y_np if can_stratify else None,
    )

    fitted_warper = None
    if warper is not None:
        fitted_warper = warper.fit(pd.DataFrame(X_tr))
        X_tr = _warp_numpy(fitted_warper, X_tr)
        X_val = _warp_numpy(fitted_warper, X_val)

    if input_transform is None:
        transform_mode = "standardize" if standardize else "none"
    else:
        transform_mode = input_transform
    transform_state = fit_input_transform(X_tr, transform_mode)
    X_tr = apply_input_transform(X_tr, transform_state)
    X_val = apply_input_transform(X_val, transform_state)

    mean = transform_state.get("mean")
    std = transform_state.get("std")

    train_ds = SpectralDataset(X_tr, y_tr, standardize=False)
    val_ds = SpectralDataset(X_val, y_val, standardize=False)

    drop_last = batch_size > 1 and (len(train_ds) % batch_size) == 1
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )
    return (
        train_loader,
        val_loader,
        {
            "mean": mean,
            "std": std,
            "warper": fitted_warper,
            "input_transform_state": transform_state,
        },
    )
