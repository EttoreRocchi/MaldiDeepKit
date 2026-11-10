"""Dataset and DataLoader helpers for MALDI-TOF spectra."""

from __future__ import annotations

import re
import warnings
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

_STD_FLOOR = 1e-7
# Rows per ``warper.transform`` call in ``_warp_numpy``.
_WARP_CHUNK_ROWS = 2048
# The per-call warning of :class:`maldiamrkit.alignment.Warping` counting
# the rows whose warp was non-monotonic (merged across chunks).
_NON_MONOTONIC_WARNING = re.compile(r"non-monotonic m/z mappings for (\d+)/(\d+)")


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
        raise ValueError(f"Expected a 2-D feature matrix; got shape {X.shape}.")
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
    y : array-like or None, default=None
        Integer class labels of shape ``(n_samples,)``. When ``None``
        (inference usage) the dataset yields only features.
    standardize : bool, default=False
        If ``True``, subtract the per-column mean and divide by the
        per-column standard deviation computed from ``X``. Columns with
        zero variance are left untouched.
    mean : array-like or None, default=None
        Pre-computed per-feature means. Used together with ``std`` to
        apply an external standardization (e.g. one fitted on a training
        fold). Ignored when ``standardize=False``.
    std : array-like or None, default=None
        Pre-computed per-feature standard deviations. Ignored when
        ``standardize=False``.
    sample_weight : array-like or None, default=None
        Per-sample weights of shape ``(n_samples,)``. When given, the
        dataset yields ``(x, y, w)`` triples; otherwise it yields the
        usual 2-tuples. Requires ``y`` to be set.

    Attributes
    ----------
    X : torch.Tensor
        Stored features as a float32 tensor.
    y : torch.Tensor or None
        Stored labels as a long tensor, or ``None`` for inference.
    sample_weight : torch.Tensor or None
        Stored per-sample weights as a float32 tensor, or ``None``.
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
        sample_weight: np.ndarray | None = None,
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

        if sample_weight is not None:
            if y is None:
                raise ValueError(
                    "sample_weight requires y to be provided; "
                    "weights have no meaning at inference time."
                )
            sw_np = np.asarray(sample_weight, dtype=np.float32).ravel()
            if sw_np.shape[0] != self.X.shape[0]:
                raise ValueError(
                    f"sample_weight has {sw_np.shape[0]} entries "
                    f"but X has {self.X.shape[0]} rows."
                )
            self.sample_weight: torch.Tensor | None = torch.from_numpy(sw_np)
        else:
            self.sample_weight = None

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(
        self, idx: int
    ) -> (
        torch.Tensor
        | tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        if self.y is None:
            return self.X[idx]
        if self.sample_weight is None:
            return self.X[idx], self.y[idx]
        return self.X[idx], self.y[idx], self.sample_weight[idx]


INPUT_TRANSFORMS = ("standardize", "log1p", "robust", "log1p+standardize", "none")


def resolve_transform_mode(standardize: bool, input_transform: str | None) -> str:
    """Return the effective input-transform mode of a preprocessing config.

    ``input_transform`` wins whenever it is given; otherwise the legacy
    ``standardize`` flag selects ``"standardize"`` (when ``True``) or
    ``"none"`` (when ``False``). Two configs that resolve to the same
    mode (e.g. ``standardize=True`` and
    ``input_transform="standardize"``) preprocess identically.

    Parameters
    ----------
    standardize : bool
        Legacy standardization flag.
    input_transform : str or None
        Explicit input-transform mode (one of :data:`INPUT_TRANSFORMS`).

    Returns
    -------
    str
        The resolved mode. It is not validated here; unknown modes are
        rejected by :func:`fit_input_transform`.
    """
    if input_transform is None:
        return "standardize" if standardize else "none"
    return input_transform


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

    Parameters
    ----------
    X_tr : ndarray of shape (n_samples, n_bins)
        Training rows.
    mode : str
        One of :data:`INPUT_TRANSFORMS`.

    Returns
    -------
    dict
        ``{"mode": mode}`` plus the per-bin statistics of the mode
        (``"mean"`` and ``"std"``, or ``"median"`` and ``"iqr"``).

    Raises
    ------
    ValueError
        If ``mode`` is unknown.
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
    """Apply a fitted :func:`fit_input_transform` state to ``X``.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_bins)
        Rows to transform.
    state : dict
        State returned by :func:`fit_input_transform`.

    Returns
    -------
    ndarray of shape (n_samples, n_bins)
        The transformed rows as ``float32``.

    Raises
    ------
    ValueError
        If the mode of ``state`` is unknown.
    """
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


def apply_input_transform_torch(X: torch.Tensor, state: dict[str, Any]) -> torch.Tensor:
    """PyTorch twin of :func:`apply_input_transform`, run on the device of ``X``.

    Same element-wise ``float32`` operations as the NumPy version:
    ``"none"``, ``"standardize"``, and ``"robust"`` give the same values,
    and the ``log1p`` modes may differ in the last bit of ``log1p``
    (relative ``1e-7``, so within ``1e-6`` on spectra of typical scale;
    ``"log1p+standardize"`` then divides that difference by the per-bin
    standard deviation).

    Parameters
    ----------
    X : torch.Tensor of shape (n_samples, n_bins)
        Raw spectra on any device (cast to ``float32``).
    state : dict
        State from :func:`fit_input_transform`. Its statistics may be
        NumPy arrays or tensors; pre-converting them to tensors on the
        device of ``X`` avoids a host-to-device copy per call.

    Returns
    -------
    torch.Tensor
        ``float32`` tensor on the device of ``X``. Mode ``"none"``
        returns ``X`` itself when it is already ``float32``.
    """
    mode = state.get("mode", "none")
    X = X.float()

    def stat(name: str) -> torch.Tensor:
        return torch.as_tensor(state[name], dtype=torch.float32, device=X.device)

    if mode == "none":
        return X
    if mode == "log1p":
        return torch.log1p(X.clamp_min(0))
    if mode == "standardize":
        return (X - stat("mean")) / stat("std").clamp_min(_STD_FLOOR)
    if mode == "log1p+standardize":
        logged = torch.log1p(X.clamp_min(0))
        return (logged - stat("mean")) / stat("std").clamp_min(_STD_FLOOR)
    if mode == "robust":
        return (X - stat("median")) / stat("iqr")
    raise ValueError(f"Unknown input_transform state mode={mode!r}.")


def _transform_leading_cols(
    X: np.ndarray,
    n_cols: int | None,
    fn: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """Apply ``fn`` to the first ``n_cols`` columns of ``X`` only.

    Dual-input classifiers pack a per-row side feature into ``X`` by
    column concatenation (``[ X_spec | X_side ]``). Spectrum-specific
    preprocessing (warping, ``log1p``, standardization) is
    meaningless on the side block (an antibiotic fingerprint is not a
    spectrum), so it is confined to the leading ``n_cols`` columns and
    the trailing block is passed through untouched.

    ``n_cols=None``, or a value covering every column, applies ``fn``
    to the whole matrix (the single-input case).

    Raises
    ------
    ValueError
        If ``fn`` changes the column count of the leading block, which
        would silently misalign the spectrum / side split point.
    """
    if n_cols is None or int(n_cols) >= X.shape[1]:
        return np.asarray(fn(X), dtype=np.float32)
    n = int(n_cols)
    head = np.asarray(fn(X[:, :n]), dtype=np.float32)
    if head.shape[1] != n:
        raise ValueError(
            f"A preprocessing step changed the spectrum block width from "
            f"{n} to {head.shape[1]} columns. Dual-input models need the "
            "split point to stay fixed; re-bin the spectra before fit() "
            "instead, or drop the width-changing transform."
        )
    tail = X[:, n:].astype(np.float32, copy=False)
    return np.hstack([head, tail]).astype(np.float32, copy=False)


def _warp_rows(warper: Any, X_np: np.ndarray, start: int = 0) -> np.ndarray:
    """Run ``warper.transform`` on ``X_np`` (rows indexed from ``start``)."""
    index = pd.RangeIndex(start, start + X_np.shape[0])
    out = warper.transform(pd.DataFrame(X_np, index=index))
    if hasattr(out, "to_numpy"):
        out = out.to_numpy()
    return np.asarray(out, dtype=np.float32)


def _warp_numpy(warper: Any, X_np: np.ndarray) -> np.ndarray:
    """Apply a fitted warper to a numpy matrix, returning a numpy matrix.

    The warper is applied in blocks of ``_WARP_CHUNK_ROWS`` rows
    (each block keeps its global row index), which bounds the peak
    memory of :class:`maldiamrkit.alignment.Warping` (several times the
    size of its input) on large matrices. Warpers align every row
    independently to their fitted reference, so the result equals a
    single ``transform`` call on the whole matrix. Its warnings are
    caught per block: the "non-monotonic m/z mappings for k/n
    sample(s)" warnings of the blocks are merged into one warning with
    the summed count over the whole matrix, any other warning is
    re-emitted as it is.
    """
    n = X_np.shape[0]
    chunk = _WARP_CHUNK_ROWS
    if n <= chunk:
        return _warp_rows(warper, X_np)
    out: np.ndarray | None = None
    n_bad = 0
    merged: tuple[warnings.WarningMessage, re.Match[str]] | None = None
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            rows = _warp_rows(warper, X_np[start:stop], start)
        if out is None:
            out = np.empty((n, rows.shape[1]), dtype=np.float32)
        out[start:stop] = rows
        for w in caught:
            match = _NON_MONOTONIC_WARNING.search(str(w.message))
            if match is None:
                warnings.warn_explicit(
                    w.message, w.category, w.filename, w.lineno, source=w.source
                )
            else:
                n_bad += int(match.group(1))
                merged = (w, match)
    if merged is not None:
        w, match = merged
        text = str(w.message)
        warnings.warn(
            f"{text[: match.start(1)]}{n_bad}/{n}{text[match.end(2) :]}",
            w.category,
            stacklevel=2,
        )
    assert out is not None
    return out


def apply_fitted_preprocessing(
    X: Any,
    *,
    warper: Any | None = None,
    input_transform_state: dict[str, Any] | None = None,
    standardize: bool = False,
    feature_mean: np.ndarray | None = None,
    feature_std: np.ndarray | None = None,
    n_transform_cols: int | None = None,
) -> np.ndarray:
    """Apply a fitted inference-time preprocessing chain to ``X``.

    Runs, in order, the optional fitted spectral ``warper``, then the
    fitted ``input_transform_state`` (see :func:`fit_input_transform`)
    or, when that state is absent, the legacy ``standardize`` path
    with ``feature_mean`` / ``feature_std``. Every step is confined to
    the leading ``n_transform_cols`` columns (dual-input models never
    transform their packed side block). Shared by the classifiers and
    :class:`~maldideepkit.pretraining.MaldiSpectrumEncoder`, so both
    preprocess new data identically.

    Parameters
    ----------
    X : array-like or MaldiSet of shape (n_samples, n_bins)
        Raw input spectra.
    warper : fitted sklearn-style transformer or None, default=None
        Fitted spectral-alignment transformer.
    input_transform_state : dict or None, default=None
        State returned by :func:`fit_input_transform`. Mode ``"none"``
        is a no-op.
    standardize : bool, default=False
        Legacy flag; only used when ``input_transform_state`` is
        ``None`` or has mode ``"none"``.
    feature_mean, feature_std : ndarray or None, default=None
        Legacy standardization statistics, used together with
        ``standardize``.
    n_transform_cols : int or None, default=None
        Number of leading spectrum columns to transform. ``None``
        transforms every column.

    Returns
    -------
    ndarray of shape (n_samples, n_bins)
        Preprocessed feature matrix as ``float32``.
    """
    X_np = _to_numpy(X)
    n_tc = n_transform_cols
    if warper is not None:
        X_np = _transform_leading_cols(X_np, n_tc, lambda A: _warp_numpy(warper, A))
    state = input_transform_state
    if state is not None and state.get("mode", "none") != "none":
        X_np = _transform_leading_cols(
            X_np, n_tc, lambda A: apply_input_transform(A, state)
        )
    elif standardize and feature_mean is not None:
        safe_std = np.maximum(feature_std, _STD_FLOOR).astype(np.float32)
        X_np = _transform_leading_cols(
            X_np, n_tc, lambda A: (A - feature_mean) / safe_std
        )
    return X_np.astype(np.float32, copy=False)


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
    sample_weight: np.ndarray | None = None,
    n_transform_cols: int | None = None,
    fitted_preprocessing: dict[str, Any] | None = None,
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
    input_transform : str or None, default=None
        One of ``{"none", "standardize", "log1p", "robust",
        "log1p+standardize"}``. Fitted on the (warped) training split
        only and applied to both splits. Overrides ``standardize``
        when both are given.
    stratify : bool, default=True
        If ``True`` and all classes have at least two samples, stratify
        the split on ``y``. Falls back to random split otherwise.
    num_workers : int, default=0
        ``DataLoader`` worker count.
    warper : sklearn-style transformer or None, default=None
        Unfitted spectral-alignment transformer with ``fit(X) ->
        self`` + ``transform(X) -> X``. Fitted on the training split
        only and used to transform both splits. The fitted object is
        returned in ``stats["warper"]``.
    sample_weight : ndarray of shape (n_samples,) or None, default=None
        Per-sample weights. When given, they are split alongside ``X``
        and ``y`` via the same stratified train/val split, attached to
        the training :class:`SpectralDataset` (so batches yield
        ``(x, y, w)`` triples), and returned in ``stats["sample_weight_val"]``
        for downstream val-side weighting.
    n_transform_cols : int or None, default=None
        Confine ``warper``, ``standardize``, and ``input_transform`` to
        the first ``n_transform_cols`` columns of ``X``, passing the
        rest through untouched. Used by dual-input classifiers, whose
        ``X`` is ``[ X_spec | X_side ]`` and whose side block must not
        be log-transformed, warped, or standardized as if it were a
        spectrum. ``None`` transforms every column. The
        returned ``stats["mean"]`` / ``stats["std"]`` then cover only
        the leading block.
    fitted_preprocessing : dict or None, default=None
        Already-fitted preprocessing to reuse instead of fitting it on
        the training split: ``{"warper": fitted warper or None,
        "input_transform_state": dict}`` (e.g. the state of a fitted
        :class:`~maldideepkit.pretraining.MaldiSpectrumEncoder`). When
        given, ``warper.fit`` and :func:`fit_input_transform` are
        skipped, the provided objects are applied to both splits and
        echoed in ``stats``; ``warper``, ``standardize``, and
        ``input_transform`` are then ignored.

    Returns
    -------
    train_loader : DataLoader
        Shuffling training loader. Drops the last batch when it would
        contain a single sample (avoids ``BatchNorm`` issues).
    val_loader : DataLoader
        Non-shuffling validation loader.
    stats : dict
        ``{"mean": array or None, "std": array or None, "warper":
        fitted warper or None, "input_transform_state": dict,
        "sample_weight_val": ndarray or None}``.
    """
    X_np = _to_numpy(X)
    if hasattr(y, "to_numpy"):
        y = y.to_numpy()
    y_np = np.asarray(y).ravel()

    _, counts = np.unique(y_np, return_counts=True)
    can_stratify = stratify and counts.min() >= 2

    if sample_weight is None:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_np,
            y_np,
            test_size=val_size,
            random_state=random_state,
            stratify=y_np if can_stratify else None,
        )
        sw_tr: np.ndarray | None = None
        sw_val: np.ndarray | None = None
    else:
        sw_np = np.asarray(sample_weight, dtype=np.float32).ravel()
        if sw_np.shape[0] != X_np.shape[0]:
            raise ValueError(
                f"sample_weight has {sw_np.shape[0]} entries "
                f"but X has {X_np.shape[0]} rows."
            )
        X_tr, X_val, y_tr, y_val, sw_tr, sw_val = train_test_split(
            X_np,
            y_np,
            sw_np,
            test_size=val_size,
            random_state=random_state,
            stratify=y_np if can_stratify else None,
        )

    n_tc = None if n_transform_cols is None else int(n_transform_cols)
    fit_slice = (lambda A: A) if n_tc is None else (lambda A: A[:, :n_tc])

    if fitted_preprocessing is None:
        fitted_warper = None
        if warper is not None:
            fitted_warper = warper.fit(pd.DataFrame(fit_slice(X_tr)))
    else:
        if "input_transform_state" not in fitted_preprocessing:
            raise ValueError(
                "fitted_preprocessing must contain an 'input_transform_state' "
                f"entry; got keys {sorted(fitted_preprocessing)}."
            )
        fitted_warper = fitted_preprocessing.get("warper")
    if fitted_warper is not None:
        X_tr = _transform_leading_cols(
            X_tr, n_tc, lambda A: _warp_numpy(fitted_warper, A)
        )
        X_val = _transform_leading_cols(
            X_val, n_tc, lambda A: _warp_numpy(fitted_warper, A)
        )

    if fitted_preprocessing is None:
        transform_mode = resolve_transform_mode(standardize, input_transform)
        transform_state = fit_input_transform(fit_slice(X_tr), transform_mode)
    else:
        transform_state = fitted_preprocessing["input_transform_state"]
    X_tr = _transform_leading_cols(
        X_tr, n_tc, lambda A: apply_input_transform(A, transform_state)
    )
    X_val = _transform_leading_cols(
        X_val, n_tc, lambda A: apply_input_transform(A, transform_state)
    )

    mean = transform_state.get("mean")
    std = transform_state.get("std")

    train_ds = SpectralDataset(X_tr, y_tr, standardize=False, sample_weight=sw_tr)
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
            "sample_weight_val": sw_val,
        },
    )
