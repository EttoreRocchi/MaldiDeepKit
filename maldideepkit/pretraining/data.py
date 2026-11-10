"""Unlabeled / partially labeled data loaders for self-supervised pre-training."""

from __future__ import annotations

import numbers
import warnings
from collections.abc import Sequence
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset

from ..base.data import (
    _to_numpy,
    _warp_numpy,
    apply_input_transform,
    fit_input_transform,
)

AUX_TARGETS = ("classification", "regression")

_RECON_STATS_MODES = ("log1p", "none")
_RECON_CHUNK_BYTES = 16 * 2**20


class _RowBatch(Sequence):
    """The items of one batch, read at once (``Dataset.__getitems__`` result).

    Behaves as the list of ``(x_raw, target, labeled_mask)`` items a
    custom ``collate_fn`` expects (item ``i`` is row ``i`` of the batched
    tensors), and lets :func:`stack_collate` return the batched tensors
    directly instead of stacking the items again.
    """

    def __init__(
        self, x_raw: torch.Tensor, target: torch.Tensor, labeled: torch.Tensor
    ) -> None:
        self.stacked = (x_raw, target, labeled)

    def __len__(self) -> int:
        return int(self.stacked[0].shape[0])

    def __getitem__(self, i: Any) -> Any:
        if isinstance(i, slice):
            return [self[j] for j in range(*i.indices(len(self)))]
        x_raw, target, labeled = self.stacked
        return x_raw[i], target[i], labeled[i]


class _PretrainDataset(Dataset):
    """Lazy view over a subset of rows of a (possibly memory-mapped) pool.

    Rows are read on demand from ``X`` (never materialised as a whole),
    so a memory-mapped ``.npy`` pool stays on disk; a ``DataLoader``
    batch is read with one indexing call (``__getitems__``). Each item
    is the **raw** (warped, not input-transformed) spectrum: the input
    transform is applied per collated batch by
    :func:`~maldideepkit.pretraining._training.ssl_train_loop`.

    Parameters
    ----------
    X : ndarray of shape (n_pool, n_bins)
        Float32 pool (``np.memmap`` accepted).
    indices : array-like of int
        Pool rows exposed by this dataset, in order.
    targets : torch.Tensor of shape (n_pool,)
        Auxiliary targets for every pool row (``int64`` class indices
        or ``float32`` values; undefined on unlabeled rows).
    labeled : torch.Tensor of shape (n_pool,)
        Boolean mask of pool rows carrying a target.

    Notes
    -----
    Items are ``(x_raw, target, labeled_mask)`` triples: a float32
    tensor of shape ``(n_bins,)`` and two 0-d tensors.
    """

    def __init__(
        self,
        X: np.ndarray,
        indices: np.ndarray,
        targets: torch.Tensor,
        labeled: torch.Tensor,
    ) -> None:
        self.X = X
        self.indices = np.asarray(indices, dtype=np.int64)
        self.targets = targets
        self.labeled = labeled

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        j = int(self.indices[i])
        x_raw = torch.from_numpy(np.array(self.X[j], dtype=np.float32))
        return x_raw, self.targets[j], self.labeled[j]

    def __getitems__(self, positions: list[int]) -> _RowBatch:
        rows = self.indices[np.asarray(positions, dtype=np.int64)]
        x_raw = torch.from_numpy(np.array(self.X[rows], dtype=np.float32))
        rows_t = torch.from_numpy(rows)
        return _RowBatch(x_raw, self.targets[rows_t], self.labeled[rows_t])


class _DevicePoolDataset(Dataset):
    """Subset of rows of a pool held in device memory.

    The device-resident counterpart of :class:`_PretrainDataset`: a
    batch is gathered on the device (``index_select``), so the loader
    yields device tensors and no spectrum crosses the host-device bus
    after the one-off upload. Only the row positions come from the
    ``DataLoader`` sampler, so the batch composition is the one of the
    host path for the same global RNG state.

    Parameters
    ----------
    pool : torch.Tensor of shape (n_pool, n_bins)
        Float32 pool on the device.
    indices : array-like of int
        Pool rows exposed by this dataset, in order.
    targets, labeled : torch.Tensor of shape (n_pool,)
        Auxiliary targets and labeled mask, on the device of ``pool``.
    """

    def __init__(
        self,
        pool: torch.Tensor,
        indices: np.ndarray,
        targets: torch.Tensor,
        labeled: torch.Tensor,
    ) -> None:
        self.pool = pool
        self.indices = np.asarray(indices, dtype=np.int64)
        self.rows = torch.from_numpy(self.indices).to(pool.device)
        self.targets = targets
        self.labeled = labeled

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        j = int(self.indices[i])
        return self.pool[j], self.targets[j], self.labeled[j]

    def __getitems__(self, positions: list[int]) -> _RowBatch:
        pos = torch.as_tensor(positions, dtype=torch.int64)
        rows = self.rows[pos.to(self.pool.device, non_blocking=True)]
        return _RowBatch(
            self.pool.index_select(0, rows), self.targets[rows], self.labeled[rows]
        )


# Rows per host-to-device copy when a pool is uploaded to the device.
_UPLOAD_CHUNK_ROWS = 4096


def _upload_pool(X: np.ndarray, device: torch.device) -> torch.Tensor:
    """Copy a (possibly memory-mapped) float32 pool to ``device`` in row chunks.

    The pool is never materialised as a whole in host memory: only one
    chunk of ``_UPLOAD_CHUNK_ROWS`` rows is read at a time.
    """
    n_rows = int(X.shape[0])
    out = torch.empty((n_rows, int(X.shape[1])), dtype=torch.float32, device=device)
    for start in range(0, n_rows, _UPLOAD_CHUNK_ROWS):
        stop = min(start + _UPLOAD_CHUNK_ROWS, n_rows)
        out[start:stop].copy_(
            torch.from_numpy(np.array(X[start:stop], dtype=np.float32))
        )
    return out


def _recon_stats(
    X: np.ndarray, idx_tr: np.ndarray, transform_state: dict[str, Any]
) -> dict[str, np.ndarray] | None:
    """Per-bin mean / std of the model input over the training rows.

    ``None`` unless the resolved mode is one of
    :data:`_RECON_STATS_MODES`. The rows are transformed in chunks of at
    most :data:`_RECON_CHUNK_BYTES` and summed in ``float64``, so a
    memory-mapped pool is never materialised.
    """
    if transform_state["mode"] not in _RECON_STATS_MODES or idx_tr.size == 0:
        return None
    n_bins = int(X.shape[1])
    step = max(1, _RECON_CHUNK_BYTES // (4 * n_bins))
    total = np.zeros(n_bins, dtype=np.float64)
    total_sq = np.zeros(n_bins, dtype=np.float64)
    for start in range(0, idx_tr.size, step):
        rows = np.asarray(X[idx_tr[start : start + step]], dtype=np.float32)
        x = apply_input_transform(rows, transform_state)
        total += x.sum(axis=0, dtype=np.float64)
        total_sq += np.square(x).sum(axis=0, dtype=np.float64)
    mean = total / idx_tr.size
    var = np.maximum(total_sq / idx_tr.size - np.square(mean), 0.0)
    return {
        "mean": mean.astype(np.float32),
        "std": np.sqrt(var).astype(np.float32),
    }


def stack_collate(
    batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stack pre-training samples into a batch (the default ``collate_fn``).

    Parameters
    ----------
    batch : sequence of (x_raw, target, labeled_mask)
        Items of a pre-training dataset (a batch read at once is
        returned as it is, already stacked).

    Returns
    -------
    tuple of torch.Tensor
        ``(x_raw, target, labeled_mask)`` of shapes
        ``(batch, n_bins)``, ``(batch,)``, and ``(batch,)``. The input
        transform is **not** applied here (see
        :func:`~maldideepkit.pretraining._training.ssl_train_loop`).
    """
    if isinstance(batch, _RowBatch):
        return batch.stacked
    xs, targets, labeled = zip(*batch, strict=True)
    return torch.stack(xs), torch.stack(targets), torch.stack(labeled)


def _is_missing_marker(marker: Any) -> bool:
    """Return ``True`` when ``marker`` is itself a missing value (NaN / None)."""
    try:
        return bool(pd.isna(marker))
    except (TypeError, ValueError):
        return False


def _encode_targets(
    y: Any | None,
    n_rows: int,
    *,
    aux_target: str | None,
    unlabeled_marker: Any,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray | None]:
    """Encode partial auxiliary targets for every pool row.

    Returns ``(targets, labeled, classes)``. Unlabeled rows (missing
    per :func:`pandas.isna`, or equal to ``unlabeled_marker``) get
    ``-1`` (classification) or ``NaN`` (regression and no target) so a
    loss that forgets the mask fails loudly.
    """
    if y is None:
        if aux_target == "classification":
            targets = torch.full((n_rows,), -1, dtype=torch.int64)
            classes: np.ndarray | None = np.asarray([])
        else:
            targets = torch.full((n_rows,), float("nan"), dtype=torch.float32)
            classes = None
        return targets, torch.zeros(n_rows, dtype=torch.bool), classes
    if aux_target is None:
        raise ValueError(
            "y was given but aux_target is None; pass aux_target="
            "'classification' or 'regression' to use the targets, or y=None."
        )

    if hasattr(y, "to_numpy"):
        y = y.to_numpy()
    y_obj = np.asarray(y, dtype=object).ravel()
    if y_obj.shape[0] != n_rows:
        raise ValueError(f"X has {n_rows} rows but y has {y_obj.shape[0]}.")

    missing = np.asarray(pd.isna(y_obj), dtype=bool)
    if not _is_missing_marker(unlabeled_marker):
        missing |= np.asarray(y_obj == unlabeled_marker, dtype=bool)
    labeled = ~missing
    values = y_obj[labeled].tolist()

    if aux_target == "classification":
        targets_np = np.full(n_rows, -1, dtype=np.int64)
        if values:
            encoder = LabelEncoder().fit(np.asarray(values))
            targets_np[labeled] = encoder.transform(np.asarray(values))
            classes = encoder.classes_
        else:
            classes = np.asarray([])
        targets = torch.from_numpy(targets_np)
    else:
        if isinstance(unlabeled_marker, numbers.Number) and not _is_missing_marker(
            unlabeled_marker
        ):
            warnings.warn(
                f"aux_target='regression' with the numeric unlabeled_marker="
                f"{unlabeled_marker!r}: rows whose target equals it are treated "
                "as unlabeled, although it may be a valid value (e.g. a "
                "log2-MIC). Prefer unlabeled_marker=np.nan for regression.",
                UserWarning,
                stacklevel=3,
            )
        targets_np = np.full(n_rows, np.nan, dtype=np.float32)
        try:
            targets_np[labeled] = np.asarray(values, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "aux_target='regression' needs numeric targets on labeled rows."
            ) from exc
        targets = torch.from_numpy(targets_np)
        classes = None
    return targets, torch.from_numpy(labeled), classes


def make_pretraining_loaders(
    X: Any,
    y: Any | None = None,
    *,
    batch_size: int = 256,
    val_size: float = 0.1,
    random_state: int | None = 0,
    warper: Any | None = None,
    transform_mode: str = "none",
    unlabeled_marker: Any = -1,
    aux_target: str | None = None,
    collate_fn: Callable[[list[Any]], Any] | None = None,
    num_workers: int = 0,
    data_device: str | torch.device | None = None,
    pin_memory: bool = False,
) -> tuple[DataLoader, DataLoader, dict[str, Any]]:
    """Build train / validation loaders over an unlabeled (or partially labeled) pool.

    Pipeline, leak-safe with respect to the validation split:

    1. Random (non-stratified) split of the row indices.
    2. Spectral warping (if ``warper`` is given): fitted on the
       training rows, then the **whole pool is warped once**.
    3. Input transform (``transform_mode``): fitted on the (warped)
       training rows only. It is **not** applied here; the loaders
       yield raw rows and
       :func:`~maldideepkit.pretraining._training.ssl_train_loop`
       applies the fitted state to every collated batch, so a custom
       ``collate_fn`` cannot bypass it. With a mode that leaves the
       bins on the raw intensity scale (``"log1p"``, ``"none"``) the
       per-bin mean and standard deviation of the transformed training
       rows are also computed: the masked-bin pre-texts reconstruct the
       model input standardized with them
       (:meth:`~maldideepkit.pretraining.SSLBatch.recon_target`).
    4. Placement (``data_device``): the (warped) pool stays on the
       host and every batch is read from it at once (a memory-mapped
       pool stays on disk), or it is uploaded once to a device and
       every batch is gathered there. The batch composition does not
       depend on the placement: both use the same ``DataLoader``
       sampler, so the same global PyTorch RNG state gives the same
       batches.

    Parameters
    ----------
    X : array-like or MaldiSet of shape (n_samples, n_bins)
        Spectra pool. A float32 ndarray (including an ``np.memmap``)
        is used without copying when no ``warper`` is given.
    y : array-like of shape (n_samples,) or None, default=None
        Partial auxiliary targets (any dtype, including strings). Rows
        that are missing (:func:`pandas.isna`) or equal to
        ``unlabeled_marker`` are unlabeled. Requires ``aux_target``.
    batch_size : int, default=256
        Mini-batch size of both loaders.
    val_size : float, default=0.1
        Fraction of rows held out for the validation loss.
    random_state : int or None, default=0
        Seed for the split.
    warper : sklearn-style transformer or None, default=None
        Unfitted spectral-alignment transformer (``fit(X)`` +
        ``transform(X)``). The fitted object is returned in
        ``stats["warper"]``.
    transform_mode : str, default="none"
        Resolved input-transform mode (see
        :func:`~maldideepkit.base.data.resolve_transform_mode`).
    unlabeled_marker : object, default=-1
        Value of ``y`` that marks an unlabeled row. ``np.nan`` (or
        ``None``) relies on :func:`pandas.isna` only. With
        ``aux_target="regression"`` a numeric marker emits a
        ``UserWarning`` (it may be a valid target value).
    aux_target : {"classification", "regression"} or None, default=None
        How ``y`` is encoded. ``"classification"``: labels of the
        labeled rows are encoded with
        :class:`~sklearn.preprocessing.LabelEncoder` (classes in
        ``stats["aux_classes"]``), unlabeled rows get ``-1``.
        ``"regression"``: ``float32`` values, unlabeled rows get
        ``NaN``. ``None``: no targets (``y`` must be ``None``).
    collate_fn : callable or None, default=None
        Batching function receiving a list of ``(x_raw, target,
        labeled_mask)`` items and returning a ``(x_raw, target,
        labeled_mask)`` batch. ``None`` uses :func:`stack_collate`.
        This is the seam for variable-length (peak-set) batching.
    num_workers : int, default=0
        ``DataLoader`` worker count (must be ``0`` with a device
        ``data_device``).
    data_device : str, torch.device or None, default=None
        Where the pool lives. ``None`` or ``"cpu"``: on the host, read
        batch by batch. A non-CPU device (e.g. ``"cuda"``): the whole
        (warped) pool is copied to it once, in row chunks (so a
        memory-mapped pool is never materialised in host memory), and
        both loaders yield ``(x_raw, target, labeled_mask)`` tensors
        on that device.
    pin_memory : bool, default=False
        ``DataLoader`` ``pin_memory`` of the host loaders (faster
        asynchronous copies to a GPU). Ignored with a device
        ``data_device``.

    Returns
    -------
    train_loader : DataLoader
        Shuffling loader over the training rows. Drops the last batch
        when it would contain a single sample (``BatchNorm``).
    val_loader : DataLoader
        Non-shuffling loader over the validation rows.
    stats : dict
        The :func:`~maldideepkit.base.data.make_loaders` keys
        (``"mean"``, ``"std"``, ``"warper"``,
        ``"input_transform_state"``, ``"sample_weight_val"`` which is
        always ``None``) plus ``"aux_classes"`` (ndarray or ``None``),
        ``"n_labeled"``, ``"train_indices"``, ``"val_indices"``, and
        ``"recon_stats"`` (``{"mean", "std"}`` of the transformed
        training rows for ``"log1p"`` / ``"none"``, else ``None``).

    Raises
    ------
    ValueError
        If ``X`` has fewer than two rows, ``aux_target`` is unknown,
        ``y`` is given without ``aux_target``, or ``num_workers > 0``
        with a device ``data_device``.
    """
    if aux_target is not None and aux_target not in AUX_TARGETS:
        raise ValueError(
            f"Unknown aux_target={aux_target!r}; expected one of {AUX_TARGETS}."
        )
    device = torch.device("cpu" if data_device is None else data_device)
    if device.type != "cpu" and num_workers > 0:
        raise ValueError(
            f"num_workers must be 0 with data_device={str(device)!r}: the "
            "batches are gathered on the device by the main process."
        )
    X_np = _to_numpy(X)
    n_rows = int(X_np.shape[0])
    if n_rows < 2:
        raise ValueError(f"Pre-training needs at least 2 spectra; got {n_rows}.")
    targets, labeled, classes = _encode_targets(
        y, n_rows, aux_target=aux_target, unlabeled_marker=unlabeled_marker
    )

    idx_tr, idx_val = train_test_split(
        np.arange(n_rows), test_size=val_size, random_state=random_state
    )
    idx_tr = np.sort(idx_tr)
    idx_val = np.sort(idx_val)

    fitted_warper = None
    if warper is not None:
        fitted_warper = warper.fit(pd.DataFrame(X_np[idx_tr]))
        X_np = _warp_numpy(fitted_warper, X_np)

    # Only the stateful modes read the training rows: an empty view
    # gives the same state for "none" / "log1p" without copying the pool.
    if transform_mode in {"standardize", "robust", "log1p+standardize"}:
        X_fit = X_np[idx_tr]
    else:
        X_fit = X_np[:0]
    transform_state = fit_input_transform(X_fit, transform_mode)
    recon_stats = _recon_stats(X_np, idx_tr, transform_state)

    if device.type == "cpu":
        train_ds: Dataset = _PretrainDataset(X_np, idx_tr, targets, labeled)
        val_ds: Dataset = _PretrainDataset(X_np, idx_val, targets, labeled)
    else:
        pool = _upload_pool(X_np, device)
        targets_dev = targets.to(device)
        labeled_dev = labeled.to(device)
        train_ds = _DevicePoolDataset(pool, idx_tr, targets_dev, labeled_dev)
        val_ds = _DevicePoolDataset(pool, idx_val, targets_dev, labeled_dev)
        pin_memory = False
    collate = stack_collate if collate_fn is None else collate_fn
    drop_last = batch_size > 1 and (len(train_ds) % batch_size) == 1
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=bool(pin_memory),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=bool(pin_memory),
    )
    stats: dict[str, Any] = {
        "mean": transform_state.get("mean"),
        "std": transform_state.get("std"),
        "warper": fitted_warper,
        "input_transform_state": transform_state,
        "sample_weight_val": None,
        "aux_classes": classes,
        "n_labeled": int(labeled.sum().item()),
        "train_indices": idx_tr,
        "val_indices": idx_val,
        "recon_stats": recon_stats,
    }
    return train_loader, val_loader, stats


__all__ = ["AUX_TARGETS", "make_pretraining_loaders", "stack_collate"]
