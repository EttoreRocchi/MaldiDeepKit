"""Zenodo-hosted demo dataset loader for the MaldiDeepKit notebooks.

The notebooks in this folder share a single dataset: **MALDI-Kleb-AI**
(Rocchi et al. 2026, Zenodo DOI ``10.5281/zenodo.17405072``), a 370 MB
archive of real MALDI-TOF mass spectra of *Klebsiella* isolates from
three Italian clinical centres, with Amikacin / Meropenem antimicrobial-
resistance annotations.

The helper in this module downloads the tarball once, caches it under
``~/.cache/maldideepkit/`` (or the directory pointed to by the
``MALDIDEEPKIT_CACHE_DIR`` environment variable), extracts it, and
returns a :class:`DemoDataset` populated with:

* ``X``    - binned feature matrix obtained via
  :class:`maldiamrkit.MaldiSet` (samples x m/z bins).
* ``meta`` - per-sample metadata: ``Batch`` (= acquisition centre),
  ``Species`` (Klebsiella species), and the AMR labels
  ``Amikacin`` / ``Meropenem`` (``R``/``S``/``I``).
* ``mz``   - m/z axis reported by ``MaldiSet`` (bin starts in Da).
* ``maldi_set`` - the underlying ``MaldiSet`` object, for notebooks
  that want to plug straight into the MaldiAMRKit ecosystem.

This module lives exclusively under ``notebooks/`` and is intentionally
kept out of the installable ``maldideepkit`` package.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tarfile
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "ZENODO_DOI",
    "ZENODO_RECORD_ID",
    "ZENODO_TAR_MD5",
    "ZENODO_TAR_URL",
    "DemoDataset",
    "binary_labels",
    "get_cache_dir",
    "load_maldi_kleb_ai",
]


ZENODO_RECORD_ID = "17405072"
ZENODO_DOI = "10.5281/zenodo.17405072"
ZENODO_TAR_URL = (
    f"https://zenodo.org/records/{ZENODO_RECORD_ID}/files/maldi-tof.tar?download=1"
)
ZENODO_TAR_MD5 = "c14b6c6b4210553962faa7f1dc27d275"

_DATASET_DIRNAME = "maldi-kleb-ai"
_TAR_NAME = "maldi-tof.tar"


def get_cache_dir() -> Path:
    """Resolve the root cache directory for MaldiDeepKit demo data.

    Priority:

    1. ``$MALDIDEEPKIT_CACHE_DIR`` environment variable (absolute path).
    2. ``~/.cache/maldideepkit/`` (XDG-style default, cross-platform).

    The directory is created on first access.
    """
    env = os.environ.get("MALDIDEEPKIT_CACHE_DIR")
    if env:
        root = Path(env).expanduser().resolve()
    else:
        root = Path.home() / ".cache" / "maldideepkit"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _dataset_paths(cache_dir: Path | None = None) -> dict[str, Path]:
    root = (cache_dir or get_cache_dir()) / _DATASET_DIRNAME
    root.mkdir(parents=True, exist_ok=True)
    return {
        "root": root,
        "tar": root / _TAR_NAME,
        "extract": root / "extracted",
    }


def _md5_of(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_with_progress(url: str, dest: Path, *, verbose: bool = True) -> None:
    """Stream ``url`` to ``dest`` with an optional progress bar."""
    tmp = dest.with_suffix(dest.suffix + ".partial")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "maldideepkit-demo"})
    with urllib.request.urlopen(req) as resp, tmp.open("wb") as out:
        total = int(resp.headers.get("Content-Length", "0") or 0)
        read = 0
        step = max(1 << 20, total // 50) if total else 1 << 20  # ~50 ticks
        last_mark = 0
        chunk_size = 1 << 16
        while True:
            buf = resp.read(chunk_size)
            if not buf:
                break
            out.write(buf)
            read += len(buf)
            if verbose and total and read - last_mark >= step:
                pct = 100.0 * read / total
                print(f"  downloading maldi-tof.tar ... {pct:5.1f} %", end="\r")
                last_mark = read
    tmp.replace(dest)
    if verbose:
        print(" " * 60, end="\r")


def _ensure_tar(
    cache_dir: Path | None, *, force: bool = False, verbose: bool = True
) -> Path:
    paths = _dataset_paths(cache_dir)
    tar = paths["tar"]
    if force and tar.exists():
        tar.unlink()
    if tar.exists() and _md5_of(tar) == ZENODO_TAR_MD5:
        return tar
    if tar.exists():  # stale / corrupted
        tar.unlink()
    if verbose:
        print(
            f"Downloading MALDI-Kleb-AI from Zenodo (DOI {ZENODO_DOI}; "
            f"370 MB, one-shot) to {tar} ..."
        )
    _download_with_progress(ZENODO_TAR_URL, tar, verbose=verbose)
    got = _md5_of(tar)
    if got != ZENODO_TAR_MD5:
        tar.unlink(missing_ok=True)
        raise RuntimeError(
            f"MD5 mismatch for {tar.name}: expected {ZENODO_TAR_MD5}, "
            f"got {got}. The download may be corrupted; re-run with "
            f"force_redownload=True."
        )
    return tar


def _ensure_extracted(
    cache_dir: Path | None,
    *,
    force: bool = False,
    verbose: bool = True,
) -> Path:
    paths = _dataset_paths(cache_dir)
    extract_dir = paths["extract"]
    sentinel = extract_dir / "metadata.csv"
    if force and extract_dir.exists():
        shutil.rmtree(extract_dir)
    if sentinel.exists():
        return extract_dir
    tar = _ensure_tar(cache_dir, verbose=verbose)
    if verbose:
        print(f"Extracting {tar.name} ...")
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar, mode="r:*") as tf:
        tf.extractall(extract_dir, filter="data")
    if not sentinel.exists():
        raise RuntimeError(
            f"Extraction completed but {sentinel} is missing. "
            f"Is the tarball layout still 'spectra/*.txt + metadata.csv'?"
        )
    return extract_dir


@dataclass
class DemoDataset:
    """MALDI-Kleb-AI binned intensities + aligned metadata."""

    X: pd.DataFrame
    """Binned feature matrix, shape ``(n_samples, n_bins)``."""

    meta: pd.DataFrame
    """Per-sample metadata (``Batch``, ``Species``, ``Amikacin``,
    ``Meropenem``), indexed by ``X.index``."""

    mz: np.ndarray
    """m/z axis as reported by ``MaldiSet`` (bin-start values in Da,
    e.g. ``[2000, 2003, ..., 19997]`` for the default 2000-20000 Da
    window with ``bin_width=3``). Length ``n_bins``."""

    info: dict = field(default_factory=dict)
    """Provenance info: Zenodo DOI, MD5 checksum, loader arguments."""

    maldi_set: Any = None
    """Underlying ``maldiamrkit.MaldiSet`` for notebooks that need it."""

    @property
    def batch(self) -> pd.Series:
        """Alias for ``meta['Batch']`` (acquisition centre)."""
        return self.meta["Batch"]

    @property
    def species(self) -> pd.Series:
        """Alias for ``meta['Species']``."""
        return self.meta["Species"]


def load_maldi_kleb_ai(
    *,
    antibiotic: str = "Amikacin",
    bin_width: int = 3,
    cache_dir: Path | None = None,
    force_redownload: bool = False,
    verbose: bool = False,
) -> DemoDataset:
    """Download (once) and return the MALDI-Kleb-AI demo dataset.

    Parameters
    ----------
    antibiotic : {'Amikacin', 'Meropenem'}, default='Amikacin'
        Which AMR column to expose as the primary label in
        ``ds.maldi_set.y``. Both columns are kept in ``ds.meta``
        regardless.
    bin_width : int, default=3
        Bin width in Daltons forwarded to
        :meth:`maldiamrkit.MaldiSet.from_directory`.
    cache_dir : Path, optional
        Root cache directory override. When ``None``, the module uses
        ``$MALDIDEEPKIT_CACHE_DIR`` if set, otherwise
        ``~/.cache/maldideepkit/``.
    force_redownload : bool, default=False
        Re-download the tarball and re-extract even if a valid cache
        already exists.
    verbose : bool, default=False
        Print progress for the download and extraction steps.

    Returns
    -------
    DemoDataset
        With ``X``, ``meta`` (Batch / Species / Amikacin / Meropenem),
        ``mz``, and the underlying ``maldi_set``.

    Notes
    -----
    * The dataset is **real clinical data** (Klebsiella isolates from
      Rome, Milan, Catania); please cite the Zenodo record if you reuse
      it. DOI: ``10.5281/zenodo.17405072``.
    * First call downloads 370 MB; subsequent calls are millisecond-fast
      (the loader re-uses the extracted spectra).
    """
    try:
        from maldiamrkit import MaldiSet
    except ImportError as exc:  # pragma: no cover - maldiamrkit is a core dep
        raise ImportError(
            "load_maldi_kleb_ai requires maldiamrkit. Install it with "
            "`pip install maldiamrkit`."
        ) from exc

    if antibiotic not in ("Amikacin", "Meropenem"):
        raise ValueError(
            f"antibiotic must be one of 'Amikacin' / 'Meropenem', got {antibiotic!r}."
        )

    paths = _dataset_paths(cache_dir)
    extract_dir = _ensure_extracted(
        cache_dir, force=force_redownload, verbose=verbose
    )
    metadata_csv = extract_dir / "metadata.csv"
    spectra_dir = extract_dir / "spectra"

    ds = MaldiSet.from_directory(
        str(spectra_dir),
        str(metadata_csv),
        aggregate_by={"antibiotics": [antibiotic]},
        bin_width=bin_width,
        verbose=verbose,
    )
    X = ds.X
    meta = ds.meta.loc[X.index].copy()

    # Normalise metadata columns to the canonical names used by the
    # other notebooks: City -> Batch.
    if "City" in meta.columns:
        meta = meta.rename(columns={"City": "Batch"})

    mz = np.asarray(X.columns, dtype=float)

    info = {
        "source": "Zenodo MALDI-Kleb-AI",
        "doi": ZENODO_DOI,
        "record_id": ZENODO_RECORD_ID,
        "md5_tar": ZENODO_TAR_MD5,
        "n_samples": X.shape[0],
        "n_bins": X.shape[1],
        "bin_width": bin_width,
        "antibiotic": antibiotic,
        "cache_dir": str(paths["root"]),
    }
    return DemoDataset(
        X=X, meta=meta, mz=mz, info=info, maldi_set=ds
    )


def binary_labels(
    demo: DemoDataset,
    *,
    antibiotic: str | None = None,
    positive: str = "R",
) -> tuple[pd.DataFrame, pd.Series]:
    """Drop unlabelled / intermediate samples and return ``(X, y)`` for
    binary classification.

    Spectra whose ``meta[antibiotic]`` value is missing or not in
    ``{positive, *negatives}`` are dropped. ``y == 1`` for the
    ``positive`` class (default ``'R'``, resistant) and ``0`` for the
    other resolved label (typically ``'S'``, susceptible).

    Parameters
    ----------
    demo : DemoDataset
        As returned by :func:`load_maldi_kleb_ai`.
    antibiotic : {'Amikacin', 'Meropenem'}, optional
        Which AMR column to binarise. Defaults to
        ``demo.info['antibiotic']``.
    positive : str, default='R'
        Label that maps to ``y == 1``.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix restricted to labelled rows.
    y : pd.Series of int
        Binary labels aligned to ``X.index``.
    """
    col = antibiotic or demo.info["antibiotic"]
    labels = demo.meta[col].astype("string")
    keep = labels.isin([positive, "S"])
    X = demo.X.loc[keep]
    y = (labels.loc[keep] == positive).astype(int).rename(col)
    return X, y
