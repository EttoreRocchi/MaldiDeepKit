"""Seeding and device-placement helpers for deterministic training."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + CUDA) RNGs in one call.

    Parameters
    ----------
    seed : int
        Non-negative integer used for every RNG. Also fixes
        ``PYTHONHASHSEED`` in the current process environment.
    deterministic : bool, default=False
        When ``True``, additionally enable PyTorch's deterministic
        algorithm mode. Sets
        ``torch.use_deterministic_algorithms(True, warn_only=True)``,
        ``torch.backends.cudnn.deterministic = True``,
        ``torch.backends.cudnn.benchmark = False``, and
        ``CUBLAS_WORKSPACE_CONFIG=:4096:8``. The env-var must be set
        before the first CUDA context is created. Once enabled,
        determinism is **sticky** - subsequent plain
        ``seed_everything(seed)`` calls do not turn it off.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def resolve_device(device: str | torch.device | None) -> torch.device:
    """Resolve a user-facing device specifier to a :class:`torch.device`.

    Parameters
    ----------
    device : {"auto", "cpu", "cuda"} or torch.device or None
        ``"auto"`` (or ``None``) picks ``cuda`` when available and falls
        back to ``cpu``.

    Returns
    -------
    torch.device
        The resolved device.

    Raises
    ------
    ValueError
        If ``device`` is an unknown string.
    """
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(device, torch.device):
        return device
    if isinstance(device, str):
        return torch.device(device)
    raise ValueError(f"Unsupported device specifier: {device!r}")
