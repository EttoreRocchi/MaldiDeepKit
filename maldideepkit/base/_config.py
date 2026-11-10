"""Recursive configuration comparison for sklearn-style estimators."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
import torch

_PANDAS_TYPES = (pd.DataFrame, pd.Series, pd.Index)


def _is_estimator_like(obj: Any) -> bool:
    """Return ``True`` for instances exposing a callable ``get_params``."""
    return not isinstance(obj, type) and callable(getattr(obj, "get_params", None))


def _same_value(a: Any, b: Any, ignore: frozenset[str]) -> bool:
    """Compare two hyperparameter values for :func:`same_config`."""
    if a is b:
        return True
    if _is_estimator_like(a) or _is_estimator_like(b):
        return same_config(a, b, ignore=ignore)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(
            _same_value(x, y, ignore) for x, y in zip(a, b, strict=True)
        )
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same_value(a[k], b[k], ignore) for k in a)
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        if not (isinstance(a, np.ndarray) and isinstance(b, np.ndarray)):
            return False
        try:
            return bool(np.array_equal(a, b, equal_nan=True))
        except TypeError:
            return bool(np.array_equal(a, b))
    if isinstance(a, _PANDAS_TYPES) or isinstance(b, _PANDAS_TYPES):
        return type(a) is type(b) and bool(a.equals(b))
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)):
            return False
        if a.shape != b.shape or a.dtype != b.dtype:
            return False
        a_cpu, b_cpu = a.detach().cpu(), b.detach().cpu()
        if a_cpu.is_floating_point():
            nan_a, nan_b = torch.isnan(a_cpu), torch.isnan(b_cpu)
            return torch.equal(nan_a, nan_b) and torch.equal(
                a_cpu[~nan_a], b_cpu[~nan_b]
            )
        return torch.equal(a_cpu, b_cpu)
    if isinstance(a, np.random.RandomState) or isinstance(b, np.random.RandomState):
        return type(a) is type(b) and _same_value(
            a.get_state(legacy=False), b.get_state(legacy=False), ignore
        )
    if isinstance(a, np.random.Generator) or isinstance(b, np.random.Generator):
        return type(a) is type(b) and _same_value(
            a.bit_generator.state, b.bit_generator.state, ignore
        )
    if (
        isinstance(a, float)
        and isinstance(b, float)
        and math.isnan(a)
        and math.isnan(b)
    ):
        return True
    try:
        eq = a == b
    except Exception:
        return False
    if isinstance(eq, (bool, np.bool_)):
        return bool(eq)
    return False


def same_config(
    a: Any,
    b: Any,
    ignore: Iterable[str] = ("n_jobs", "verbose"),
) -> bool:
    """Return whether two objects carry the same configuration.

    Unlike ``a.get_params() == b.get_params()``, the comparison is
    recursive, so it is robust to estimators that build sub-estimators
    in ``__init__`` (e.g. :class:`maldiamrkit.alignment.Warping`, which
    instantiates a fresh ``MaldiPeakDetector`` and therefore never
    compares equal by plain parameter equality). Fitted state is
    ignored: a fitted and an unfitted estimator with the same
    constructor parameters have the same config.

    Rules:

    - Estimator-like objects (with ``get_params``) must have exactly
      the same class; they are compared key by key on
      ``get_params(deep=False)``, skipping the keys in ``ignore``, and
      estimator-valued parameters are compared recursively.
    - Lists and tuples are compared element-wise (so
      ``Pipeline.steps`` recurses into each step, and a list equals a
      tuple with the same elements), dicts key by key.
    - NumPy arrays are compared with :func:`numpy.array_equal`, pandas
      objects (``DataFrame``, ``Series``, ``Index``) with their
      ``equals`` method, and PyTorch tensors with :func:`torch.equal`
      (same shape and dtype; NaN equals NaN in all three).
    - NumPy random generators (``RandomState`` and ``Generator``) are
      compared by type and current state, so two generators seeded
      alike and not yet drawn from have the same config.
    - Any other value is compared with ``==`` (two NaN floats are
      equal); when ``==`` does not return a boolean (or raises), the
      values are equal only if they are the same object.

    Parameters
    ----------
    a, b : object
        Objects to compare (typically unfitted or fitted estimators).
    ignore : iterable of str, default=("n_jobs", "verbose")
        Parameter names skipped at every recursion level (settings
        that do not change what the estimator computes).

    Returns
    -------
    bool
        ``True`` when ``a`` and ``b`` are configured identically.
    """
    ignore_set = frozenset(ignore)
    if a is b:
        return True
    if not (_is_estimator_like(a) or _is_estimator_like(b)):
        return _same_value(a, b, ignore_set)
    if type(a) is not type(b):
        return False
    params_a = a.get_params(deep=False)
    params_b = b.get_params(deep=False)
    keys_a = set(params_a) - ignore_set
    keys_b = set(params_b) - ignore_set
    if keys_a != keys_b:
        return False
    return all(_same_value(params_a[k], params_b[k], ignore_set) for k in keys_a)


__all__ = ["same_config"]
