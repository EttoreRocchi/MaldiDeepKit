"""JSON codec of the estimator parameters stored in checkpoints.

An object parameter (a :class:`~maldideepkit.pretraining.Pretext`, a
:class:`~maldideepkit.augment.SpectrumAugment`, a loss module, a
callback) is stored as its class import path plus its constructor
parameters; a class, a function, or a :func:`functools.partial` of them
by import path. They are re-imported by :func:`_import_class` at load
time.

With the default ``trusted=False``, no new module is imported (importing
a module runs its code): every class named by the checkpoint must come
from MaldiDeepKit or from an imported module. An object given as
``optimizer``, ``scheduler``, ``loss`` or ``callbacks`` must also be a
torch module, optimizer or scheduler, or a ``Callback``
(:func:`_restricted_class`), and a function must be defined in
MaldiDeepKit or in ``torch.nn.functional`` (:func:`_restricted_function`),
so a class such as :class:`subprocess.Popen` or a function such as
:func:`os.system` is refused. ``True`` lifts these limits, for
checkpoints from a trusted source. A ``.warper.pkl`` is a pickle, read
whatever ``trusted`` is: load checkpoints with a warper only from trusted
sources.
"""

from __future__ import annotations

import functools
import importlib
import inspect
import os
import sys
from typing import Any

import numpy as np
import torch

from ..augment.spectra import SpectrumAugment
from ..utils.callbacks import TrainingHooks

_PACKAGE = "maldideepkit"
_SAFE_FUNCTION_MODULES = (_PACKAGE, "torch.nn.functional")
_SAFE_OBJECT_BASES = (
    torch.nn.Module,
    torch.optim.Optimizer,
    torch.optim.lr_scheduler.LRScheduler,
    torch.optim.lr_scheduler.ReduceLROnPlateau,
    TrainingHooks,
)


class _UntrustedClassError(ValueError):
    """A checkpoint names a class or function that ``trusted=False`` refuses."""


def _json_safe(value: Any) -> Any:
    """Convert NumPy scalars, devices, and paths to JSON-serialisable values."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, os.PathLike):
        return str(value)
    return value


def _class_path(cls: type) -> str:
    """Return the importable ``module.QualName`` path of ``cls``."""
    return f"{cls.__module__}.{cls.__qualname__}"


def _may_import(module: str) -> bool:
    """Return whether ``module`` may be imported without ``trusted=True``.

    True for MaldiDeepKit modules and for modules already in
    :data:`sys.modules` (importing them runs no new code).
    """
    return (
        module == _PACKAGE
        or module.startswith(_PACKAGE + ".")
        or sys.modules.get(module) is not None
    )


def _import_class(
    path: str, base: type | None, what: str, *, trusted: bool = False
) -> Any:
    """Import the ``base`` subclass saved as ``path`` (see :func:`_class_path`).

    The longest importable module prefix of ``path`` is imported and the
    rest of the path is resolved as attributes (nested classes
    included). Unless ``trusted``, a module that is neither part of
    MaldiDeepKit nor already in :data:`sys.modules` is never imported.

    Parameters
    ----------
    path : str
        ``module.QualName`` path of the class.
    base : type or None
        Required base class; ``None`` accepts any callable (a function
        or a class).
    what : str
        Name of the class role, for error messages.
    trusted : bool, default=False
        Allow importing any module.

    Returns
    -------
    type or callable
        The class (or callable, with ``base=None``).

    Raises
    ------
    ValueError
        If no module prefix of ``path`` holds the class, if a prefix
        that may hold it was not imported because ``trusted=False``, or
        if it is not a ``base`` subclass.
    ImportError
        If importing a module of ``path`` fails for another reason than
        the module itself being missing (e.g. an import inside it).
    """
    parts = path.split(".")
    skipped: list[str] = []
    for i in range(len(parts) - 1, 0, -1):
        module = ".".join(parts[:i])
        if not all(part.isidentifier() for part in parts[:i]):
            continue
        if not (trusted or _may_import(module)):
            skipped.append(module)
            continue
        try:
            obj: Any = importlib.import_module(module)
        except ModuleNotFoundError as exc:
            # Only ``module`` itself (or a parent package) being missing
            # means "try a shorter prefix"; any other missing module was
            # imported by the code of ``module``.
            missing = exc.name
            if missing is None or not (
                module == missing or module.startswith(missing + ".")
            ):
                raise
            continue
        if not hasattr(obj, "__path__"):
            # A plain module has no submodules: no longer prefix skipped
            # above can be a module.
            skipped.clear()
        try:
            for name in parts[i:]:
                obj = getattr(obj, name)
        except AttributeError:
            continue
        if base is None:
            if callable(obj):
                return obj
            raise ValueError(f"The saved {what} {path!r} is not callable.")
        if isinstance(obj, type) and issubclass(obj, base):
            return obj
        raise ValueError(
            f"The saved {what} class {path!r} is not a {base.__name__} subclass."
        )
    if skipped:
        raise _UntrustedClassError(
            f"Cannot import the {what} class {path!r} saved in the checkpoint: "
            f"its module ({skipped[-1]!r} or a submodule) is not part of "
            "MaldiDeepKit and not imported in this session, and load() only "
            "imports such a module with trusted=True (importing a module runs "
            "its code). Import the module that defines the class before "
            "load(), or pass trusted=True if the checkpoint comes from a "
            "trusted source."
        )
    raise ValueError(
        f"Cannot import the {what} class {path!r} saved in the checkpoint. "
        "Define it at module level in an importable module (and import that "
        "module) before loading."
    )


def _restricted_class(path: str, what: str, *, trusted: bool) -> type:
    """Import a class named by a checkpoint, limited to training components.

    Unless ``trusted``, the class must derive from one of
    :data:`_SAFE_OBJECT_BASES`, on top of the module rule of
    :func:`_import_class`.

    Raises
    ------
    _UntrustedClassError
        If the class is another one and ``trusted`` is false.
    """
    cls = _import_class(path, object, what, trusted=trusted)
    if not (trusted or issubclass(cls, _SAFE_OBJECT_BASES)):
        raise _UntrustedClassError(
            f"The checkpoint names the {what} class {path!r}, which is not a "
            "training component (a torch module, optimizer or scheduler, or a "
            "Callback). load() only uses such a class with trusted=True, if the "
            "checkpoint comes from a trusted source."
        )
    return cls


def _restricted_function(path: str, what: str, *, trusted: bool) -> Any:
    """Import a function named by a checkpoint.

    Unless ``trusted``, it must be defined in one of
    :data:`_SAFE_FUNCTION_MODULES`, whatever the path it was reached
    through.

    Raises
    ------
    _UntrustedClassError
        If it is defined elsewhere and ``trusted`` is false.
    """
    function = _import_class(path, None, what, trusted=trusted)
    module = getattr(function, "__module__", None) or ""
    if not (
        trusted
        or any(
            module == safe or module.startswith(safe + ".")
            for safe in _SAFE_FUNCTION_MODULES
        )
    ):
        raise _UntrustedClassError(
            f"The checkpoint names the {what} function {path!r}, defined in "
            f"{module!r}. load() only uses a function of MaldiDeepKit or "
            "torch.nn.functional by default: pass trusted=True for your own "
            "functions, if the checkpoint comes from a trusted source."
        )
    return function


def _importable_class_path(cls: Any, base: type | None, what: str, name: str) -> str:
    """Return the import path of ``cls``, checking that it re-imports to ``cls``.

    Called at save time so that a class :func:`_import_class` could not
    restore (a local class, a class redefined after the instance was
    built, or one of a module that cannot be imported) fails before any
    file is written.

    Raises
    ------
    ValueError
        If the path does not re-import to ``cls``.
    """
    module = getattr(cls, "__module__", None)
    path = f"{module}.{getattr(cls, '__qualname__', repr(cls))}"
    message = (
        f"Cannot save the {name} parameter: its {what} "
        f"{'class' if isinstance(cls, type) else 'function'} {path!r} cannot "
        "be re-imported from that path at load time. Define it at module "
        "level in an importable module (not a lambda, not inside a function, "
        "and not redefined after the instance was created)."
    )
    try:
        found = _import_class(path, base, what, trusted=True)
    except (ImportError, ValueError) as exc:
        raise ValueError(message) from exc
    if found is not cls:
        raise ValueError(message)
    return path


def _estimator_spec(
    estimator: Any, params: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Return the reconstructable spec of a classifier used as a parameter.

    ``"__class__"`` records the import path, so a custom classifier class
    (absent from the built-in registry) can be re-imported; it is checked
    to re-import to the class itself before anything is saved.

    Parameters
    ----------
    estimator : BaseSpectralClassifier
        The classifier.
    params : dict or None, default=None
        Its encoded parameters; ``None`` uses ``estimator._hparam_dict()``.

    Returns
    -------
    dict
        ``{"__estimator__": <class name>, "__class__": <path>, "params": ...}``.
    """
    # Lazy import: ``.classifier`` imports this module.
    from .classifier import BaseSpectralClassifier

    return {
        "__estimator__": type(estimator).__name__,
        "__class__": _importable_class_path(
            type(estimator), BaseSpectralClassifier, "backbone", "backbone"
        ),
        "params": estimator._hparam_dict() if params is None else params,
    }


def _init_parameters(cls: type) -> dict[str, inspect.Parameter]:
    """Return the constructor parameters of ``cls``, following ``**kwargs`` up the MRO.

    A class whose ``__init__`` takes ``**kwargs`` also has the parameters
    of the next ``__init__`` in the MRO; ``*args`` and ``**kwargs``
    themselves are left out.
    """
    found: dict[str, inspect.Parameter] = {}
    for klass in cls.__mro__:
        init = klass.__dict__.get("__init__")
        if init is None or klass is object:
            continue
        parameters = list(inspect.signature(init).parameters.values())[1:]
        for parameter in parameters:
            if parameter.kind not in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
                found.setdefault(parameter.name, parameter)
        if not any(p.kind is p.VAR_KEYWORD for p in parameters):
            break
    return found


def _object_params(name: str, value: Any) -> dict[str, Any]:
    """Return the constructor parameters of the object ``value``, read back.

    From ``get_params`` when it has one, else from the attributes of the
    same name. An argument with a default and no such attribute (e.g. the
    deprecated ``size_average`` of the :mod:`torch.nn` losses) is left to
    its default.

    Raises
    ------
    TypeError
        If a required argument has no attribute of the same name.
    """
    if callable(getattr(value, "get_params", None)):
        return value.get_params(deep=False)
    params: dict[str, Any] = {}
    for key, parameter in _init_parameters(type(value)).items():
        if hasattr(value, key):
            params[key] = getattr(value, key)
        elif parameter.default is parameter.empty:
            raise TypeError(
                f"Cannot save the {name} parameter {value!r}: "
                f"{type(value).__name__} does not keep its constructor argument "
                f"{key!r} in an attribute of the same name."
            )
    return params


def _encode_param(name: str, value: Any) -> Any:
    """Return a JSON-safe spec of the parameter ``name`` with value ``value``.

    A :class:`~maldideepkit.pretraining.Pretext` becomes
    ``{"__pretext__": <class path>, "params": {...}}``, a
    :class:`~maldideepkit.augment.SpectrumAugment`
    ``{"__augment__": <class path>, "params": {...}}`` and any other
    object ``{"__object__": <class path>, "params": {...}}`` (its
    constructor parameters, from ``get_params`` or from the attributes of
    the same name). A class becomes ``{"__class_ref__": <path>}``, a
    function ``{"__function__": <path>}`` and a
    :func:`functools.partial` ``{"__partial__": ..., "args": [...],
    "keywords": {...}}``. Lists, tuples, and dicts are encoded
    element-wise, tensors and arrays as lists.

    Raises
    ------
    TypeError
        If an object does not keep a required constructor argument in an
        attribute of the same name.
    ValueError
        If a class or function cannot be re-imported from its path (see
        :func:`_importable_class_path`).
    """
    # Lazy import: ``maldideepkit.pretraining`` imports this module.
    from ..pretraining._pretext import Pretext

    value = _json_safe(value)
    if isinstance(value, Pretext):
        return {
            "__pretext__": _importable_class_path(
                type(value), Pretext, "pretext", name
            ),
            "params": {
                k: _encode_param(f"{name}__{k}", v)
                for k, v in value.get_params(deep=False).items()
            },
        }
    if isinstance(value, SpectrumAugment):
        return {
            "__augment__": _importable_class_path(
                type(value), SpectrumAugment, "augment", name
            ),
            "params": {
                k: _encode_param(f"{name}__{k}", v)
                for k, v in _object_params(name, value).items()
            },
        }
    if isinstance(value, (list, tuple)):
        return [_encode_param(name, v) for v in value]
    if isinstance(value, dict):
        return {str(k): _encode_param(f"{name}__{k}", v) for k, v in value.items()}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, torch.Tensor):
        return {"__tensor__": value.detach().cpu().tolist()}
    if isinstance(value, np.ndarray):
        return {"__ndarray__": value.tolist()}
    if isinstance(value, functools.partial):
        return {
            "__partial__": _encode_param(name, value.func),
            "args": [_encode_param(name, v) for v in value.args],
            "keywords": {
                k: _encode_param(f"{name}__{k}", v) for k, v in value.keywords.items()
            },
        }
    if isinstance(value, type):
        return {"__class_ref__": _importable_class_path(value, None, "object", name)}
    if inspect.isroutine(value):
        return {"__function__": _importable_class_path(value, None, "object", name)}
    return _encode_object(name, value)


def _encode_object(name: str, value: Any) -> dict[str, Any]:
    """Return the ``{"__object__": ...}`` spec of an object parameter."""
    cls = type(value)
    if cls.__module__ == "builtins":
        raise TypeError(
            f"Cannot save the {name} parameter {value!r}: a {cls.__name__} is "
            "neither a JSON value nor an instance of an importable class."
        )
    return {
        "__object__": _importable_class_path(cls, None, "object", name),
        "params": {
            k: _encode_param(f"{name}__{k}", v)
            for k, v in _object_params(name, value).items()
        },
    }


def _decode_param(value: Any, *, trusted: bool = False) -> Any:
    """Inverse of :func:`_encode_param` (classes are re-imported by path).

    ``trusted`` is passed to :func:`_import_class`,
    :func:`_restricted_class`, and :func:`_restricted_function`.
    """
    from ..pretraining._pretext import Pretext

    for key, base, what in (
        ("__pretext__", Pretext, "pretext"),
        ("__augment__", SpectrumAugment, "augment"),
    ):
        if isinstance(value, dict) and key in value:
            cls = _import_class(value[key], base, what, trusted=trusted)
            return cls(
                **{
                    k: _decode_param(v, trusted=trusted)
                    for k, v in value["params"].items()
                }
            )
    if isinstance(value, list):
        return [_decode_param(v, trusted=trusted) for v in value]
    if not isinstance(value, dict):
        return value
    if "__object__" in value:
        cls = _restricted_class(value["__object__"], "object", trusted=trusted)
        return cls(
            **{k: _decode_param(v, trusted=trusted) for k, v in value["params"].items()}
        )
    if "__class_ref__" in value:
        return _restricted_class(value["__class_ref__"], "parameter", trusted=trusted)
    if "__function__" in value:
        return _restricted_function(value["__function__"], "parameter", trusted=trusted)
    if "__partial__" in value:
        return functools.partial(
            _decode_param(value["__partial__"], trusted=trusted),
            *[_decode_param(v, trusted=trusted) for v in value["args"]],
            **{
                k: _decode_param(v, trusted=trusted)
                for k, v in value["keywords"].items()
            },
        )
    if "__tensor__" in value:
        return torch.tensor(value["__tensor__"])
    if "__ndarray__" in value:
        return np.asarray(value["__ndarray__"])
    return {k: _decode_param(v, trusted=trusted) for k, v in value.items()}
