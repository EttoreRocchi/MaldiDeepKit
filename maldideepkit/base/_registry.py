"""Names of the classifier architectures.

A name is accepted wherever the package takes an architecture:
``MaldiSpectrumEncoder(backbone=...)``,
``MaldiDualBranchClassifier(backbone=...)``,
``MaldiSelfTrainingClassifier(base_estimator=...)``, and
``SpectralEnsemble([...])``. :func:`register_backbone` adds a name.
"""

from __future__ import annotations

_RESERVED = {
    "peakformer": (
        "backbone='peakformer' is reserved for the peak-set transformer "
        "planned for MaldiDeepKit v0.5.0; use a registered name or a "
        "classifier template."
    ),
}
_REGISTERED: dict[str, type] = {}


def _builtin_backbones() -> dict[str, type]:
    """Return the built-in names (lazy imports: these modules import ``base``)."""
    from ..attention.mlp import MaldiMLPClassifier
    from ..cnn.cnn import MaldiCNNClassifier
    from ..resnet.resnet import MaldiResNetClassifier
    from ..transformer.transformer import MaldiTransformerClassifier

    return {
        "mlp": MaldiMLPClassifier,
        "cnn": MaldiCNNClassifier,
        "resnet": MaldiResNetClassifier,
        "transformer": MaldiTransformerClassifier,
    }


def register_backbone(name: str, cls: type) -> type:
    """Register a classifier class under ``name``.

    The name is then accepted wherever the package takes an architecture,
    and builds ``cls()`` with its default parameters, as ``"resnet"``
    does; pass an instance to set parameters. Registering a name again
    replaces its class; the built-in names and ``"peakformer"`` cannot be
    registered. A checkpoint saved with a registered name stores the name
    only, so register it again before ``load``.

    Parameters
    ----------
    name : str
        The name.
    cls : type
        Concrete single-input
        :class:`~maldideepkit.BaseSpectralClassifier` subclass.

    Returns
    -------
    type
        ``cls`` (unchanged).

    Raises
    ------
    ValueError
        If ``name`` is empty, built-in, or reserved.
    TypeError
        If ``cls`` is not a concrete single-input classifier class.
    """
    from .classifier import BaseSpectralClassifier
    from .dual import BaseDualSpectralClassifier

    if not isinstance(name, str) or not name:
        raise ValueError(f"name must be a non-empty string; got {name!r}.")
    if not (
        isinstance(cls, type)
        and issubclass(cls, BaseSpectralClassifier)
        and not issubclass(cls, BaseDualSpectralClassifier)
        and not getattr(cls, "__abstractmethods__", frozenset())
    ):
        raise TypeError(
            "cls must be a concrete single-input BaseSpectralClassifier subclass; "
            f"got {cls!r}."
        )
    if name in _builtin_backbones() or name in _RESERVED:
        raise ValueError(
            f"{name!r} is a built-in or reserved backbone and cannot be registered; "
            "pass an instance of your class instead."
        )
    _REGISTERED[name] = cls
    return cls


def get_backbone_class(name: str) -> type:
    """Return the classifier class registered under ``name``.

    Parameters
    ----------
    name : str
        A built-in name (``"mlp"``, ``"cnn"``, ``"resnet"``,
        ``"transformer"``) or one added with :func:`register_backbone`.

    Returns
    -------
    type
        The class.

    Raises
    ------
    NotImplementedError
        If ``name`` is reserved for a later release.
    ValueError
        If ``name`` is not registered.
    """
    if name in _RESERVED:
        raise NotImplementedError(_RESERVED[name])
    known = {**_builtin_backbones(), **_REGISTERED}
    if name not in known:
        raise ValueError(
            f"Unknown backbone={name!r}; expected one of {tuple(known)}, "
            f"{tuple(_RESERVED)} (planned) or a classifier template."
        )
    return known[name]


__all__ = ["get_backbone_class", "register_backbone"]
