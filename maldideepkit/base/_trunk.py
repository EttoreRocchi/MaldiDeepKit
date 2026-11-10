"""Trunk / final-layer boundary helpers for the backbone modules.

Every built-in backbone module exposes a ``final_layer`` property (the
:class:`~torch.nn.Linear` producing the logits) and a ``feature_dim``
attribute (its input width). Everything before ``final_layer`` is the
trunk; :func:`final_layer_prefix` locates ``final_layer`` by module
identity so callers can split a state dict into trunk and final-layer
tensors without hard-coding attribute names. A custom module without
``final_layer`` follows the dual-branch convention: its final layer is
the :class:`~torch.nn.Linear` that produces its output (:func:`trunk_prefix`,
:func:`split_trunk`). :class:`TrunkFreezer` holds the trunk frozen
during the linear-probing phase of ``freeze_trunk_epochs``.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from itertools import chain
from typing import TYPE_CHECKING, Any, NamedTuple

import torch
from torch import nn

if TYPE_CHECKING:
    from ..utils.training import EarlyStopping


def final_layer_prefix(module: nn.Module) -> str:
    """Return the qualified name of ``module.final_layer`` inside ``module``.

    The lookup compares module identity over
    :meth:`~torch.nn.Module.named_modules`, so it follows the actual
    registration (e.g. ``"head.8"`` for the default MLP, ``"fc"`` for
    the ResNet) rather than an assumed attribute path. State-dict keys
    of the final layer start with ``prefix + "."``.

    Parameters
    ----------
    module : torch.nn.Module
        Backbone module exposing a ``final_layer`` property.

    Returns
    -------
    str
        Dotted submodule name of ``module.final_layer``.

    Raises
    ------
    AttributeError
        If ``module`` has no ``final_layer``.
    ValueError
        If ``module.final_layer`` is not a registered strict submodule
        of ``module``.
    """
    final = module.final_layer
    for name, sub in module.named_modules():
        if sub is final and name:
            return name
    raise ValueError(
        f"final_layer of {type(module).__name__} is not a registered "
        "submodule of the module."
    )


def _traced_final_linear(module: nn.Module, input_dim: int) -> str | None:
    """Return the name of the Linear producing the output of ``module``, by tracing.

    Runs ``module`` once on a zero batch of shape ``(2, input_dim)`` with
    forward hooks on its :class:`~torch.nn.Linear` layers. The layer whose
    output is the module's output wins; when nothing is (e.g. a final
    ``squeeze``), the last layer called. The module is left as it was:
    ``eval()`` and no gradients during the call, training flags restored,
    RNG state untouched.

    Returns
    -------
    str or None
        The qualified name, or ``None`` when the forward fails (e.g. a
        module expecting another input shape) or calls no Linear.
    """
    names = {
        sub: name
        for name, sub in module.named_modules()
        if name and isinstance(sub, nn.Linear)
    }
    if not names:
        return None
    called: list[tuple[nn.Module, Any]] = []
    handles = [
        sub.register_forward_hook(lambda s, _in, out: called.append((s, out)))
        for sub in names
    ]
    flags = [(sub, sub.training) for sub in module.modules()]
    reference = next(chain(module.parameters(), module.buffers()), None)
    device = torch.device("cpu") if reference is None else reference.device
    floating = reference is not None and reference.is_floating_point()
    dtype = reference.dtype if floating else torch.float32
    module.eval()
    try:
        with (
            torch.no_grad(),
            torch.random.fork_rng(devices=[device] if device.type == "cuda" else []),
        ):
            output = module(torch.zeros(2, input_dim, device=device, dtype=dtype))
    except Exception:
        return None
    finally:
        for handle in handles:
            handle.remove()
        for sub, flag in flags:
            sub.training = flag
    if not called:
        return None
    outputs = output if isinstance(output, (tuple, list)) else (output,)
    for sub, out in reversed(called):
        if any(out is candidate for candidate in outputs):
            return names[sub]
    return names[called[-1][0]]


def trunk_prefix(module: nn.Module, input_dim: int | None = None) -> str:
    """Return the qualified name of the final layer of a backbone module.

    ``module.final_layer`` (see :func:`final_layer_prefix`) when the
    module exposes it, as the built-in backbones do. A custom module
    follows the convention shared with the dual-branch spectral branch:
    with ``input_dim``, the :class:`~torch.nn.Linear` that produces its
    output, found with a forward pass on zeros, whatever the order in
    which the layers are registered; without ``input_dim``, or when that
    forward fails, the last :class:`~torch.nn.Linear` in
    :meth:`~torch.nn.Module.named_modules` order.

    Parameters
    ----------
    module : torch.nn.Module
        Backbone module.
    input_dim : int or None, default=None
        Width of the module input, used to locate the final layer of a
        custom module.

    Returns
    -------
    str
        Dotted submodule name of the final layer.

    Raises
    ------
    ValueError
        If ``module`` has no ``final_layer`` and no
        :class:`~torch.nn.Linear` submodule.
    """
    if hasattr(module, "final_layer"):
        return final_layer_prefix(module)
    if input_dim is not None:
        traced = _traced_final_linear(module, int(input_dim))
        if traced is not None:
            return traced
    last = None
    for name, sub in module.named_modules():
        if name and isinstance(sub, nn.Linear):
            last = name
    if last is None:
        raise ValueError(
            f"{type(module).__name__} exposes no final_layer and has no "
            "nn.Linear submodule, so its trunk / final-layer boundary is "
            "undefined."
        )
    return last


class TrunkSplit(NamedTuple):
    """Trunk / final-layer boundary of a backbone module (see :func:`split_trunk`).

    Attributes
    ----------
    features : callable
        Maps a ``(batch, input_dim)`` input to the ``(batch,
        feature_dim)`` embedding (the module without its final layer).
    feature_dim : int
        Embedding width (input width of the final layer).
    final_prefix : str
        Qualified name of the final layer (:func:`trunk_prefix`).
    """

    features: Callable[[torch.Tensor], torch.Tensor]
    feature_dim: int
    final_prefix: str


class _FeaturesView:
    """Callable running ``module`` with its final layer swapped for an Identity.

    Built from shallow copies of ``module`` and of every ancestor of the
    final layer, the final layer being an :class:`~torch.nn.Identity` in
    the copy. Every other submodule, parameter, and buffer is shared with
    ``module`` (so training through the view trains ``module``, and
    device / dtype moves of ``module`` are seen by the view), while the
    module's own structure and ``state_dict`` stay untouched. The
    ``training`` flags of the copied modules follow the originals at
    every call. The module's forward must reach the final layer through
    its registered submodule.
    """

    def __init__(self, module: nn.Module, prefix: str) -> None:
        *path, leaf = prefix.split(".")
        root = self._shallow(module)
        self.pairs = [(root, module)]
        parent_copy, parent = root, module
        for part in path:
            child = parent._modules[part]
            child_copy = self._shallow(child)
            parent_copy._modules[part] = child_copy
            self.pairs.append((child_copy, child))
            parent_copy, parent = child_copy, child
        parent_copy._modules[leaf] = nn.Identity()
        self.view = root

    @staticmethod
    def _shallow(module: nn.Module) -> nn.Module:
        new = copy.copy(module)
        new._modules = type(module._modules)(module._modules)
        return new

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for view, original in self.pairs:
            view.training = original.training
        return self.view(x)


def split_trunk(module: nn.Module, input_dim: int | None = None) -> TrunkSplit:
    """Locate the trunk / final-layer boundary of a backbone module.

    A module exposing ``final_layer`` (the built-in backbones) uses it,
    with its ``feature_dim`` and ``forward_features`` when present. A
    custom module follows the dual-branch convention: the trunk is
    everything except its final :class:`~torch.nn.Linear` (see
    :func:`trunk_prefix`), ``feature_dim`` is that layer's
    ``in_features`` and the features are the module's forward with that
    layer replaced by an :class:`~torch.nn.Identity` on a
    parameter-sharing copy (the module itself is not modified).

    Parameters
    ----------
    module : torch.nn.Module
        Backbone module.
    input_dim : int or None, default=None
        Width of the module input (see :func:`trunk_prefix`).

    Returns
    -------
    TrunkSplit
        ``(features, feature_dim, final_prefix)``.

    Raises
    ------
    ValueError
        If the final layer cannot be located (see :func:`trunk_prefix`).
    """
    prefix = trunk_prefix(module, input_dim)
    interface = hasattr(module, "final_layer")
    feature_dim = getattr(module, "feature_dim", None) if interface else None
    if feature_dim is None:
        feature_dim = module.get_submodule(prefix).in_features
    if interface and hasattr(module, "forward_features"):
        features = module.forward_features
    else:
        features = _FeaturesView(module, prefix)
    return TrunkSplit(features, int(feature_dim), prefix)


def _in_prefix(key: str, prefix: str | None) -> bool:
    """Return whether ``key`` names ``prefix`` or one of its sub-entries."""
    return prefix is not None and (key == prefix or key.startswith(prefix + "."))


def trunk_state(module: nn.Module, skip_prefix: str | None) -> dict[str, torch.Tensor]:
    """Return the trunk entries of ``module.state_dict()``.

    Parameters
    ----------
    module : torch.nn.Module
        Module holding the trunk.
    skip_prefix : str or None
        Qualified name of the final layer (e.g. from
        :func:`final_layer_prefix`); its parameters and buffers are
        left out. ``None`` keeps every entry.

    Returns
    -------
    dict of str to torch.Tensor
        Parameters and buffers outside ``skip_prefix`` (the tensors
        share storage with ``module``).
    """
    return {
        k: v for k, v in module.state_dict().items() if not _in_prefix(k, skip_prefix)
    }


def load_trunk_state(
    module: nn.Module, skip_prefix: str | None, tensors: dict[str, torch.Tensor]
) -> None:
    """Copy ``tensors`` into the trunk of ``module`` (final layer untouched).

    The load is strict: every key of ``tensors`` must exist in
    ``module`` with a matching shape. Values are copied in place (so
    ``tensors`` may live on another device).
    """
    state = module.state_dict()
    unknown = sorted(k for k in tensors if k not in state or _in_prefix(k, skip_prefix))
    if unknown:
        raise ValueError(f"Not trunk entries of {type(module).__name__}: {unknown}.")
    state.update(tensors)
    module.load_state_dict(state)


class TrunkFreezer:
    """``on_epoch_start`` hook holding the trunk frozen for the first epochs.

    Implements the linear-probing phase of LP-FT: for epochs
    ``< n_frozen`` every trunk parameter of ``module`` (everything
    outside ``skip_prefix``) has ``requires_grad=False`` and the trunk
    sub-modules run in ``eval()`` mode (BatchNorm running statistics
    and dropout frozen), so only the final layer trains. At epoch
    ``n_frozen`` the original ``requires_grad`` flags are restored
    (a parameter frozen by the user stays frozen) and the
    early-stopping patience restarts.

    Parameters
    ----------
    module : torch.nn.Module
        Module whose trunk is frozen (typically the whole model).
    skip_prefix : str or None
        Qualified name of the final layer, which stays trainable and in
        training mode.
    n_frozen : int
        Number of frozen epochs.
    early_stopping : EarlyStopping
        Its patience is reset when the trunk is unfrozen.

    Attributes
    ----------
    unfroze : bool
        Whether epoch ``n_frozen`` was reached (``True`` when
        ``n_frozen == 0``).
    last_epoch : int
        Last epoch started (``-1`` before training).
    initial_trunk : dict of str to torch.Tensor
        Copy of the trunk parameters and buffers taken at construction.
    """

    def __init__(
        self,
        module: nn.Module,
        skip_prefix: str | None,
        n_frozen: int,
        early_stopping: EarlyStopping,
    ) -> None:
        self.module = module
        self.skip_prefix = skip_prefix
        self.final = None if skip_prefix is None else module.get_submodule(skip_prefix)
        self.n_frozen = int(n_frozen)
        self.early_stopping = early_stopping
        self.flags = [(p, p.requires_grad) for p in module.parameters()]
        self.trunk_params = [
            p
            for name, p in module.named_parameters()
            if not _in_prefix(name, skip_prefix)
        ]
        self.initial_trunk = {
            k: v.detach().clone() for k, v in trunk_state(module, skip_prefix).items()
        }
        self.unfroze = self.n_frozen == 0
        self.last_epoch = -1

    def __call__(self, epoch: int, model: nn.Module) -> None:
        self.last_epoch = int(epoch)
        if epoch < self.n_frozen:
            for p in self.trunk_params:
                p.requires_grad_(False)
            self.module.eval()
            if self.final is not None:
                self.final.train()
        elif not self.unfroze:
            self.restore_flags()
            self.early_stopping.reset_patience()
            self.unfroze = True

    def restore_flags(self) -> None:
        """Restore the ``requires_grad`` flags recorded at construction."""
        for p, flag in self.flags:
            p.requires_grad_(flag)

    def restore_initial_trunk(self) -> None:
        """Reload the trunk parameters and buffers recorded at construction."""
        load_trunk_state(self.module, self.skip_prefix, self.initial_trunk)
