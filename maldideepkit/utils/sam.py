"""Sharpness-Aware Minimization (SAM) optimizer wrapper.

SAM pushes weights toward flatter regions of the loss landscape, at
the cost of ~2x the forward / backward compute per step.

Usage
-----

.. code-block:: python

    optimizer = SAMOptimizer(
        model.parameters(), base_optimizer=torch.optim.AdamW,
        rho=0.05, lr=1e-3, weight_decay=0.05,
    )

    loss = criterion(model(x), y)
    loss.backward()
    optimizer.first_step(zero_grad=True)

    loss = criterion(model(x), y)
    loss.backward()
    optimizer.second_step(zero_grad=True)
"""

from __future__ import annotations

from typing import Any

import torch


class SAMOptimizer(torch.optim.Optimizer):
    """Wrap a base optimizer in the SAM two-step update.

    Parameters
    ----------
    params : iterable
        Parameters or param-group dicts (as for any torch optimizer).
    base_optimizer : type
        The base optimizer **class** (e.g. :class:`torch.optim.AdamW`).
        Instantiated internally against the same param groups.
    rho : float, default=0.05
        Size of the ascent step in parameter space. Paper default is
        ``0.05``. Typical range: ``[0.01, 0.2]``.
    **base_kwargs
        Forwarded to the base optimizer (e.g. ``lr``, ``weight_decay``).
    """

    def __init__(
        self,
        params: Any,
        base_optimizer: type[torch.optim.Optimizer],
        rho: float = 0.05,
        **base_kwargs: Any,
    ) -> None:
        if rho <= 0:
            raise ValueError(f"rho must be > 0; got {rho!r}.")
        defaults = {"rho": float(rho), **base_kwargs}
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **base_kwargs)
        self.param_groups = self.base_optimizer.param_groups
        for g in self.param_groups:
            g.setdefault("rho", float(rho))

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False) -> None:
        """Ascend to ``w + e`` using the current gradients."""
        grad_norm = self._grad_norm()
        eps = 1e-12
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + eps)
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale
                self.state[p]["e_w"] = e_w
                p.add_(e_w)
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False) -> None:
        """Undo the ascent and apply the base optimizer step from ``w``."""
        for group in self.param_groups:
            for p in group["params"]:
                if "e_w" in self.state.get(p, {}):
                    p.sub_(self.state[p]["e_w"])
                    del self.state[p]["e_w"]
        self.base_optimizer.step()
        self._step_count = getattr(self.base_optimizer, "_step_count", 0)
        self._opt_called = True
        if zero_grad:
            self.zero_grad()

    def step(self, closure: Any = None) -> Any:
        """Unsupported. Use ``first_step`` / ``second_step`` instead."""
        raise RuntimeError(
            "SAMOptimizer requires an explicit two-pass training loop. "
            "Call first_step() after the first backward, then recompute "
            "the loss and backward, then call second_step()."
        )

    @torch.no_grad()
    def _grad_norm(self) -> torch.Tensor:
        ref_device = None
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    ref_device = p.grad.device
                    break
            if ref_device is not None:
                break
        if ref_device is None:
            return torch.tensor(0.0)
        norms = [
            p.grad.norm(p=2).to(ref_device)
            for group in self.param_groups
            for p in group["params"]
            if p.grad is not None
        ]
        return torch.norm(torch.stack(norms), p=2)
