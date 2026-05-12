"""Uncertainty-quantification estimators for fitted MaldiDeepKit classifiers.

Three drop-in estimators that share a single
:meth:`predict_with_uncertainty` interface:

- :class:`MCDropoutEstimator` - Monte Carlo Dropout. Stochastic
  forward passes with dropout active; decomposes total uncertainty
  into epistemic (model disagreement) and aleatoric (data noise)
  components.
- :class:`LaplaceEstimator` - Last-layer or full-network Laplace
  approximation via the optional ``laplace-torch`` dependency.
- :class:`ConformalPredictor` - Distribution-free split conformal
  prediction with the LAC non-conformity score; produces calibrated
  prediction sets without retraining the model.

All three return a :class:`UncertaintyResult` so downstream code can
swap methods without changing call sites.
"""

from __future__ import annotations

from ._base import BaseUncertaintyEstimator
from ._result import UncertaintyResult
from .conformal import ConformalPredictor
from .laplace import LaplaceEstimator
from .mc_dropout import MCDropoutEstimator

__all__ = [
    "BaseUncertaintyEstimator",
    "ConformalPredictor",
    "LaplaceEstimator",
    "MCDropoutEstimator",
    "UncertaintyResult",
]
