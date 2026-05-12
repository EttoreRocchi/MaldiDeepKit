# Changelog

All notable changes to MaldiDeepKit will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-05-12

### Added

- `maldideepkit.uncertainty` subpackage with three drop-in uncertainty-quantification estimators sharing a single `predict_with_uncertainty` interface:
  - `MCDropoutEstimator` for Monte Carlo Dropout with epistemic / aleatoric decomposition.
  - `LaplaceEstimator` (last-layer or full-network Laplace approximation via the optional `laplace-torch` dependency).
  - `ConformalPredictor` for split conformal prediction with the LAC non-conformity score.
- `UncertaintyResult` dataclass as the unified return value.
- Optional install extra `uncertainty` (`pip install MaldiDeepKit[uncertainty]`) for `laplace-torch`.
- `warm_start: bool = False` keyword-only argument on `BaseSpectralClassifier.fit()` (forwarded through `MaldiMLPClassifier.fit()`). When `True` and `self.model_` already exists, `fit()` reuses the existing `torch.nn.Module` as the starting point of training instead of rebuilding it via `_build_model()`. Unblocks federated-learning, continual-learning, and fine-tuning workflows that need `fit()` to resume from the current weights rather than reinitialise.

## [0.1.0] - 2026-04-28

Initial release.
