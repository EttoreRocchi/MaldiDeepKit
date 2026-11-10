# Changelog

All notable changes to MaldiDeepKit will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2026-11-10

### Added

- `maldideepkit.pretraining.MaldiSpectrumEncoder` (also `maldideepkit.MaldiSpectrumEncoder`), a scikit-learn transformer that pre-trains a backbone on unlabeled spectra and returns its embeddings.
- Four pre-text tasks: `"mae"` (the default), `"shuffle"`, `"shuffle+mae"`, and `"contrastive"`. Custom tasks subclass `Pretext` and can be named with `register_pretext`.
- An optional auxiliary head on the encoder (`aux_lambda`, `aux_target`) that also learns from the labeled rows of a partially labeled `y`.
- `init_from` on every classifier: start from the trunk of a pre-trained encoder (a fitted instance or a checkpoint path) and reuse its fitted preprocessing.
- `freeze_trunk_epochs` on every classifier, for linear probing before fine-tuning (LP-FT).
- `init_from` and `freeze_trunk_epochs` on `MaldiDualBranchClassifier`, applied to its spectral branch.
- `get_embeddings(X)` on every classifier.
- `optimizer`, `scheduler`, and `callbacks` on every classifier and on the encoder, with the `maldideepkit.Callback` base class. `loss` also takes a module or a function.
- `register_backbone` and `get_backbone_class`: a registered name works wherever an architecture is accepted.
- Custom backbone modules work with the encoder, `init_from`, and `freeze_trunk_epochs`: the trunk is everything except the `nn.Linear` that produces the output.
- `torch_compile` on every classifier and on the encoder, and `amp_dtype` (`"float16"`, `"bfloat16"`, or `"auto"`) for `use_amp=True`.
- `data_device` on the encoder, to keep the pre-training spectra on the GPU.
- `make_pretraining_loaders`, lazy loaders over a pool of spectra, which can be memory-mapped.
- `MaldiSelfTrainingClassifier.load`, and `sample_weight` on its `fit`.
- `from_spectrum` on `BaseSpectralClassifier` (inherited by custom subclasses) and on `MaldiDualBranchClassifier`.
- `save` stores object parameters (an optimizer, a loss module, a callback, ...) by import path, so they round-trip through `load`. `BaseSpectralClassifier.load` also loads custom classifier classes.
- `trusted` on every `load`: by default a checkpoint imports no new module and uses only a restricted set of classes and functions; `trusted=True` lifts the limit.

### Changed

- **Breaking:** the classifier constructors take keyword arguments only, except `input_dim` and `n_classes`. Calls that pass other arguments by position must name them.
- Requires `maldiamrkit>=0.22.1` (was `>=0.19.0`) and `torch>=2.3` (was `>=2.0`). The reworked `Warping` of maldiamrkit 0.22 aligns spectra differently, so models trained with `warping` differ from 0.3.0, and checkpoints with a warper fitted by maldiamrkit 0.21 or earlier must be retrained. Other 0.3.0 checkpoints load and predict as before, and a fit with the default settings and no `warping` gives the same weights.
- `SpectralEnsemble` is a scikit-learn estimator (`get_params`, `clone`, `GridSearchCV`), and its `fit` takes `sample_weight`.
- `save` is atomic for classifiers, encoders, and ensembles: a failing write leaves an existing checkpoint untouched.
- `find_lr` uses the classifier's `loss` and `optimizer`.
- Docs: shared MaldiSuite theme and links to the other MaldiSuite packages.

### Fixed

- `get_attention_weights(X)` returns the gates of every row of `X`, not only of the last inference batch.
- File names keep their dots: `save("lr0.05")` and `save("lr0.10")` no longer overwrite each other. Classifiers and ensembles saved by 0.3.0 under a dotted name still load, with a warning.
- A loaded classifier keeps its `warping`, so a refit or a `clone` warps the spectra as the original did.
- `save` works with a `SpectrumAugment` as `augment` and with path-valued parameters such as `metrics_log_path`.

## [0.3.0] - 2026-09-11

### Added

- `maldideepkit.dual_branch.MaldiDualBranchClassifier`, which conditions predictions on a per-row side feature (e.g. an antibiotic fingerprint) alongside the spectrum. Side features are passed as extra columns of `X`.
- `maldideepkit.dual_branch.smiles_to_fingerprint` to convert SMILES strings into ECFP4 fingerprints. Optional install extra `rdkit` (`pip install MaldiDeepKit[rdkit]`).
- `maldideepkit.semisupervised.MaldiSelfTrainingClassifier` for iterative pseudo-labeling of partially labeled data (unlabeled rows of `y` are marked with `-1`).
- `sample_weight` argument on `fit()` for every backbone.

### Changed

- `class_weight` combined with `mixup_alpha` or `cutmix_alpha` now normalises the loss consistently with the other training paths, so results differ from 0.2.0 for this combination. Training without mixing is unchanged.
- Minimum `maldiamrkit` requirement raised from `>=0.12.0` to `>=0.19.0`.

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
