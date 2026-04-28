# Contributing to MaldiDeepKit

Thanks for considering a contribution!

## Development setup

```bash
git clone https://github.com/EttoreRocchi/MaldiDeepKit.git
cd MaldiDeepKit
pip install -e ".[dev]"
pre-commit install
```

## Testing

```bash
make test            # fast subset (excludes slow markers)
make test-cov        # full run with term + HTML coverage report
```

The coverage gate is 95%. Please add tests for any new public API.

## Linting

```bash
make lint            # ruff check --fix
make format          # ruff format
```

Pre-commit hooks run `ruff-check --fix`, `ruff-format`,
`end-of-file-fixer`, and `trailing-whitespace`. Please make sure they
are installed (`pre-commit install`).

## Style

- NumPy-style docstrings for all public API.
- `BaseSpectralClassifier` subclasses should override only
  `_build_model()` - the base class handles device placement, validation
  splits, early stopping, calibration, and persistence. Store fitted
  attributes with trailing underscores (`threshold_`, `temperature_`,
  `warper_`, ...).
- `predict` / `predict_proba` must be idempotent - no side effects
  outside `fit`.
- Seed Python, NumPy, and PyTorch via `random_state` so identical
  configs produce identical weights.

## Adding a new classifier

Every classifier lives in its own subpackage (`maldideepkit/<family>/`)
and consists of:

1. A `torch.nn.Module` subclass implementing the architecture.
2. A classifier subclass of `BaseSpectralClassifier` whose only required
   override is `_build_model()` returning the module.

Follow the existing modules as templates. Add the classifier to the
package `__init__.py`, wire it into `conftest.py`'s `ALL_FACTORIES`
dict, and add per-model tests in `tests/test_<family>.py`.

## Pull requests

- Branch off `main`, rebase before opening the PR.
- Summarise the change in one sentence in the PR title and use the PR
  body to motivate the design decision.
- Update `CHANGELOG.md` under the next version heading.
- Make sure `make test` and `make lint` are clean; coverage stays
  ≥ 95%.

## Reporting issues

Open an issue on
[GitHub](https://github.com/EttoreRocchi/MaldiDeepKit/issues) with a
minimal reproducer:

- MaldiDeepKit version (`python -c "import maldideepkit; print(maldideepkit.__version__)"`)
- PyTorch, scikit-learn, and `maldiamrkit` versions
- Feature matrix shape, classifier and kwargs, and the exception or
  wrong result
