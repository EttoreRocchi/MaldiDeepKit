"""Dual-input spectral classifiers.

This subpackage hosts :class:`MaldiDualBranchClassifier`, which scores
``(spectrum, side-feature)`` pairs by combining a spectral branch
with a small MLP over the side input. The first published use case is
antibiotic-fingerprint conditioning for AMR prediction; the
``smiles_to_fingerprint`` helper converts SMILES strings to ECFP4
fingerprints when the optional ``[rdkit]`` extra is installed.

Future dual-input classifiers (e.g. spectrum + patient-feature,
spectrum + instrument-ID) can subclass
:class:`maldideepkit.base.dual.BaseDualSpectralClassifier` with a
single ``_build_dual_model`` override.
"""

from ._smiles import smiles_to_fingerprint
from .classifier import MaldiDualBranchClassifier

__all__ = [
    "MaldiDualBranchClassifier",
    "smiles_to_fingerprint",
]
