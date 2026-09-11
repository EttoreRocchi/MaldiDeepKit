"""SMILES -> ECFP4 fingerprint helper for the dual-branch classifier."""

from __future__ import annotations

import warnings

import numpy as np


def smiles_to_fingerprint(
    smiles: list[str],
    *,
    fp_dim: int = 2048,
    radius: int = 2,
) -> np.ndarray:
    """Convert a list of SMILES strings to Morgan (ECFP4) fingerprints.

    Requires the optional RDKit extra::

        pip install "maldideepkit[rdkit]"

    Parameters
    ----------
    smiles : list of str
        SMILES strings, one per sample.
    fp_dim : int, default=2048
        Bit-vector length (Morgan / ECFP4 default).
    radius : int, default=2
        Morgan radius. ``radius=2`` + ``fp_dim=2048`` is the ECFP4
        convention.

    Returns
    -------
    fp : ndarray of shape (n_samples, fp_dim), dtype float32
        Binary fingerprint matrix. Rows RDKit cannot parse (and
        non-string entries such as the ``NaN`` a missing table cell
        becomes in pandas) are left all-zero, and a
        :class:`UserWarning` names their positions: an all-zero side
        feature is indistinguishable from a valid one downstream, so
        a typo in a drug table would otherwise train silently.

    Raises
    ------
    ImportError
        If RDKit is not installed.

    Warns
    -----
    UserWarning
        If any SMILES string fails to parse.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import rdFingerprintGenerator
    except ImportError as exc:
        raise ImportError(
            "smiles_to_fingerprint requires the optional RDKit extra. "
            "Install via:  pip install 'maldideepkit[rdkit]'"
        ) from exc

    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=int(radius), fpSize=int(fp_dim)
    )
    out = np.zeros((len(smiles), int(fp_dim)), dtype=np.float32)
    failed: list[int] = []
    for i, smi in enumerate(smiles):
        # Non-strings (NaN/None from a missing pandas cell) would raise
        # inside RDKit's C++ binding; treat them as unparseable rows.
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        if mol is None:
            failed.append(i)
            continue
        bv = generator.GetFingerprint(mol)
        for bit in bv.GetOnBits():
            out[i, bit] = 1.0

    if failed:
        shown = ", ".join(f"{i} ({smiles[i]!r})" for i in failed[:5])
        more = f", and {len(failed) - 5} more" if len(failed) > 5 else ""
        warnings.warn(
            f"smiles_to_fingerprint could not parse {len(failed)} of "
            f"{len(smiles)} SMILES strings; their rows are all-zero and "
            f"carry no information: {shown}{more}.",
            UserWarning,
            stacklevel=2,
        )
    return out
