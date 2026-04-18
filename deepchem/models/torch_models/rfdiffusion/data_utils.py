"""Utilities for preparing RFdiffusion-style DeepChem datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np

from deepchem.data import NumpyDataset


@dataclass
class RFDiffusionBatch:
    """A simple container for batched RFdiffusion inputs."""

    residue_types: np.ndarray
    clean_coordinates: np.ndarray
    residue_mask: np.ndarray
    noisy_coordinates: Optional[np.ndarray] = None
    timesteps: Optional[np.ndarray] = None


def _validate_dataset_inputs(residue_types: np.ndarray,
                             coordinates: np.ndarray,
                             residue_mask: Optional[np.ndarray] = None) -> None:
    """Validate the padded arrays used to build an RFdiffusion dataset."""
    if residue_types.ndim != 2:
        raise ValueError(
            "residue_types must have shape (n_samples, n_residues).")
    if coordinates.ndim != 3 or coordinates.shape[-1] != 3:
        raise ValueError(
            "coordinates must have shape (n_samples, n_residues, 3).")
    if residue_types.shape[0] != coordinates.shape[0] or residue_types.shape[
            1] != coordinates.shape[1]:
        raise ValueError(
            "residue_types and coordinates must agree on sample and residue dimensions."
        )
    if residue_mask is not None and residue_mask.shape != residue_types.shape:
        raise ValueError(
            "residue_mask must have the same shape as residue_types.")


def build_rfdiffusion_dataset(
        residue_types: np.ndarray,
        coordinates: np.ndarray,
        residue_mask: Optional[np.ndarray] = None,
        ids: Optional[Sequence[Any]] = None) -> NumpyDataset:
    """Build a DeepChem ``NumpyDataset`` for RFdiffusion-style denoising.

    Parameters
    ----------
    residue_types: np.ndarray
        Integer residue identifiers with shape ``(n_samples, n_residues)``.
        Sequences should already be padded to a common length.
    coordinates: np.ndarray
        Target backbone or C-alpha coordinates with shape
        ``(n_samples, n_residues, 3)``.
    residue_mask: np.ndarray, optional
        Float or boolean mask with shape ``(n_samples, n_residues)``.
        A value of ``1`` marks a valid residue and ``0`` marks padding.
    ids: Sequence, optional
        Optional sample identifiers.

    Returns
    -------
    deepchem.data.NumpyDataset
        A dataset whose ``X`` entries contain residue types and masks, ``y``
        contains clean coordinates, and ``w`` stores the residue mask for
        masked coordinate reconstruction loss.
    """
    residue_types = np.asarray(residue_types, dtype=np.int64)
    coordinates = np.asarray(coordinates, dtype=np.float32)
    if residue_mask is None:
        residue_mask = np.ones_like(residue_types, dtype=np.float32)
    else:
        residue_mask = np.asarray(residue_mask, dtype=np.float32)
    _validate_dataset_inputs(residue_types, coordinates, residue_mask)

    n_samples = residue_types.shape[0]
    features = np.empty((n_samples,), dtype=object)
    for sample_index in range(n_samples):
        features[sample_index] = {
            "residue_types": residue_types[sample_index],
            "residue_mask": residue_mask[sample_index]
        }

    return NumpyDataset(X=features, y=coordinates, w=residue_mask, ids=ids)


def stack_rfdiffusion_inputs(features: np.ndarray) -> Dict[str, np.ndarray]:
    """Collate a batch of object-array RFdiffusion features into dense arrays."""
    if not isinstance(features, np.ndarray) or features.dtype != object:
        raise ValueError(
            "Expected an object array produced by build_rfdiffusion_dataset().")
    if len(features) == 0:
        raise ValueError("RFdiffusion batches must contain at least one sample.")

    residue_types = np.stack(
        [np.asarray(sample["residue_types"], dtype=np.int64) for sample in features])
    residue_mask = np.stack([
        np.asarray(sample.get("residue_mask",
                              np.ones_like(sample["residue_types"],
                                           dtype=np.float32)),
                   dtype=np.float32) for sample in features
    ])
    batch = {
        "residue_types": residue_types,
        "residue_mask": residue_mask
    }
    if "noisy_coordinates" in features[0]:
        batch["noisy_coordinates"] = np.stack([
            np.asarray(sample["noisy_coordinates"], dtype=np.float32)
            for sample in features
        ])
    if "timesteps" in features[0]:
        batch["timesteps"] = np.asarray(
            [sample["timesteps"] for sample in features], dtype=np.float32)
    return batch
