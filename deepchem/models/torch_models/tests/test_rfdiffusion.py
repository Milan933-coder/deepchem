import numpy as np
import pytest
import torch
from typing import Tuple

import deepchem as dc
from deepchem.models.torch_models import DeepChemSE3StructureModule
from deepchem.models.torch_models import IterativeSimulator
from deepchem.models.torch_models import RFInputEmbedding
from deepchem.models.torch_models import RFDiffusion
from deepchem.models.torch_models import RFDiffusionModel
from deepchem.models.torch_models import build_rfdiffusion_dataset


def _make_toy_protein_batch(n_samples: int = 4,
                            n_residues: int = 6) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    residue_types = np.tile(
        np.arange(1, n_residues + 1, dtype=np.int64), (n_samples, 1)) % 20
    coordinates = np.zeros((n_samples, n_residues, 3), dtype=np.float32)
    for sample_index in range(n_samples):
        coordinates[sample_index, :, 0] = np.linspace(
            0.0, 1.0 + 0.1 * sample_index, n_residues)
        coordinates[sample_index, :, 1] = 0.2 * sample_index
        coordinates[sample_index, :, 2] = np.linspace(
            -0.5, 0.5, n_residues)
    residue_mask = np.ones((n_samples, n_residues), dtype=np.float32)
    residue_mask[-1, -2:] = 0.0
    coordinates[-1, -2:] = 0.0
    return residue_types, coordinates, residue_mask


@pytest.mark.torch
def test_build_rfdiffusion_dataset():
    residue_types, coordinates, residue_mask = _make_toy_protein_batch()
    dataset = build_rfdiffusion_dataset(residue_types=residue_types,
                                        coordinates=coordinates,
                                        residue_mask=residue_mask)

    assert isinstance(dataset, dc.data.NumpyDataset)
    assert len(dataset) == residue_types.shape[0]
    assert dataset.y.shape == coordinates.shape
    assert dataset.w.shape == residue_mask.shape
    assert dataset.X[0]["residue_types"].shape == (residue_types.shape[1],)


@pytest.mark.torch
def test_rfdiffusion_embedding_and_track_shapes():
    residue_types, coordinates, residue_mask = _make_toy_protein_batch(
        n_samples=2, n_residues=5)
    residue_types_t = torch.tensor(residue_types, dtype=torch.long)
    residue_mask_t = torch.tensor(residue_mask, dtype=torch.float32)
    timesteps_t = torch.tensor([0.3, 0.7], dtype=torch.float32)
    coordinates_t = torch.tensor(coordinates, dtype=torch.float32)

    embedding = RFInputEmbedding(d_msa=32, d_msa_full=16, d_pair=24, d_state=20)
    msa, msa_full, pair, state, residue_indices = embedding(
        residue_types_t, residue_mask_t, timesteps_t)

    simulator = IterativeSimulator(n_extra_block=1,
                                   n_main_block=1,
                                   n_ref_block=1,
                                   d_msa=32,
                                   d_msa_full=16,
                                   d_pair=24,
                                   d_state=20,
                                   n_head_msa=4,
                                   n_head_pair=4,
                                   use_deepchem_se3=False,
                                   se3_top_k=4)
    _, pair_out, coordinate_trajectory, torsion_trajectory, state_out = simulator(
        msa=msa,
        msa_full=msa_full,
        pair=pair,
        coordinates=coordinates_t,
        state=state,
        residue_mask=residue_mask_t,
        residue_indices=residue_indices)

    assert msa.shape == (2, 1, 5, 32)
    assert msa_full.shape == (2, 1, 5, 16)
    assert pair_out.shape == (2, 5, 5, 24)
    assert coordinate_trajectory.shape == (3, 2, 5, 3)
    assert torsion_trajectory.shape == (3, 2, 5, 10, 2)
    assert state_out.shape == (2, 5, 20)


@pytest.mark.torch
def test_deepchem_se3_structure_wrapper():
    node_states = torch.randn(2, 5, 16)
    pair_features = torch.randn(2, 5, 5, 12)
    coordinates = torch.randn(2, 5, 3)
    residue_mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]],
                                dtype=torch.float32)

    structure_module = DeepChemSE3StructureModule(node_dim=16,
                                                  pair_dim=12,
                                                  use_deepchem_se3=True,
                                                  top_k=4)
    updated_states, updated_coordinates = structure_module(node_states,
                                                           pair_features,
                                                           coordinates,
                                                           residue_mask)

    assert updated_states.shape == node_states.shape
    assert updated_coordinates.shape == coordinates.shape
    assert torch.allclose(updated_coordinates[1, -1], torch.zeros(3), atol=1e-5)


@pytest.mark.torch
def test_rfdiffusion_module_forward_and_sample():
    residue_types, coordinates, residue_mask = _make_toy_protein_batch(
        n_samples=2, n_residues=5)
    model = RFDiffusion(d_msa=32,
                        d_msa_full=16,
                        d_pair=24,
                        d_state=20,
                        n_extra_block=1,
                        n_main_block=1,
                        n_ref_block=1,
                        use_deepchem_se3=False,
                        se3_top_k=4)

    batch = {
        "residue_types": torch.tensor(residue_types, dtype=torch.long),
        "noisy_coordinates": torch.tensor(coordinates, dtype=torch.float32),
        "residue_mask": torch.tensor(residue_mask, dtype=torch.float32),
        "timesteps": torch.tensor([0.5, 0.8], dtype=torch.float32)
    }
    outputs = model(batch)

    assert outputs.shape == (2, 5, 3)

    sampled = model.sample(residue_types=residue_types,
                           residue_mask=residue_mask,
                           num_steps=4)
    assert sampled.shape == (2, 5, 3)
    assert torch.allclose(sampled[1, -1], torch.zeros(3), atol=1e-5)


@pytest.mark.torch
def test_rfdiffusion_model_fit_and_predict():
    np.random.seed(0)
    torch.manual_seed(0)

    residue_types, coordinates, residue_mask = _make_toy_protein_batch(
        n_samples=6, n_residues=5)
    dataset = build_rfdiffusion_dataset(residue_types=residue_types,
                                        coordinates=coordinates,
                                        residue_mask=residue_mask)
    model = RFDiffusionModel(d_msa=32,
                             d_msa_full=16,
                             d_pair=24,
                             d_state=20,
                             n_extra_block=1,
                             n_main_block=1,
                             n_ref_block=1,
                             batch_size=2,
                             learning_rate=0.001,
                             use_deepchem_se3=False,
                             random_seed=0)

    loss = model.fit(dataset, nb_epoch=1, deterministic=True)
    predictions = model.predict(dataset)

    assert np.isfinite(loss)
    assert predictions.shape == coordinates.shape
    assert np.isfinite(predictions).all()
