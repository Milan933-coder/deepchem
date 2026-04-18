"""A small PR-style RFdiffusion module for DeepChem."""

from __future__ import annotations

import datetime
import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

from deepchem.data import Dataset
from deepchem.models.losses import L2Loss, Loss
from deepchem.models.torch_models.rfdiffusion.data_utils import stack_rfdiffusion_inputs
from deepchem.models.torch_models.rfdiffusion.embeddings import RFInputEmbedding
from deepchem.models.torch_models.rfdiffusion.track_module import IterativeSimulator
from deepchem.models.torch_models.torch_model import TorchModel

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:
    raise ImportError("These classes require PyTorch to be installed.")

logger = logging.getLogger(__name__)


def _masked_centroid(coordinates: torch.Tensor,
                     residue_mask: torch.Tensor) -> torch.Tensor:
    """Compute a masked centroid for each structure."""
    weights = residue_mask.unsqueeze(-1)
    denominator = weights.sum(dim=1).clamp(min=1.0)
    return (coordinates * weights).sum(dim=1) / denominator


class RFDiffusion(nn.Module):
    """DeepChem-native RFdiffusion starter with Embeddings and Track modules.

    The architecture is intentionally modest, but it now mirrors the upstream
    code layout more closely:
    - `embeddings.py`: latent/full MSA, pair, and state construction
    - `track_module.py`: iterative MSA, pair, and structure updates
    - `se3.py`: optional DeepChem SE(3) structural updates
    """

    def __init__(self,
                 num_residue_types: int = 21,
                 d_msa: int = 128,
                 d_msa_full: int = 64,
                 d_pair: int = 64,
                 d_state: int = 64,
                 n_extra_block: int = 1,
                 n_main_block: int = 2,
                 n_ref_block: int = 1,
                 n_head_msa: int = 4,
                 n_head_pair: int = 4,
                 d_rbf: int = 36,
                 p_drop: float = 0.1,
                 use_deepchem_se3: bool = True,
                 se3_top_k: int = 16):
        super().__init__()
        self.input_embedding = RFInputEmbedding(num_residue_types=num_residue_types,
                                                d_msa=d_msa,
                                                d_msa_full=d_msa_full,
                                                d_pair=d_pair,
                                                d_state=d_state,
                                                p_drop=p_drop)
        self.simulator = IterativeSimulator(n_extra_block=n_extra_block,
                                            n_main_block=n_main_block,
                                            n_ref_block=n_ref_block,
                                            d_msa=d_msa,
                                            d_msa_full=d_msa_full,
                                            d_pair=d_pair,
                                            d_state=d_state,
                                            n_head_msa=n_head_msa,
                                            n_head_pair=n_head_pair,
                                            d_rbf=d_rbf,
                                            p_drop=p_drop,
                                            use_deepchem_se3=use_deepchem_se3,
                                            se3_top_k=se3_top_k)

    @property
    def using_deepchem_se3(self) -> bool:
        """Whether any structural block is using DeepChem SE(3) layers."""
        return self.simulator.uses_deepchem_se3()

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        residue_types = inputs["residue_types"].long()
        noisy_coordinates = inputs["noisy_coordinates"].float()
        residue_mask = inputs["residue_mask"].float()
        timesteps = inputs["timesteps"].float()

        centroid = _masked_centroid(noisy_coordinates, residue_mask)
        centered_coordinates = (noisy_coordinates - centroid.unsqueeze(1))
        centered_coordinates = centered_coordinates * residue_mask.unsqueeze(-1)

        msa, msa_full, pair, state, residue_indices = self.input_embedding(
            residue_types=residue_types,
            residue_mask=residue_mask,
            timesteps=timesteps)
        _, _, coordinate_trajectory, _, _ = self.simulator(
            msa=msa,
            msa_full=msa_full,
            pair=pair,
            coordinates=centered_coordinates,
            state=state,
            residue_mask=residue_mask,
            residue_indices=residue_indices)
        predicted_coordinates = coordinate_trajectory[-1] + centroid.unsqueeze(1)
        return predicted_coordinates * residue_mask.unsqueeze(-1)

    @torch.no_grad()
    def sample(self,
               residue_types: Union[np.ndarray, torch.Tensor],
               residue_mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
               num_steps: int = 25,
               initial_noise_scale: float = 1.0) -> torch.Tensor:
        """Generate coordinates by iterative denoising."""
        device = next(self.parameters()).device
        residue_types = torch.as_tensor(residue_types,
                                        dtype=torch.long,
                                        device=device)
        if residue_types.ndim == 1:
            residue_types = residue_types.unsqueeze(0)

        if residue_mask is None:
            residue_mask = torch.ones_like(residue_types,
                                           dtype=torch.float32,
                                           device=device)
        else:
            residue_mask = torch.as_tensor(residue_mask,
                                           dtype=torch.float32,
                                           device=device)
            if residue_mask.ndim == 1:
                residue_mask = residue_mask.unsqueeze(0)

        coordinates = torch.randn(residue_types.shape[0],
                                  residue_types.shape[1],
                                  3,
                                  device=device) * initial_noise_scale
        coordinates = coordinates * residue_mask.unsqueeze(-1)

        schedule = torch.linspace(1.0, 0.0, steps=num_steps + 1, device=device)
        for step_index in range(num_steps):
            current_t = float(schedule[step_index].item())
            next_t = float(schedule[step_index + 1].item())
            predicted_coordinates = self({
                "residue_types":
                residue_types,
                "noisy_coordinates":
                coordinates,
                "residue_mask":
                residue_mask,
                "timesteps":
                torch.full((residue_types.shape[0],),
                           current_t,
                           dtype=torch.float32,
                           device=device)
            })
            blend = 1.0 if current_t <= 1e-6 else 1.0 - (next_t / current_t)
            coordinates = coordinates + blend * (predicted_coordinates -
                                                 coordinates)
            coordinates = coordinates * residue_mask.unsqueeze(-1)

        return coordinates


class RFDiffusionModel(TorchModel):
    """DeepChem `TorchModel` wrapper for the RFdiffusion starter."""

    def __init__(self,
                 num_residue_types: int = 21,
                 d_msa: int = 128,
                 d_msa_full: int = 64,
                 d_pair: int = 64,
                 d_state: int = 64,
                 n_extra_block: int = 1,
                 n_main_block: int = 2,
                 n_ref_block: int = 1,
                 n_head_msa: int = 4,
                 n_head_pair: int = 4,
                 d_rbf: int = 36,
                 p_drop: float = 0.1,
                 use_deepchem_se3: bool = True,
                 se3_top_k: int = 16,
                 sigma_min: float = 0.05,
                 sigma_max: float = 1.0,
                 min_timestep: float = 0.02,
                 random_seed: Optional[int] = None,
                 loss: Optional[Loss] = None,
                 batch_size: int = 8,
                 **kwargs):
        model = RFDiffusion(num_residue_types=num_residue_types,
                            d_msa=d_msa,
                            d_msa_full=d_msa_full,
                            d_pair=d_pair,
                            d_state=d_state,
                            n_extra_block=n_extra_block,
                            n_main_block=n_main_block,
                            n_ref_block=n_ref_block,
                            n_head_msa=n_head_msa,
                            n_head_pair=n_head_pair,
                            d_rbf=d_rbf,
                            p_drop=p_drop,
                            use_deepchem_se3=use_deepchem_se3,
                            se3_top_k=se3_top_k)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.min_timestep = min_timestep
        self._rng = np.random.default_rng(random_seed)
        super().__init__(model=model,
                         loss=loss if loss is not None else L2Loss(),
                         batch_size=batch_size,
                         **kwargs)

    def _prepare_batch(
        self, batch: Tuple[Any, Any, Any]
    ) -> Tuple[List[Dict[str, torch.Tensor]], List[torch.Tensor], List[torch.Tensor]]:
        inputs, labels, weights = batch

        if isinstance(inputs, list) and len(inputs) == 1:
            inputs = inputs[0]
        if isinstance(labels, list) and len(labels) == 1:
            labels = labels[0]
        if isinstance(weights, list) and len(weights) == 1:
            weights = weights[0]

        if isinstance(inputs, np.ndarray):
            inputs = stack_rfdiffusion_inputs(inputs)
        if not isinstance(inputs, dict):
            raise ValueError(
                "RFDiffusionModel expects batched dict inputs or object-array DeepChem features."
            )

        input_tensors = [{
            "residue_types":
            torch.as_tensor(inputs["residue_types"],
                            dtype=torch.long,
                            device=self.device),
            "noisy_coordinates":
            torch.as_tensor(inputs["noisy_coordinates"],
                            dtype=torch.float32,
                            device=self.device),
            "residue_mask":
            torch.as_tensor(inputs["residue_mask"],
                            dtype=torch.float32,
                            device=self.device),
            "timesteps":
            torch.as_tensor(inputs["timesteps"],
                            dtype=torch.float32,
                            device=self.device)
        }]

        label_tensors = []
        if labels is not None:
            label_tensors = [
                torch.as_tensor(labels, dtype=torch.float32, device=self.device)
            ]

        weight_tensors = []
        if weights is not None:
            weight_tensors = [
                torch.as_tensor(weights, dtype=torch.float32, device=self.device)
            ]

        return input_tensors, label_tensors, weight_tensors

    def default_generator(
            self,
            dataset: Dataset,
            epochs: int = 1,
            mode: str = 'fit',
            deterministic: bool = True,
            pad_batches: bool = False) -> Iterable[Tuple[Dict[str, np.ndarray], List[np.ndarray], List[np.ndarray]]]:
        for epoch in range(epochs):
            logger.info("Starting training for epoch %d at %s", epoch,
                        datetime.datetime.now().ctime())
            for X_b, y_b, w_b, ids_b in dataset.iterbatches(
                    batch_size=self.batch_size,
                    deterministic=deterministic,
                    pad_batches=pad_batches):
                del ids_b
                features = stack_rfdiffusion_inputs(X_b)
                clean_coordinates = y_b.astype(np.float32)
                residue_mask = w_b.astype(np.float32)
                features["residue_mask"] = residue_mask

                if mode == 'fit':
                    timesteps = self._rng.uniform(self.min_timestep,
                                                  1.0,
                                                  size=clean_coordinates.shape[0]
                                                  ).astype(np.float32)
                    sigmas = self.sigma_min + timesteps * (self.sigma_max -
                                                           self.sigma_min)
                    noise = self._rng.normal(
                        size=clean_coordinates.shape).astype(np.float32)
                    noisy_coordinates = clean_coordinates + noise * sigmas[:,
                                                                           None,
                                                                           None] * residue_mask[
                                                                               :,
                                                                               :,
                                                                               None]
                else:
                    timesteps = np.zeros((clean_coordinates.shape[0],),
                                         dtype=np.float32)
                    noisy_coordinates = clean_coordinates

                model_inputs = {
                    "residue_types": features["residue_types"],
                    "noisy_coordinates": noisy_coordinates,
                    "residue_mask": residue_mask,
                    "timesteps": timesteps
                }
                yield model_inputs, [clean_coordinates], [residue_mask]

    def sample(self,
               residue_types: Union[np.ndarray, torch.Tensor],
               residue_mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
               num_steps: int = 25,
               initial_noise_scale: float = 1.0) -> np.ndarray:
        """Run the denoising sampler and return NumPy coordinates."""
        self._ensure_built()
        self.model.eval()
        with torch.no_grad():
            sampled_coordinates = self.model.sample(
                residue_types=residue_types,
                residue_mask=residue_mask,
                num_steps=num_steps,
                initial_noise_scale=initial_noise_scale)
        return sampled_coordinates.detach().cpu().numpy()
