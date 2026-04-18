"""Embedding layers for the DeepChem RFdiffusion starter."""

from __future__ import annotations

import math
from typing import Optional, Tuple

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:
    raise ImportError("These classes require PyTorch to be installed.")

from deepchem.models.torch_models.layers import MultilayerPerceptron


class SinusoidalTimeEmbedding(nn.Module):
    """Embed diffusion timesteps with sinusoidal features."""

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.embedding_dim // 2
        if half_dim == 0:
            return timesteps.unsqueeze(-1)

        frequency_scale = math.log(10000.0) / max(half_dim - 1, 1)
        frequencies = torch.exp(
            -frequency_scale *
            torch.arange(half_dim, device=timesteps.device, dtype=torch.float32))
        angles = timesteps.float().unsqueeze(-1) * frequencies.unsqueeze(0)
        embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        if embedding.shape[-1] < self.embedding_dim:
            embedding = torch.nn.functional.pad(
                embedding, (0, self.embedding_dim - embedding.shape[-1]))
        return embedding


class RelativePositionEmbedding(nn.Module):
    """Add relative positional encodings to residue-pair features."""

    def __init__(self,
                 d_pair: int,
                 max_relative_index: int = 32,
                 p_drop: float = 0.1):
        super().__init__()
        self.max_relative_index = max_relative_index
        self.embedding = nn.Embedding(2 * max_relative_index + 1, d_pair)
        self.dropout = nn.Dropout(p_drop)

    def forward(self, pair: torch.Tensor,
                residue_indices: torch.Tensor) -> torch.Tensor:
        sequence_offsets = residue_indices[:, :, None] - residue_indices[:, None,
                                                                          :]
        sequence_offsets = torch.clamp(sequence_offsets,
                                       -self.max_relative_index,
                                       self.max_relative_index)
        bins = sequence_offsets + self.max_relative_index
        pair = pair + self.embedding(bins.long())
        return self.dropout(pair)


class RFInputEmbedding(nn.Module):
    """Construct latent MSA, extra MSA, pair, and state embeddings.

    This is a compact DeepChem-side analogue of the upstream RFdiffusion
    embedding stack. It starts from residue identities and timesteps, keeping
    the input contract simple enough for a starter PR.
    """

    def __init__(self,
                 num_residue_types: int = 21,
                 d_msa: int = 128,
                 d_msa_full: int = 64,
                 d_pair: int = 64,
                 d_state: int = 64,
                 max_relative_index: int = 32,
                 p_drop: float = 0.1):
        super().__init__()
        self.num_residue_types = num_residue_types

        self.latent_embedding = nn.Embedding(num_residue_types, d_msa)
        self.full_embedding = nn.Embedding(num_residue_types, d_msa_full)
        self.left_pair_embedding = nn.Embedding(num_residue_types, d_pair)
        self.right_pair_embedding = nn.Embedding(num_residue_types, d_pair)
        self.state_embedding = nn.Embedding(num_residue_types, d_state)

        self.time_embedding = SinusoidalTimeEmbedding(max(d_msa, d_pair, d_state,
                                                          d_msa_full))
        self.latent_time = MultilayerPerceptron(d_input=self.time_embedding.embedding_dim,
                                                d_hidden=(d_msa,),
                                                d_output=d_msa,
                                                activation_fn='silu')
        self.full_time = MultilayerPerceptron(d_input=self.time_embedding.embedding_dim,
                                              d_hidden=(d_msa_full,),
                                              d_output=d_msa_full,
                                              activation_fn='silu')
        self.pair_time = MultilayerPerceptron(d_input=self.time_embedding.embedding_dim,
                                              d_hidden=(d_pair,),
                                              d_output=d_pair,
                                              activation_fn='silu')
        self.state_time = MultilayerPerceptron(d_input=self.time_embedding.embedding_dim,
                                               d_hidden=(d_state,),
                                               d_output=d_state,
                                               activation_fn='silu')

        self.position = RelativePositionEmbedding(d_pair=d_pair,
                                                  max_relative_index=max_relative_index,
                                                  p_drop=p_drop)
        self.dropout = nn.Dropout(p_drop)

    def forward(
            self,
            residue_types: torch.Tensor,
            residue_mask: torch.Tensor,
            timesteps: torch.Tensor,
            residue_indices: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        residue_types = residue_types.long().clamp(
            min=0, max=self.num_residue_types - 1)
        residue_mask = residue_mask.float()
        batch_size, n_residues = residue_types.shape

        if residue_indices is None:
            residue_indices = torch.arange(n_residues,
                                           device=residue_types.device).unsqueeze(0)
            residue_indices = residue_indices.expand(batch_size, -1)

        time_features = self.time_embedding(timesteps.float())

        latent_msa = self.latent_embedding(residue_types).unsqueeze(1)
        latent_msa = latent_msa + self.latent_time(time_features).unsqueeze(1).unsqueeze(1)
        latent_msa = self.dropout(latent_msa)

        full_msa = self.full_embedding(residue_types).unsqueeze(1)
        full_msa = full_msa + self.full_time(time_features).unsqueeze(1).unsqueeze(1)
        full_msa = self.dropout(full_msa)

        left = self.left_pair_embedding(residue_types)
        right = self.right_pair_embedding(residue_types)
        pair = left.unsqueeze(2) + right.unsqueeze(1)
        pair = pair + self.pair_time(time_features).unsqueeze(1).unsqueeze(1)
        pair = self.position(pair, residue_indices)

        state = self.state_embedding(residue_types)
        state = state + self.state_time(time_features).unsqueeze(1)
        state = self.dropout(state)

        pair_mask = residue_mask[:, :, None] * residue_mask[:, None, :]
        latent_msa = latent_msa * residue_mask[:, None, :, None]
        full_msa = full_msa * residue_mask[:, None, :, None]
        pair = pair * pair_mask.unsqueeze(-1)
        state = state * residue_mask.unsqueeze(-1)

        return latent_msa, full_msa, pair, state, residue_indices
