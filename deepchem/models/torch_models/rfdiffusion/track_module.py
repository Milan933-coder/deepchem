"""Track-style blocks for the DeepChem RFdiffusion starter."""

from __future__ import annotations

from typing import Optional, Tuple

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:
    raise ImportError("These classes require PyTorch to be installed.")

from deepchem.models.torch_models.layers import MultilayerPerceptron
from deepchem.models.torch_models.rfdiffusion.se3 import DeepChemSE3StructureModule


def pairwise_rbf(coordinates: torch.Tensor,
                 num_rbf: int = 36,
                 rbf_min: float = 0.0,
                 rbf_max: float = 20.0) -> torch.Tensor:
    """Encode pairwise distances with Gaussian radial basis functions."""
    distances = torch.cdist(coordinates, coordinates)
    centers = torch.linspace(rbf_min,
                             rbf_max,
                             steps=num_rbf,
                             device=coordinates.device)
    widths = (rbf_max - rbf_min) / max(num_rbf - 1, 1)
    return torch.exp(-((distances.unsqueeze(-1) - centers) / max(widths, 1e-6))**2)


def sequence_separation(residue_indices: torch.Tensor) -> torch.Tensor:
    """Compute signed residue-index offsets for pair features."""
    offsets = residue_indices[:, :, None] - residue_indices[:, None, :]
    return offsets.float().unsqueeze(-1)


class MSAPairStr2MSA(nn.Module):
    """Update MSA states using pair and structure context."""

    def __init__(self,
                 d_msa: int = 128,
                 d_pair: int = 64,
                 d_state: int = 64,
                 n_head: int = 4,
                 d_rbf: int = 36,
                 p_drop: float = 0.1):
        super().__init__()
        self.msa_norm = nn.LayerNorm(d_msa)
        self.state_norm = nn.LayerNorm(d_state)
        self.row_attention = nn.MultiheadAttention(embed_dim=d_msa,
                                                   num_heads=n_head,
                                                   dropout=p_drop,
                                                   batch_first=True)
        self.column_attention = nn.MultiheadAttention(embed_dim=d_msa,
                                                      num_heads=n_head,
                                                      dropout=p_drop,
                                                      batch_first=True)
        self.pair_to_msa = nn.Linear(d_pair + d_rbf, d_msa)
        self.state_to_msa = nn.Linear(d_state, d_msa)
        self.ff_norm = nn.LayerNorm(d_msa)
        self.ff = MultilayerPerceptron(d_input=d_msa,
                                       d_hidden=(4 * d_msa,),
                                       d_output=d_msa,
                                       activation_fn='gelu')

    def forward(self, msa: torch.Tensor, pair: torch.Tensor,
                rbf_feat: torch.Tensor, state: torch.Tensor,
                residue_mask: torch.Tensor) -> torch.Tensor:
        batch_size, n_sequences, n_residues, d_msa = msa.shape
        msa_input = self.msa_norm(msa)

        pair_context = self.pair_to_msa(torch.cat([pair, rbf_feat], dim=-1))
        pair_context = pair_context.mean(dim=2, keepdim=False)
        state_context = self.state_to_msa(self.state_norm(state))
        msa_input[:, 0] = msa_input[:, 0] + pair_context + state_context

        row_input = msa_input.reshape(batch_size * n_sequences, n_residues, d_msa)
        key_padding_mask = residue_mask <= 0
        row_padding = key_padding_mask.repeat_interleave(n_sequences, dim=0)
        row_output, _ = self.row_attention(row_input,
                                           row_input,
                                           row_input,
                                           key_padding_mask=row_padding)
        msa = msa + row_output.reshape(batch_size, n_sequences, n_residues, d_msa)

        if n_sequences > 1:
            column_input = self.msa_norm(msa).permute(0, 2, 1, 3)
            column_input = column_input.reshape(batch_size * n_residues,
                                                n_sequences, d_msa)
            column_output, _ = self.column_attention(column_input, column_input,
                                                     column_input)
            column_output = column_output.reshape(batch_size, n_residues,
                                                  n_sequences, d_msa)
            msa = msa + column_output.permute(0, 2, 1, 3)

        msa = msa + self.ff(self.ff_norm(msa))
        return msa * residue_mask[:, None, :, None]


class MSA2Pair(nn.Module):
    """Extract pair features from MSA coevolution-style outer products."""

    def __init__(self,
                 d_msa: int = 128,
                 d_pair: int = 64,
                 d_hidden: int = 32):
        super().__init__()
        self.norm = nn.LayerNorm(d_msa)
        self.proj_left = nn.Linear(d_msa, d_hidden)
        self.proj_right = nn.Linear(d_msa, d_hidden)
        self.proj_out = nn.Linear(d_hidden * d_hidden, d_pair)

    def forward(self, msa: torch.Tensor, pair: torch.Tensor,
                residue_mask: torch.Tensor) -> torch.Tensor:
        batch_size, n_sequences, n_residues = msa.shape[:3]
        msa = self.norm(msa)
        left = self.proj_left(msa)
        right = self.proj_right(msa) / float(max(n_sequences, 1))
        outer = torch.einsum('bnli,bnmj->blmij', left, right)
        pair = pair + self.proj_out(outer.reshape(batch_size, n_residues,
                                                  n_residues, -1))
        pair_mask = residue_mask[:, :, None] * residue_mask[:, None, :]
        return pair * pair_mask.unsqueeze(-1)


class PairStr2Pair(nn.Module):
    """Update pair features with axial attention and distance context."""

    def __init__(self,
                 d_pair: int = 64,
                 n_head: int = 4,
                 d_rbf: int = 36,
                 p_drop: float = 0.1):
        super().__init__()
        self.input_norm = nn.LayerNorm(d_pair)
        self.rbf_projection = nn.Linear(d_rbf, d_pair)
        self.row_attention = nn.MultiheadAttention(embed_dim=d_pair,
                                                   num_heads=n_head,
                                                   dropout=p_drop,
                                                   batch_first=True)
        self.column_attention = nn.MultiheadAttention(embed_dim=d_pair,
                                                      num_heads=n_head,
                                                      dropout=p_drop,
                                                      batch_first=True)
        self.ff_norm = nn.LayerNorm(d_pair)
        self.ff = MultilayerPerceptron(d_input=d_pair,
                                       d_hidden=(2 * d_pair,),
                                       d_output=d_pair,
                                       activation_fn='gelu')

    def forward(self, pair: torch.Tensor, rbf_feat: torch.Tensor,
                residue_mask: torch.Tensor) -> torch.Tensor:
        batch_size, n_residues = pair.shape[:2]
        pair_input = self.input_norm(pair) + self.rbf_projection(rbf_feat)
        key_padding_mask = residue_mask <= 0
        row_padding = key_padding_mask.repeat_interleave(n_residues, dim=0)

        row_input = pair_input.reshape(batch_size * n_residues, n_residues, -1)
        row_output, _ = self.row_attention(row_input,
                                           row_input,
                                           row_input,
                                           key_padding_mask=row_padding)
        row_output = row_output.reshape(batch_size, n_residues, n_residues, -1)

        column_input = pair_input.permute(0, 2, 1, 3).reshape(
            batch_size * n_residues, n_residues, -1)
        column_output, _ = self.column_attention(column_input,
                                                 column_input,
                                                 column_input,
                                                 key_padding_mask=row_padding)
        column_output = column_output.reshape(batch_size, n_residues, n_residues,
                                              -1).permute(0, 2, 1, 3)

        pair = pair + row_output + column_output
        pair = pair + self.ff(self.ff_norm(pair))
        pair_mask = residue_mask[:, :, None] * residue_mask[:, None, :]
        return pair * pair_mask.unsqueeze(-1)


class SCPred(nn.Module):
    """Predict coarse torsion-style outputs from sequence and state features."""

    def __init__(self,
                 d_msa: int = 128,
                 d_state: int = 64,
                 d_hidden: int = 128):
        super().__init__()
        self.seq_norm = nn.LayerNorm(d_msa)
        self.state_norm = nn.LayerNorm(d_state)
        self.seq_projection = nn.Linear(d_msa, d_hidden)
        self.state_projection = nn.Linear(d_state, d_hidden)
        self.residual = MultilayerPerceptron(d_input=d_hidden,
                                             d_hidden=(d_hidden,),
                                             d_output=d_hidden,
                                             activation_fn='gelu')
        self.output = nn.Linear(d_hidden, 20)

    def forward(self, seq: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        hidden = self.seq_projection(self.seq_norm(seq)) + self.state_projection(
            self.state_norm(state))
        hidden = hidden + self.residual(hidden)
        return self.output(hidden).view(seq.shape[0], seq.shape[1], 10, 2)


class Str2Str(nn.Module):
    """Update structure using DeepChem SE(3) primitives and pair features."""

    def __init__(self,
                 d_msa: int = 128,
                 d_pair: int = 64,
                 d_state: int = 64,
                 d_rbf: int = 36,
                 p_drop: float = 0.1,
                 n_heads: int = 4,
                 num_layers: int = 2,
                 top_k: int = 16,
                 use_deepchem_se3: bool = True):
        super().__init__()
        self.msa_norm = nn.LayerNorm(d_msa)
        self.pair_norm = nn.LayerNorm(d_pair)
        self.state_norm = nn.LayerNorm(d_state)
        self.node_projection = nn.Linear(d_msa + d_state, d_state)
        self.edge_projection = nn.Linear(d_pair + d_rbf + 1, d_pair)
        self.structure_module = DeepChemSE3StructureModule(
            node_dim=d_state,
            pair_dim=d_pair,
            n_heads=n_heads,
            num_layers=num_layers,
            top_k=top_k,
            use_deepchem_se3=use_deepchem_se3)
        self.sc_predictor = SCPred(d_msa=d_msa, d_state=d_state)
        self.dropout = nn.Dropout(p_drop)

    def forward(self, msa: torch.Tensor, pair: torch.Tensor,
                coordinates: torch.Tensor, state: torch.Tensor,
                residue_mask: torch.Tensor,
                residue_indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rbf_feat = pairwise_rbf(coordinates)
        seqsep = sequence_separation(residue_indices)

        node_features = torch.cat(
            [self.msa_norm(msa[:, 0]), self.state_norm(state)], dim=-1)
        node_features = self.dropout(self.node_projection(node_features))

        edge_features = torch.cat(
            [self.pair_norm(pair), rbf_feat, seqsep], dim=-1)
        edge_features = self.dropout(self.edge_projection(edge_features))

        state, coordinates = self.structure_module(node_features, edge_features,
                                                   coordinates, residue_mask)
        alpha = self.sc_predictor(msa[:, 0], state)
        return coordinates, state, alpha


class IterBlock(nn.Module):
    """A compact DeepChem-side analogue of an RFdiffusion track block."""

    def __init__(self,
                 d_msa: int = 128,
                 d_pair: int = 64,
                 d_state: int = 64,
                 n_head_msa: int = 4,
                 n_head_pair: int = 4,
                 d_rbf: int = 36,
                 p_drop: float = 0.1,
                 use_deepchem_se3: bool = True,
                 se3_top_k: int = 16):
        super().__init__()
        self.msa2msa = MSAPairStr2MSA(d_msa=d_msa,
                                      d_pair=d_pair,
                                      d_state=d_state,
                                      n_head=n_head_msa,
                                      d_rbf=d_rbf,
                                      p_drop=p_drop)
        self.msa2pair = MSA2Pair(d_msa=d_msa,
                                 d_pair=d_pair,
                                 d_hidden=max(d_pair // 2, 8))
        self.pair2pair = PairStr2Pair(d_pair=d_pair,
                                      n_head=n_head_pair,
                                      d_rbf=d_rbf,
                                      p_drop=p_drop)
        self.str2str = Str2Str(d_msa=d_msa,
                               d_pair=d_pair,
                               d_state=d_state,
                               d_rbf=d_rbf,
                               p_drop=p_drop,
                               n_heads=n_head_pair,
                               top_k=se3_top_k,
                               use_deepchem_se3=use_deepchem_se3)
        self.d_rbf = d_rbf

    def forward(self, msa: torch.Tensor, pair: torch.Tensor,
                coordinates: torch.Tensor, state: torch.Tensor,
                residue_mask: torch.Tensor,
                residue_indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        rbf_feat = pairwise_rbf(coordinates, num_rbf=self.d_rbf)
        msa = self.msa2msa(msa, pair, rbf_feat, state, residue_mask)
        pair = self.msa2pair(msa, pair, residue_mask)
        pair = self.pair2pair(pair, rbf_feat, residue_mask)
        coordinates, state, alpha = self.str2str(msa, pair, coordinates, state,
                                                 residue_mask, residue_indices)
        return msa, pair, coordinates, state, alpha


class IterativeSimulator(nn.Module):
    """Run extra, main, and refinement track blocks over protein features."""

    def __init__(self,
                 n_extra_block: int = 1,
                 n_main_block: int = 2,
                 n_ref_block: int = 1,
                 d_msa: int = 128,
                 d_msa_full: int = 64,
                 d_pair: int = 64,
                 d_state: int = 64,
                 n_head_msa: int = 4,
                 n_head_pair: int = 4,
                 d_rbf: int = 36,
                 p_drop: float = 0.1,
                 use_deepchem_se3: bool = True,
                 se3_top_k: int = 16):
        super().__init__()
        self.extra_block = nn.ModuleList([
            IterBlock(d_msa=d_msa_full,
                      d_pair=d_pair,
                      d_state=d_state,
                      n_head_msa=n_head_msa,
                      n_head_pair=n_head_pair,
                      d_rbf=d_rbf,
                      p_drop=p_drop,
                      use_deepchem_se3=use_deepchem_se3,
                      se3_top_k=se3_top_k) for _ in range(n_extra_block)
        ])
        self.main_block = nn.ModuleList([
            IterBlock(d_msa=d_msa,
                      d_pair=d_pair,
                      d_state=d_state,
                      n_head_msa=n_head_msa,
                      n_head_pair=n_head_pair,
                      d_rbf=d_rbf,
                      p_drop=p_drop,
                      use_deepchem_se3=use_deepchem_se3,
                      se3_top_k=se3_top_k) for _ in range(n_main_block)
        ])
        self.refiner = nn.ModuleList([
            Str2Str(d_msa=d_msa,
                    d_pair=d_pair,
                    d_state=d_state,
                    d_rbf=d_rbf,
                    p_drop=p_drop,
                    n_heads=n_head_pair,
                    top_k=se3_top_k,
                    use_deepchem_se3=use_deepchem_se3)
            for _ in range(n_ref_block)
        ])

    def uses_deepchem_se3(self) -> bool:
        """Report whether any structural block is using DeepChem SE(3)."""
        blocks = list(self.extra_block) + list(self.main_block)
        for block in blocks:
            if block.str2str.structure_module.using_deepchem_se3:
                return True
        for refiner in self.refiner:
            if refiner.structure_module.using_deepchem_se3:
                return True
        return False

    def forward(self, msa: torch.Tensor, msa_full: torch.Tensor,
                pair: torch.Tensor, coordinates: torch.Tensor,
                state: torch.Tensor, residue_mask: torch.Tensor,
                residue_indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        coordinate_trajectory = []
        torsion_trajectory = []

        for block in self.extra_block:
            msa_full, pair, coordinates, state, alpha = block(
                msa_full, pair, coordinates, state, residue_mask, residue_indices)
            coordinate_trajectory.append(coordinates)
            torsion_trajectory.append(alpha)

        for block in self.main_block:
            msa, pair, coordinates, state, alpha = block(
                msa, pair, coordinates, state, residue_mask, residue_indices)
            coordinate_trajectory.append(coordinates)
            torsion_trajectory.append(alpha)

        for refiner in self.refiner:
            coordinates, state, alpha = refiner(msa, pair, coordinates, state,
                                                residue_mask, residue_indices)
            coordinate_trajectory.append(coordinates)
            torsion_trajectory.append(alpha)

        if len(coordinate_trajectory) == 0:
            coordinate_trajectory.append(coordinates)
            torsion_trajectory.append(torch.zeros_like(state[..., :20]).view(
                state.shape[0], state.shape[1], 10, 2))

        return msa, pair, torch.stack(coordinate_trajectory,
                                      dim=0), torch.stack(torsion_trajectory,
                                                          dim=0), state
