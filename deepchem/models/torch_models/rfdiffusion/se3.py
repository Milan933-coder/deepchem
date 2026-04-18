"""SE(3)-aware structural updates for the DeepChem RFdiffusion starter."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:
    raise ImportError("These classes require PyTorch to be installed.")

from deepchem.models.torch_models.layers import MultilayerPerceptron


def _pair_mask(residue_mask: torch.Tensor) -> torch.Tensor:
    """Create a valid pair mask without self-edges."""
    valid = residue_mask > 0
    pair_mask = valid.unsqueeze(1) & valid.unsqueeze(2)
    eye = torch.eye(pair_mask.shape[-1],
                    device=pair_mask.device,
                    dtype=torch.bool).unsqueeze(0)
    return pair_mask & ~eye


class DeepChemSE3StructureModule(nn.Module):
    """Structure update block with an optional DeepChem SE(3) path.

    When `dgl` is available, scalar node states are updated with DeepChem's
    SE(3)-equivariant attention and convolution layers. Coordinate updates are
    then predicted from relative directions. If the environment lacks `dgl`,
    the module falls back to a lightweight pairwise update while keeping the
    same API.
    """

    def __init__(self,
                 node_dim: int,
                 pair_dim: int,
                 n_heads: int = 4,
                 num_layers: int = 2,
                 top_k: int = 16,
                 use_deepchem_se3: bool = True):
        super().__init__()
        self.node_dim = node_dim
        self.pair_dim = pair_dim
        self.top_k = top_k

        self.node_norm = nn.LayerNorm(node_dim)
        self.fallback_update = MultilayerPerceptron(d_input=2 * node_dim +
                                                    pair_dim,
                                                    d_hidden=(node_dim,),
                                                    d_output=node_dim,
                                                    activation_fn='silu')
        self.coordinate_gate = MultilayerPerceptron(d_input=2 * node_dim +
                                                    pair_dim + 1,
                                                    d_hidden=(node_dim,),
                                                    d_output=1,
                                                    activation_fn='silu')

        self.using_deepchem_se3 = False
        self._get_basis_and_r = None
        if use_deepchem_se3:
            try:
                import dgl  # noqa: F401
                from deepchem.models.torch_models.layers import Fiber
                from deepchem.models.torch_models.layers import SE3GraphConv
                from deepchem.models.torch_models.layers import SE3GraphNorm
                from deepchem.models.torch_models.layers import SE3ResidualAttention
                from deepchem.utils.equivariance_utils import get_equivariant_basis_and_r

                self._dgl = dgl
                self._Fiber = Fiber
                self._get_basis_and_r = get_equivariant_basis_and_r
                fiber = Fiber(dictionary={0: node_dim})
                blocks: List[nn.Module] = []
                for _ in range(num_layers):
                    blocks.append(
                        SE3ResidualAttention(fiber,
                                             fiber,
                                             edge_dim=pair_dim,
                                             div=2,
                                             n_heads=n_heads,
                                             skip='sum'))
                    blocks.append(SE3GraphNorm(fiber))
                self.se3_blocks = nn.ModuleList(blocks)
                self.se3_out = SE3GraphConv(fiber,
                                            fiber,
                                            self_interaction=True,
                                            edge_dim=pair_dim)
                self.using_deepchem_se3 = True
            except Exception:
                self.se3_blocks = nn.ModuleList()
                self.se3_out = None
        else:
            self.se3_blocks = nn.ModuleList()
            self.se3_out = None

    def _build_graph_batch(
            self, node_states: torch.Tensor, pair_features: torch.Tensor,
            coordinates: torch.Tensor,
            residue_mask: torch.Tensor) -> Tuple[object, torch.Tensor, List[torch.Tensor]]:
        """Build a batched DGL graph for the valid residues in each sample."""
        graphs = []
        flat_node_states = []
        valid_indices_per_sample: List[torch.Tensor] = []

        for batch_index in range(node_states.shape[0]):
            valid_indices = torch.nonzero(residue_mask[batch_index] > 0,
                                          as_tuple=False).squeeze(-1)
            if valid_indices.numel() == 0:
                valid_indices = torch.tensor([0],
                                             device=node_states.device,
                                             dtype=torch.long)
            valid_indices_per_sample.append(valid_indices)

            sample_coords = coordinates[batch_index].index_select(0, valid_indices)
            sample_pairs = pair_features[batch_index].index_select(
                0, valid_indices).index_select(1, valid_indices)
            flat_node_states.append(
                node_states[batch_index].index_select(0, valid_indices))
            n_nodes = sample_coords.shape[0]

            if n_nodes == 1:
                src = torch.tensor([0], device=coordinates.device)
                dst = torch.tensor([0], device=coordinates.device)
            else:
                distances = torch.cdist(sample_coords, sample_coords)
                if self.top_k <= 0 or self.top_k >= n_nodes:
                    adjacency = ~torch.eye(n_nodes,
                                           device=coordinates.device,
                                           dtype=torch.bool)
                    src, dst = torch.nonzero(adjacency, as_tuple=True)
                else:
                    neighbor_count = min(self.top_k + 1, n_nodes)
                    knn = torch.topk(distances,
                                     k=neighbor_count,
                                     largest=False).indices[:, 1:]
                    dst = torch.arange(n_nodes,
                                       device=coordinates.device).unsqueeze(1)
                    dst = dst.expand_as(knn).reshape(-1)
                    src = knn.reshape(-1)

            graph = self._dgl.graph((src, dst), num_nodes=n_nodes)
            graph = graph.to(coordinates.device)
            graph.edata['edge_attr'] = sample_coords.index_select(0, dst) - sample_coords.index_select(0, src)
            graph.edata['w'] = sample_pairs[dst, src]
            graphs.append(graph)

        batched_graph = self._dgl.batch(graphs).to(coordinates.device)
        return batched_graph, torch.cat(flat_node_states,
                                        dim=0), valid_indices_per_sample

    def _scatter_node_states(self, flat_states: torch.Tensor,
                             valid_indices_per_sample: List[torch.Tensor],
                             batch_size: int, n_residues: int) -> torch.Tensor:
        """Scatter flattened valid-node states back to padded tensors."""
        output = flat_states.new_zeros(batch_size, n_residues, flat_states.shape[-1])
        start = 0
        for batch_index, valid_indices in enumerate(valid_indices_per_sample):
            n_nodes = valid_indices.numel()
            output[batch_index, valid_indices] = flat_states[start:start + n_nodes]
            start += n_nodes
        return output

    def _se3_state_update(self, node_states: torch.Tensor,
                          pair_features: torch.Tensor, coordinates: torch.Tensor,
                          residue_mask: torch.Tensor) -> torch.Tensor:
        """Apply DeepChem SE(3) layers when DGL is available."""
        graph, flat_states, valid_indices = self._build_graph_batch(
            node_states, pair_features, coordinates, residue_mask)
        features: Dict[str, torch.Tensor] = {'0': flat_states.unsqueeze(-1)}
        basis, radii = self._get_basis_and_r(graph,
                                             max_degree=0,
                                             compute_gradients=False)
        for block in self.se3_blocks:
            features = block(features, G=graph, r=radii, basis=basis)
        features = self.se3_out(features, G=graph, r=radii, basis=basis)
        updated_flat_states = features['0'].squeeze(-1)
        updated_states = self._scatter_node_states(updated_flat_states,
                                                   valid_indices,
                                                   node_states.shape[0],
                                                   node_states.shape[1])
        return updated_states

    def _fallback_state_update(self, node_states: torch.Tensor,
                               pair_features: torch.Tensor,
                               residue_mask: torch.Tensor) -> torch.Tensor:
        """Apply a pairwise fallback update when DGL is unavailable."""
        pair_valid = _pair_mask(residue_mask).unsqueeze(-1).float()
        neighbor_states = node_states.unsqueeze(1).expand(-1, node_states.shape[1],
                                                          -1, -1)
        aggregated_context = torch.cat([neighbor_states, pair_features], dim=-1)
        aggregated_context = aggregated_context * pair_valid
        denominator = pair_valid.sum(dim=2).clamp(min=1.0)
        aggregated_context = aggregated_context.sum(dim=2) / denominator
        node_update = self.fallback_update(
            torch.cat([node_states, aggregated_context], dim=-1))
        return self.node_norm(node_states + node_update)

    def _coordinate_update(self, node_states: torch.Tensor,
                           pair_features: torch.Tensor,
                           coordinates: torch.Tensor,
                           residue_mask: torch.Tensor) -> torch.Tensor:
        """Update coordinates from pair-conditioned relative directions."""
        relative_coordinates = coordinates.unsqueeze(1) - coordinates.unsqueeze(2)
        distances = torch.linalg.norm(relative_coordinates, dim=-1)
        directions = relative_coordinates / distances.clamp(min=1e-6).unsqueeze(-1)

        pair_valid = _pair_mask(residue_mask).unsqueeze(-1).float()
        node_i = node_states.unsqueeze(2).expand(-1, -1, node_states.shape[1], -1)
        node_j = node_states.unsqueeze(1).expand(-1, node_states.shape[1], -1, -1)
        gate_input = torch.cat([node_i, node_j, pair_features,
                                distances.unsqueeze(-1)],
                               dim=-1)
        gates = torch.tanh(self.coordinate_gate(gate_input)) * pair_valid
        delta = (gates * directions).sum(dim=2)
        return coordinates + delta * residue_mask.unsqueeze(-1)

    def forward(self, node_states: torch.Tensor, pair_features: torch.Tensor,
                coordinates: torch.Tensor,
                residue_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.using_deepchem_se3:
            node_states = self._se3_state_update(node_states, pair_features,
                                                 coordinates, residue_mask)
        else:
            node_states = self._fallback_state_update(node_states, pair_features,
                                                      residue_mask)

        node_states = node_states * residue_mask.unsqueeze(-1)
        coordinates = self._coordinate_update(node_states, pair_features,
                                              coordinates, residue_mask)
        coordinates = coordinates * residue_mask.unsqueeze(-1)
        return node_states, coordinates
