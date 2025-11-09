import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_scatter import scatter_max, scatter_min
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange
from nets.egt import EGTNetwork
from vqvae.vq import VectorQuantize
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.nn import models
import torch_geometric
import utils.logger as log
from torch_geometric.nn import TransformerConv
from nets.unimp import UniMP


class InnerProductDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, embeddings):
        adj_matrix = torch.sigmoid(torch.mm(embeddings, embeddings.t()))
        return adj_matrix


class AdjacencyReconstructor(torch.nn.Module):
    def __init__(self, embedding_dim, hidden_dim=64):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim * 2, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1),
            torch.nn.Sigmoid(),
        )

    def forward(self, embeddings, batch, mask):
        device = embeddings.device
        num_nodes = embeddings.size(0)

        edge_index = torch.nonzero(mask).t()  # (2, E)

        node_i = embeddings[edge_index[0]]  # (E, embedding_dim)
        node_j = embeddings[edge_index[1]]  # (E, embedding_dim)

        pairs = torch.cat([node_i, node_j], dim=-1)  # (E, 2*embedding_dim)

        edge_pred = self.mlp(pairs).squeeze(-1)  # (E,)

        adj_matrix = torch.zeros(num_nodes, num_nodes, device=device)
        adj_matrix[edge_index[0], edge_index[1]] = edge_pred

        adj_matrix = (adj_matrix + adj_matrix.t()) / 2

        return adj_matrix


class EdgeReconstructor(torch.nn.Module):
    def __init__(self, embedding_dim, hidden_dim=64, edge_dim=4):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim * 2, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, edge_dim + 1),
        )
        self.edge_dim = edge_dim

    def forward(self, embeddings, batch, mask):
        device = embeddings.device
        num_nodes = embeddings.size(0)

        edge_index = torch.nonzero(mask).t()  # (2, E)

        node_i = embeddings[edge_index[0]]  # (E, embedding_dim)
        node_j = embeddings[edge_index[1]]  # (E, embedding_dim)

        pairs1 = torch.cat([node_i, node_j], dim=-1)  # (E, 2*embedding_dim)
        pairs2 = torch.cat([node_j, node_i], dim=-1)  # (E, 2*embedding_dim)
        edge_pred = (self.mlp(pairs1) + self.mlp(pairs2)) / 2  # (E, edge_dim + 1)

        adj_matrix = torch.zeros(
            num_nodes,
            num_nodes,
            self.edge_dim + 1,
            device=device,
            dtype=edge_pred.dtype,
        )
        adj_matrix[:, :, -1] = 1.0  # all edges are set to non-exists label
        adj_matrix[edge_index[0], edge_index[1]] = edge_pred

        return adj_matrix


class EdgePredictDecoder(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, edge_dim):
        super().__init__()

        self.num_layers = num_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.node_mlp = models.MLP([input_dim, hidden_dim, output_dim])

        self.edge_mlp = models.MLP([input_dim, hidden_dim, 32])
        self.adj_decoder = EdgeReconstructor(32, edge_dim=edge_dim)

    def forward(self, quantized, batch, mask):
        """forward pass of decoder

        Args:
            quantized (_type_): quantized node features for adj reconstruction (N, D) where N = sum of all nodes in batch
            mask (_type_): mask for adj reconstruction (N, N) where N = sum of all nodes in batch

        Returns:
            Tensor: quantized_node (N, D), adj_quantized (N, N)
        """
        quantized_node = self.node_mlp(quantized)

        quantized_edge = self.edge_mlp(quantized)
        adj_matrix = self.adj_decoder(quantized_edge, batch, mask)  # (E, edge_dim)
        # adj_quantized = adj_quantized * mask

        return quantized_node, adj_matrix


class EGTDecoder(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, edge_dim):
        super().__init__()

        self.num_layers = num_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.edge_mlp = models.MLP([input_dim, hidden_dim, 16])
        self.adj_decoder = EdgeReconstructor(16, edge_dim=edge_dim)

        self.net = EGTNetwork(
            input_dim,
            hidden_dim,
            output_dim,
            edge_dim + 1,
            num_layers,
            0.1,
            beta=True,
            heads=8,
        )

    def forward(self, quantized, batch, mask):
        """forward pass of decoder

        Args:
            quantized (_type_): quantized node features for adj reconstruction (N, D) where N = sum of all nodes in batch
            batch: pyg batch information
            mask (_type_): mask for adj reconstruction (N, N) where N = sum of all nodes in batch

        Returns:
            Tensor: quantized_node (N, D), adj_quantized (N, N)
        """
        # 1. calculate initial edge features
        quantized_edge = self.edge_mlp(quantized)
        edge_feat = self.adj_decoder(quantized_edge, batch, mask)

        # remove self loop
        mask.diagonal().fill_(0)
        edge_index = torch.nonzero(mask).t()
        edge_attr = edge_feat[edge_index[0], edge_index[1], :]

        x, edge_attr, mask_nodes = self.net(quantized, edge_index, batch, edge_attr)
        x = x[mask_nodes]
        edge_attr = (edge_attr + edge_attr.transpose(1, 2)) / 2

        return x, edge_attr, mask_nodes


class EGTDecoderRoPE(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, edge_dim):
        super().__init__()

        self.num_layers = num_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.edge_mlp = models.MLP([input_dim, hidden_dim, 16])
        self.adj_decoder = EdgeReconstructor(16, edge_dim=edge_dim)

        self.net = EGTNetwork(
            input_dim,
            hidden_dim,
            output_dim,
            edge_dim + 1,
            num_layers,
            0.1,
            beta=True,
            heads=8,
        )

        self.rope = RoPE(input_dim)

    def forward(self, quantized, batch, mask):
        """forward pass of decoder

        Args:
            quantized (_type_): quantized node features for adj reconstruction (N, D) where N = sum of all nodes in batch
            batch: pyg batch information
            mask (_type_): mask for adj reconstruction (N, N) where N = sum of all nodes in batch

        Returns:
            Tensor: quantized_node (N, D), adj_quantized (N, N)
        """
        num_nodes = quantized.size(0)
        node_arange = torch.arange(num_nodes, device=quantized.device)
        _uc_val, uc_ptr, uc_counts = torch.unique_consecutive(
            batch, return_inverse=True, return_counts=True
        )
        dense_graph_id_starts = torch.cat(
            (torch.tensor([0], device=batch.device), uc_counts[:-1].cumsum(0))
        )
        positions = node_arange - dense_graph_id_starts[uc_ptr]

        quantized = self.rope(quantized, positions)

        # 1. calculate initial edge features
        quantized_edge = self.edge_mlp(quantized)
        edge_feat = self.adj_decoder(quantized_edge, batch, mask)

        # remove self loop
        mask.diagonal().fill_(0)
        edge_index = torch.nonzero(mask).t()
        edge_attr = edge_feat[edge_index[0], edge_index[1], :]

        x, edge_attr, mask_nodes = self.net(quantized, edge_index, batch, edge_attr)
        x = x[mask_nodes]
        edge_attr = (edge_attr + edge_attr.transpose(1, 2)) / 2

        return x, edge_attr, mask_nodes


class RoPE(nn.Module):
    def __init__(self, dim: int, theta_base: float = 10000.0):
        super().__init__()
        assert dim % 2 == 0,
        self.dim = dim
        self.theta_base = theta_base

        inv_freq = 1.0 / (
            theta_base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )
        self.register_buffer(
            "inv_freq", inv_freq.unsqueeze(0), persistent=False
        )  # Shape: [1, D/2]

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        freqs = positions.float().unsqueeze(1) * self.inv_freq

        cos_val = torch.cos(freqs)
        sin_val = torch.sin(freqs)

        half_dim = self.dim // 2
        x1 = x[..., :half_dim]
        x2 = x[..., half_dim:]

        rotated_x1 = x1 * cos_val - x2 * sin_val
        rotated_x2 = x1 * sin_val + x2 * cos_val

        return torch.cat((rotated_x1, rotated_x2), dim=-1)
