"""Euclidean graph encoders used by Gefra."""

from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptTensor, OptPairTensor, Size


class EdgeAwareSAGEConv(MessagePassing):
    """GraphSAGE-style convolution that can add edge attributes to messages."""

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        aggr: str = "mean",
        normalize: bool = False,
        root_weight: bool = True,
        project: bool = False,
        bias: bool = True,
    ):
        super().__init__(aggr=aggr, flow="source_to_target", node_dim=0)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.root_weight = root_weight
        self.project = project

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        if project:
            self.lin = Linear(in_channels[0], in_channels[0], bias=True)

        self.lin_l = Linear(in_channels[0], out_channels, bias=bias)
        if root_weight:
            self.lin_r = Linear(in_channels[1], out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        if self.project:
            self.lin.reset_parameters()
        self.lin_l.reset_parameters()
        if self.root_weight:
            self.lin_r.reset_parameters()

    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        edge_attr: OptTensor = None,
        size: Size = None,
    ) -> Tensor:
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        if self.project:
            x = (self.lin(x[0]).relu(), x[1])

        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)
        out = self.lin_l(out)

        if self.root_weight:
            out = out + self.lin_r(x[1])

        if self.normalize:
            out = F.normalize(out, p=2.0, dim=-1)

        return out

    def message(self, x_j: Tensor, edge_attr: OptTensor) -> Tensor:
        if edge_attr is None:
            return x_j.relu()
        return (x_j + edge_attr).relu()


class GraphEncoder(nn.Module):
    """Stack a configurable sequence of Euclidean PyG convolution layers."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        activation,
        num_layers: int,
        backbone: str = "mysage",
        normalize: str = "none",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.activation = activation()
        self.dropout = nn.Dropout(dropout)
        self.normalize = normalize.lower()

        dims = [input_dim] + [hidden_dim] * num_layers
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            if backbone == "mysage":
                layer = EdgeAwareSAGEConv(
                    in_dim, out_dim, aggr="mean", root_weight=True
                )
            elif backbone == "sage":
                from torch_geometric.nn import SAGEConv

                layer = SAGEConv(in_dim, out_dim, aggr="mean", root_weight=True)
            elif backbone == "gat":
                from torch_geometric.nn import GATConv

                layer = GATConv(in_dim, out_dim, heads=1)
            elif backbone == "gcn":
                from torch_geometric.nn import GCNConv

                layer = GCNConv(in_dim, out_dim)
            elif backbone == "gin":
                from torch_geometric.nn import GINConv

                layer = GINConv(nn.Linear(in_dim, out_dim))
            else:
                raise ValueError(f"Unknown backbone: {backbone}")

            self.layers.append(layer)

            if self.normalize == "batch":
                self.norms.append(nn.BatchNorm1d(out_dim))
            elif self.normalize == "layer":
                self.norms.append(nn.LayerNorm(out_dim))
            else:
                self.norms.append(nn.Identity())

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
        for norm in self.norms:
            if hasattr(norm, "reset_parameters"):
                norm.reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: OptTensor = None):
        return self.encode(x, edge_index, edge_attr)

    def encode(self, x: Tensor, edge_index: Adj, edge_attr: OptTensor = None):
        z = x
        for i, layer in enumerate(self.layers):
            z = layer(z, edge_index)
            z = self.norms[i](z)
            if i < self.num_layers - 1:
                z = self.activation(z)
                z = self.dropout(z)
        return z


class InnerProductDecoder(nn.Module):
    """Decode edge probabilities from pairs of node embeddings."""

    def __init__(self, hidden_dim=None, output_dim=None):
        super().__init__()
        self.proj_z = hidden_dim is not None
        if self.proj_z:
            self.lin = nn.Linear(hidden_dim, output_dim)

    def forward(self, z: Tensor, edge_index: Tensor, sigmoid: bool = True) -> Tensor:
        if self.proj_z:
            z = self.lin(z)
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value

    def forward_all(self, z: Tensor, sigmoid: bool = True) -> Tensor:
        if self.proj_z:
            z = self.lin(z)
        adj = torch.matmul(z, z.t())
        return torch.sigmoid(adj) if sigmoid else adj
