"""Constant-curvature graph encoders used by Gefra experts."""

import geoopt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


class KappaLinear(nn.Module):
    """Apply a Mobius linear transform on a stereographic manifold."""

    def __init__(
        self,
        manifold: geoopt.Stereographic,
        in_dim: int,
        out_dim: int,
        dropout: float = 0.0,
        use_bias: bool = True,
    ):
        super().__init__()
        self.manifold = manifold
        self.dropout = dropout
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.Tensor(out_dim, in_dim))
        self.bias = nn.Parameter(torch.Tensor(out_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.constant_(self.bias, 0)

    def forward(self, x):
        drop_weight = F.dropout(self.weight, self.dropout, training=self.training)
        result = self.manifold.mobius_matvec(drop_weight, x, project=True)
        if self.use_bias:
            bias = self.manifold.proju(
                self.manifold.origin(self.bias.shape), self.bias
            )
            kappa_bias = self.manifold.expmap0(bias, project=True)
            result = self.manifold.mobius_add(result, kappa_bias, project=True)
        return result


class KappaGCNConv(MessagePassing):
    """Perform normalized graph convolution in a stereographic model."""

    def __init__(self, k, in_dim: int, out_dim: int, learnable=True):
        super().__init__(aggr="add")
        self.manifold = geoopt.Stereographic(k=k, learnable=learnable)
        self.lin = KappaLinear(
            manifold=self.manifold,
            in_dim=in_dim,
            out_dim=out_dim,
            use_bias=True,
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        edge_index, _ = add_self_loops(edge_index)
        x = self.lin(x)

        x_tan0 = self.manifold.logmap0(x)
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        out = self.propagate(edge_index, x=x_tan0, norm=norm)
        return self.manifold.expmap0(out, project=True)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class KappaGCN(nn.Module):
    """Encode graph features with two learnable-curvature GCN layers."""

    def __init__(
        self,
        k,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        learnable: bool = True,
    ):
        super().__init__()
        self.manifold = geoopt.Stereographic(k=k, learnable=learnable)
        self.encoder1 = KappaGCNConv(
            k, in_dim, hidden_dim, learnable=learnable
        )
        self.encoder2 = KappaGCNConv(
            k, hidden_dim, out_dim, learnable=learnable
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, **kwargs):
        x = self.manifold.proju(self.manifold.origin(x.shape), x)
        x = self.manifold.expmap0(x, project=True)
        h = self.encoder1(x, edge_index)
        return self.encoder2(h, edge_index)

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor):
        x = self.manifold.proju(self.manifold.origin(x.shape), x)
        x = self.manifold.expmap0(x, project=True)
        h = self.encoder1(x, edge_index)
        z = self.encoder2(h, edge_index)
        return self.manifold.logmap0(z)

    def embed(self, x, edge_index):
        x = self.manifold.proju(self.manifold.origin(x.shape), x)
        x = self.manifold.expmap0(x, project=True)
        h = self.encoder1(x, edge_index)
        return self.manifold.logmap0(h)
