"""Mixture-of-experts components used by Gefra."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph_encoders import GraphEncoder
from .riemannian_encoders import KappaGCN


class EuclideanExperts(nn.Module):
    """Run several independent Euclidean graph encoders in parallel."""

    def __init__(
        self,
        num_experts,
        input_dim,
        hidden_dim,
        activation,
        num_layers,
        backbone="mysage",
        normalize="batch",
        dropout=0.15,
    ):
        super().__init__()
        self.experts = nn.ModuleList()
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim

        for _ in range(num_experts):
            self.experts.append(
                GraphEncoder(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    activation=activation,
                    num_layers=num_layers,
                    backbone=backbone,
                    normalize=normalize,
                    dropout=dropout,
                )
            )

    def forward(self, x, edge_index, edge_attr=None):
        expert_outputs = []
        for expert in self.experts:
            out = expert(x, edge_index, edge_attr)
            expert_outputs.append(out.unsqueeze(-1))

        return torch.cat(expert_outputs, dim=-1)


class LinearGating(nn.Module):
    """Score each expert independently from its projected embedding."""

    def __init__(self, hidden_dim, feature_dim, dropout=0.15, **kwargs):
        super().__init__()
        self.fc1 = nn.Linear(feature_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """Return expert logits for inputs shaped `[N, feature_dim, M]`."""
        x = x.permute(0, 2, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x).squeeze(-1)

        return logits


class RiemannianExperts(nn.Module):
    """Run graph experts initialized across positive and negative curvatures."""

    def __init__(
        self, num_experts, input_dim, hidden_dim, dropout=0.15, **kwargs
    ):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        # Initialize independent expert manifolds away from zero curvature.
        ks = torch.linspace(-1, 1, num_experts + 2)[1:-1].tolist()
        self.experts = nn.ModuleList(
            [
                KappaGCN(k, input_dim, hidden_dim, hidden_dim, learnable=True)
                for k in ks
            ]
        )

    def forward(self, x, edge_index, edge_attr=None):
        expert_outputs = []
        for expert in self.experts:
            output = expert(x, edge_index)
            expert_outputs.append(output.unsqueeze(-1))

        return torch.cat(expert_outputs, dim=-1)
