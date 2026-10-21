"""Prompt-based expert readouts used during Gefra fine-tuning."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class RiemannianPromptReadout(nn.Module):
    """Aggregate global experts with content and curvature-aware attention."""

    def __init__(
        self,
        embed_dim: int,
        num_experts: int,
        num_prompts: int = 4,
        lambda_curv: float = 1.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_experts = num_experts
        self.num_prompts = num_prompts
        self.lambda_curv = lambda_curv

        self.prompt_tokens = nn.Parameter(
            torch.randn(num_prompts, embed_dim) * 0.02
        )
        self.prompt_curvatures = nn.Parameter(torch.zeros(num_prompts))

        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self, expert_embeds: torch.Tensor, expert_curvatures: torch.Tensor
    ):
        """Return the readout, attention, and curvature penalty tensors."""
        batch_size, num_experts, embed_dim = expert_embeds.shape
        prompt = self.prompt_tokens.unsqueeze(0).expand(batch_size, -1, -1)

        query = self.W_q(prompt)
        key = self.W_k(expert_embeds)
        value = self.W_v(expert_embeds)
        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(embed_dim)

        if expert_curvatures.dim() == 1:
            kappa = expert_curvatures.view(1, 1, num_experts)
        else:
            kappa = expert_curvatures.unsqueeze(1)

        eta = self.prompt_curvatures.view(1, self.num_prompts, 1)
        curvature_penalty = (eta - kappa) ** 2
        scores = scores - self.lambda_curv * curvature_penalty

        attention = self.dropout(F.softmax(scores, dim=-1))
        prompt_outputs = torch.matmul(attention, value)
        readout = self.out_proj(prompt_outputs.mean(dim=1))
        return readout, attention, curvature_penalty


class LearnableWeightReadout(nn.Module):
    """Aggregate experts with learned scalar weights for the readout ablation."""

    def __init__(self, embed_dim: int, num_experts: int, **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_experts = num_experts
        self.expert_weights = nn.Parameter(torch.zeros(num_experts))
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        # Preserve the regularization interface used by the training loop.
        self.prompt_tokens = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(
        self, expert_embeds: torch.Tensor, expert_curvatures: torch.Tensor
    ):
        weights = F.softmax(self.expert_weights, dim=0)
        readout = (expert_embeds * weights.view(1, -1, 1)).sum(dim=1)
        readout = self.out_proj(readout)
        attention = weights.view(1, 1, -1).expand(
            expert_embeds.size(0), 1, -1
        )
        curvature_penalty = torch.zeros_like(attention)
        return readout, attention, curvature_penalty
