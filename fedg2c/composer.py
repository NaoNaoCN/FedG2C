# ===== FedG2C ADD BEGIN A： relation-conditioned hypernetwork composition =====
"""PyTorch adaptation of the HYLA head-RMS and two-contraction core.

Source: https://github.com/smonsays/hypernetwork-attention/blob/
5dceaa1ebef5187a17a38d477a6b2fee8bc3dc38/hyla/models/attention.py
Attention as a Hypernetwork, ICLR 2025. MIT notice: third_party_notices/.
Graph relation bias, expert layout, residual and per-node pooling are our adaptation.
"""
import math
import torch
from torch import nn


def head_rms(scores, mask, epsilon=1e-6):
    """[...,heads,query,key]; ZERO masking and RMS along heads only."""
    scores = scores.masked_fill(~mask.unsqueeze(-3), 0)
    return scores * torch.rsqrt(scores.square().mean(dim=-3, keepdim=True) + epsilon)


def hypernetwork_contract(code, value, nonlinear=True):
    # Port of attention.py:409-417. Keep the head-axis contraction and code reuse.
    intermediate = torch.einsum("...hqk,...khd->...qkd", code, value)
    if nonlinear:
        intermediate = intermediate.relu()
    return torch.einsum("...hqk,...qkd->...qhd", code, intermediate)


class RelationComposer(nn.Module):
    def __init__(self, hidden_dim, num_heads, num_experts, mode="hypernetwork_relu",
                 relation_condition=True):
        super().__init__()
        self.dim, self.heads, self.experts = hidden_dim, num_heads, num_experts
        self.depth = hidden_dim // num_heads
        self.mode, self.relation_condition = mode, relation_condition
        if mode == "linear_gate":
            self.gate = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                      nn.Linear(hidden_dim, 1))
        else:
            self.query = nn.Linear(hidden_dim, hidden_dim)
            self.key = nn.Linear(hidden_dim, hidden_dim)
            self.value = nn.Linear(hidden_dim, hidden_dim)
            self.output = nn.Linear(hidden_dim, hidden_dim)
            # Bias=False prevents an unrelated intercept from becoming the relation signal.
            self.relation = nn.Linear(3, num_heads, bias=False) if relation_condition else None
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, expert_repr, relations, relation_gate=None, return_routing=False):
        # One graph/patch per call prevents cross-graph attention by construction.
        p, m, d = expert_repr.shape
        if m != self.experts or d != self.dim or relations.shape != (p, p, 3):
            raise ValueError("Composer expects [P,M,d] and [P,P,3]")
        if self.mode == "linear_gate":
            weights = self.gate(expert_repr).squeeze(-1).softmax(dim=-1)
            z = self.norm((weights.unsqueeze(-1) * expert_repr).sum(dim=1))
            return (z, weights) if return_routing else z
        tokens = expert_repr.reshape(p * m, d)
        q = self.query(tokens).reshape(p * m, self.heads, self.depth) # [p*m, head, d_h]
        k = self.key(tokens).reshape(p * m, self.heads, self.depth)
        v = self.value(tokens).reshape(p * m, self.heads, self.depth)
        scores = torch.einsum("qhd,khd->hqk", q / math.sqrt(self.depth), k) # [head, p*m, p*m]
        if self.relation is not None:
            bias = self.relation(relations).permute(2, 0, 1) #[head, p, p]
            if relation_gate is not None:
                if relation_gate.shape != (p, p):
                    raise ValueError("relation_gate must be [P,P]")
                bias = bias * relation_gate.unsqueeze(0)

            # TODO: 观察scores和bias的数值，决定此处使用加法是否稳妥
            scores = scores + bias.repeat_interleave(m, 1).repeat_interleave(m, 2)
        # All tokens belong to this bounded patch; no hidden node is present.
        mask = torch.ones(p * m, p * m, dtype=torch.bool, device=tokens.device)
        if self.mode == "standard_attention":
            code = scores.masked_fill(~mask.unsqueeze(0), -torch.inf).softmax(dim=-1)
            out = torch.einsum("hqk,khd->qhd", code, v)
        else:
            code = head_rms(scores, mask)
            out = hypernetwork_contract(code, v, nonlinear=self.mode == "hypernetwork_relu")
        out = self.norm(tokens + self.output(out.reshape(p * m, d)))
        z = out.reshape(p, m, d).mean(dim=1) # [p, d]
        return (z, code) if return_routing else z

    # ===== FedG2C ADD BEGIN A/C: batch axis keeps attention local to each view =====
    def forward_batch(self, expert_repr, relations, valid_nodes, relation_gate=None,
                      return_routing=False):
        b, p, m, d = expert_repr.shape
        if (m != self.experts or d != self.dim or relations.shape != (b, p, p, 3)
                or valid_nodes.shape != (b, p) or valid_nodes.dtype != torch.bool):
            raise ValueError('Composer batch expects [B,P,M,d], [B,P,P,3], boolean [B,P]')
        if self.mode == 'linear_gate':
            weights = self.gate(expert_repr).squeeze(-1).softmax(dim=-1)
            z = self.norm((weights.unsqueeze(-1) * expert_repr).sum(dim=2))
            z = z * valid_nodes.unsqueeze(-1)
            return (z, weights) if return_routing else z
        tokens = expert_repr.reshape(b, p * m, d)
        q = self.query(tokens).reshape(b, p * m, self.heads, self.depth)
        k = self.key(tokens).reshape(b, p * m, self.heads, self.depth)
        v = self.value(tokens).reshape(b, p * m, self.heads, self.depth)
        scores = torch.einsum('bqhd,bkhd->bhqk', q / math.sqrt(self.depth), k)
        if self.relation is not None:
            bias = self.relation(relations).permute(0, 3, 1, 2)
            if relation_gate is not None:
                if relation_gate.shape != (b, p, p):
                    raise ValueError('Batched relation_gate must be [B,P,P]')
                bias = bias * relation_gate.unsqueeze(1)
            scores = scores + bias.repeat_interleave(m, 2).repeat_interleave(m, 3)
        valid = valid_nodes.repeat_interleave(m, dim=1)
        mask = valid.unsqueeze(2) & valid.unsqueeze(1)
        if self.mode == 'standard_attention':
            # Mask keys first: padded queries must not softmax an all -inf row.
            code = scores.masked_fill(~valid[:, None, None, :], -torch.inf).softmax(dim=-1)
            code = code * valid[:, None, :, None]
            out = torch.einsum('bhqk,bkhd->bqhd', code, v)
        else:
            code = head_rms(scores, mask)
            out = hypernetwork_contract(code, v, nonlinear=self.mode == 'hypernetwork_relu')
        out = self.norm(tokens + self.output(out.reshape(b, p * m, d)))
        z = out.reshape(b, p, m, d).mean(dim=2) * valid_nodes.unsqueeze(-1)
        return (z, code) if return_routing else z
    # ===== FedG2C ADD END A/C =====
# ===== FedG2C ADD END A =====
