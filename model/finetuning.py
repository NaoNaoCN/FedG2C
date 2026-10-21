"""Gefra fine-tuning model with a linear task classifier."""

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

from .prompt_readout import LearnableWeightReadout, RiemannianPromptReadout


def compute_multitask_loss(pred, y):
    """Compute the masked binary loss used by multi-task graph datasets."""
    criterion = nn.BCEWithLogitsLoss(reduction="none")

    if y.ndim == 1:
        if pred.ndim == 2 and pred.shape[1] > 1:
            y = y.unsqueeze(1).expand(-1, pred.shape[1])
        else:
            y = y.unsqueeze(1)

    if pred.ndim == 1:
        pred = pred.unsqueeze(1)

    assert pred.shape[0] == y.shape[0], (
        f"Batch size mismatch: pred.shape={pred.shape}, y.shape={y.shape}"
    )
    if pred.shape[1] != y.shape[1]:
        min_tasks = min(pred.shape[1], y.shape[1])
        pred = pred[:, :min_tasks]
        y = y[:, :min_tasks]

    y = y.clone()
    y[y == 0] = -1
    is_valid = y**2 > 0
    loss = pred.new_zeros(())

    for idx in range(y.shape[1]):
        exist_y = y[is_valid[:, idx], idx]
        exist_pred = pred[is_valid[:, idx], idx]
        task_loss = criterion(exist_pred.double(), (exist_y + 1) / 2)
        loss = loss + torch.sum(task_loss)

    return loss / torch.sum(is_valid).clamp_min(1)


class GefraFinetuningModel(nn.Module):
    """Adapt private encoders and prompt readout with a linear task head."""

    def __init__(
        self,
        local_encoder,
        moe_encoder,
        geo_projector,
        num_classes,
        input_proj=None,
        num_prompts: int = 4,
        lambda_curv: float = 1.0,
        use_learnable_weight: bool = False,
        local_encoder_only: bool = False,
    ):
        super().__init__()

        self.local_encoder_only = local_encoder_only
        self.local_encoder = local_encoder
        self.moe_encoder = moe_encoder
        self.geo_projector = geo_projector

        for param in self.moe_encoder.parameters():
            param.requires_grad = False
        for param in self.geo_projector.parameters():
            param.requires_grad = False

        self.input_proj = input_proj if input_proj is not None else nn.Identity()

        num_experts = len(moe_encoder.experts)
        embed_dim = moe_encoder.hidden_dim
        if use_learnable_weight:
            self.prompt_readout = LearnableWeightReadout(
                embed_dim=embed_dim, num_experts=num_experts
            )
        else:
            self.prompt_readout = RiemannianPromptReadout(
                embed_dim=embed_dim,
                num_experts=num_experts,
                num_prompts=num_prompts,
                lambda_curv=lambda_curv,
            )

        self.fusion_weight = nn.Parameter(torch.tensor(0.0))
        self.num_classes = num_classes
        self.decoder = nn.Linear(embed_dim, num_classes)

    def encode_with_aux(self, x, edge_index, edge_attr=None):
        """Return fused, local, and shared node embeddings."""
        x = x.float()
        if edge_attr is not None:
            edge_attr = edge_attr.float()
        h = self.input_proj(x)

        expert_outputs = self.moe_encoder(h, edge_index, edge_attr)
        euclidean_expert_outputs = self.geo_projector(expert_outputs)
        expert_embeds = euclidean_expert_outputs.permute(0, 2, 1)
        if hasattr(self.moe_encoder.experts[0], "manifold"):
            expert_curvatures = torch.stack(
                [expert.manifold.k.detach() for expert in self.moe_encoder.experts]
            )
        else:
            expert_curvatures = torch.zeros(
                len(self.moe_encoder.experts), device=h.device
            )

        z_shared, _, _ = self.prompt_readout(expert_embeds, expert_curvatures)
        z_local = self.local_encoder(h, edge_index)
        if self.local_encoder_only:
            z_fusion = z_local
            z_shared = z_local
        else:
            gamma = torch.sigmoid(self.fusion_weight)
            z_fusion = gamma * z_local + (1 - gamma) * z_shared
        return z_fusion, z_local, z_shared

    @torch.no_grad()
    def encode_with_aux_full(self, x, edge_index, edge_attr=None):
        """Encode a full graph and return auxiliary expert statistics."""
        x = x.float()
        if edge_attr is not None:
            edge_attr = edge_attr.float()
        h = self.input_proj(x)

        expert_outputs = self.moe_encoder(h, edge_index, edge_attr)
        euclidean_expert_outputs = self.geo_projector(expert_outputs)
        expert_embeds = euclidean_expert_outputs.permute(0, 2, 1)
        if hasattr(self.moe_encoder.experts[0], "manifold"):
            expert_curvatures = torch.stack(
                [expert.manifold.k.detach() for expert in self.moe_encoder.experts]
            )
        else:
            expert_curvatures = torch.zeros(
                len(self.moe_encoder.experts), device=h.device
            )
        z_shared, prompt_attn, _ = self.prompt_readout(
            expert_embeds, expert_curvatures
        )
        z_local = self.local_encoder(h, edge_index)
        if self.local_encoder_only:
            z_fusion = z_local
            z_shared = z_local
        else:
            gamma = torch.sigmoid(self.fusion_weight)
            z_fusion = gamma * z_local + (1 - gamma) * z_shared
        return (
            z_fusion,
            z_local,
            z_shared,
            expert_embeds,
            expert_curvatures,
            prompt_attn,
        )

    def encode(self, x, edge_index, edge_attr=None):
        """Return the local/shared fused node embeddings."""
        z_fusion, _, _ = self.encode_with_aux(x, edge_index, edge_attr)
        return z_fusion

    def encode_graph(self, x, edge_index, edge_attr=None, batch=None, pool="mean"):
        """Pool fused node embeddings into graph embeddings."""
        z = self.encode(x, edge_index, edge_attr)
        if pool == "mean":
            z = global_mean_pool(z, batch)
        elif pool == "sum":
            z = global_add_pool(z, batch)
        elif pool == "max":
            z = global_max_pool(z, batch)
        else:
            raise ValueError(f"Unsupported graph pooling mode: {pool}")
        return z

    def get_lin_logits(self, z):
        """Return one linear-head logit vector for each input embedding."""
        return self.decoder(z).reshape(-1, 1, self.num_classes)

    def compute_linear_loss(self, z, y, task="single"):
        """Compute the supervised loss from the linear task head."""
        pred = self.get_lin_logits(z).mean(1)
        if task == "single":
            return F.cross_entropy(pred, y.long())
        if task == "multi":
            return compute_multitask_loss(pred, y)
        raise ValueError(f"Unsupported fine-tuning task loss: {task}")

    def forward(self, x, edge_index, edge_attr=None):
        z = self.encode(x, edge_index, edge_attr)
        return self.get_lin_logits(z), z
