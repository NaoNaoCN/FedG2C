"""Gefra pre-training model and self-supervised objectives."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling


class GefraPretrainingModel(nn.Module):
    """Combine private and shared encoders for federated pre-training."""

    def __init__(
        self,
        local_encoder,
        moe_encoder,
        gating,
        feat_recon_decoder,
        topo_recon_decoder,
        topo_sem_recon_decoder,
        topk=2,
        input_proj=None,
        geo_projector=None,
        **kwargs,
    ):
        super().__init__()

        self.local_encoder = local_encoder
        self.moe_encoder = moe_encoder
        self.gating = gating
        # Preserve all decoder attributes as part of the checkpoint schema.
        self.feat_recon_decoder = feat_recon_decoder
        self.topo_recon_decoder = topo_recon_decoder
        self.topo_sem_recon_decoder = topo_sem_recon_decoder

        self.input_proj = input_proj if input_proj is not None else nn.Identity()
        self.geo_projector = geo_projector

        # Blend local and MoE embeddings in the common hidden space.
        self.fusion_weight = nn.Parameter(torch.tensor(2.0))

        self.topk = topk

    @property
    def get_encoder(self):
        return self.moe_encoder

    def save_encoder(self, path):
        torch.save(self.moe_encoder.state_dict(), path)

    def save_geo_projector(self, path):
        if self.geo_projector is not None:
            torch.save(self.geo_projector.state_dict(), path)

    def feat_recon(self, z):
        return self.feat_recon_decoder(z)

    def topo_recon_loss(self, z, pos_edge_index, neg_edge_index=None, ratio=1.0):
        """Compute edge reconstruction loss with optional positive sampling."""

        if ratio == 0.0:
            return torch.tensor(0.0, device=z.device)

        if ratio != 1.0:
            num_pos_edges = int(pos_edge_index.size(1) * ratio)
            num_pos_edges = max(num_pos_edges, 1)
            perm = torch.randperm(pos_edge_index.size(1))
            perm = perm[:num_pos_edges]
            pos_edge_index = pos_edge_index[:, perm]

        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))

        # Sharpen reconstruction logits while keeping logarithms numerically stable.
        tau = 0.3
        pos_score = torch.sigmoid(
            self.topo_recon_decoder(z, pos_edge_index, sigmoid=False) / tau
        ).clamp(1e-6, 1 - 1e-6)
        neg_score = torch.sigmoid(
            self.topo_recon_decoder(z, neg_edge_index, sigmoid=False) / tau
        ).clamp(1e-6, 1 - 1e-6)
        pos_loss = -torch.log(pos_score).mean()
        neg_loss = -torch.log(1 - neg_score).mean()

        return pos_loss + neg_loss

    def batch_struct_align_loss(self, z1, z2):
        """Align pairwise similarities of local and MoE batch embeddings."""
        z1 = F.normalize(z1, p=2, dim=1)
        z2 = F.normalize(z2, p=2, dim=1)

        sim1 = torch.mm(z1, z1.t())
        sim2 = torch.mm(z2, z2.t())

        return F.mse_loss(sim1, sim2)

    def forward(self, aug_g, g, topo_recon_ratio=1.0, bs=None):
        x, edge_index, edge_attr = aug_g[0], aug_g[1], aug_g[2]
        orig_edge_index = g[1]
        # Some graph datasets store features or edge attributes as integer tensors.
        x = x.float()
        if edge_attr is not None:
            edge_attr = edge_attr.float()

        # Both encoder paths consume the same client-private feature projection.
        x_proj = self.input_proj(x)

        # Encode each node with every expert before top-k aggregation.
        expert_outputs = self.moe_encoder(x_proj, edge_index, edge_attr)

        # Aggregate manifold experts only after their per-expert Euclidean projection.
        if self.geo_projector is not None:
            euclidean_expert_outputs = self.geo_projector(expert_outputs)
        else:
            euclidean_expert_outputs = expert_outputs

        hidden_dim = euclidean_expert_outputs.size(1)

        gate_logits = self.gating(euclidean_expert_outputs)

        # Use full gating probabilities for statistics and top-k aggregation.
        k = self.topk
        gate_probs_all = F.softmax(gate_logits, dim=-1)

        topk_probs, topk_indices = torch.topk(gate_probs_all, k, dim=-1)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
        topk_probs = topk_probs.unsqueeze(1)

        topk_indices_expanded = topk_indices.unsqueeze(1).expand(
            -1, hidden_dim, -1
        )
        selected_euclidean_outputs = torch.gather(
            euclidean_expert_outputs, 2, topk_indices_expanded
        )
        z_moe = (selected_euclidean_outputs * topk_probs).sum(dim=-1)

        z_local = self.local_encoder(x_proj, edge_index)

        # Fuse local and MoE embeddings.
        alpha = torch.sigmoid(self.fusion_weight)
        z = alpha * z_local + (1 - alpha) * z_moe

        topo_recon_loss = self.topo_recon_loss(z, orig_edge_index, ratio=topo_recon_ratio)

        # Neighbor-loader context nodes are excluded to keep pairwise alignment bounded.
        if bs is not None:
            struct_align_loss = self.batch_struct_align_loss(z_local[:bs], z_moe[:bs])
        else:
            struct_align_loss = self.batch_struct_align_loss(z_local, z_moe)

        # Minimize a bounded negative pairwise distance to diversify expert curvatures.
        if hasattr(self.moe_encoder.experts[0], "encoder1"):
            ks = torch.stack(
                [e.encoder1.manifold.k for e in self.moe_encoder.experts]
            )
            curvature_div_loss = (-torch.pdist(ks.view(-1, 1)).mean()).clamp(min=-2.0)
        else:
            curvature_div_loss = torch.tensor(0.0, device=z_moe.device)

        losses = {
            "topo_recon_loss": topo_recon_loss,
            "struct_align_loss": struct_align_loss,
            "curvature_div_loss": curvature_div_loss,
        }

        mean_gate_weights = gate_probs_all.mean(dim=0).squeeze(0)
        return z, gate_probs_all, mean_gate_weights, losses
