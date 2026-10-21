# ===== FedG2C ADD BEGIN A： I-JEPA-style sequence predictor with graph conditions =====
"""Independent PyTorch implementation of I-JEPA predictor organization.

Concept reference: facebookresearch/ijepa@52c1ae9,
VisionTransformerPredictor.forward. No upstream source is copied here.
Its CC BY-NC 4.0 reference snapshot is retained for audit only.
Graph adaptation: replace image-grid positions by visible-graph walk statistics.
All node features may remain visible under edge masking, unlike image masking.
"""
import torch
from torch import nn


class NodeRepresentationPredictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config.predictor_dim
        self.embedding = nn.Linear(config.hidden_dim, d)
        self.position = nn.Linear(len(config.walk_steps), d) if config.position_encoding == "random_walk" else None
        self.mask_token = nn.Parameter(torch.zeros(1, d))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d, config.predictor_heads, dim_feedforward=4*d,
                                       dropout=config.dropout, activation="gelu",
                                       batch_first=True, norm_first=True)
            for _ in range(config.predictor_depth)
        ])
        self.norm = nn.LayerNorm(d)
        self.output = nn.Linear(d, config.hidden_dim)

    def forward(self, main_repr, visible_position, prediction_indices):
        context = self.embedding(main_repr)
        if self.position is not None:
            position = self.position(visible_position)
        else:
            position = torch.zeros_like(context)
        context = context + position
        queries = self.mask_token.expand(len(prediction_indices), -1) + position[prediction_indices]
        sequence = torch.cat([context, queries], dim=0).unsqueeze(0)
        for block in self.blocks:
            sequence = block(sequence)
        return self.output(self.norm(sequence[0, len(context):]))
# ===== FedG2C ADD END A =====
