# ===== FedG2C ADD BEGIN A： geometry-free experts and private input adapters =====
"""Reuses Gefra GraphEncoder; does not import experts.py (which imports geometry)."""
from copy import deepcopy
import torch
from torch import nn
from torch.nn import functional as F
from model.graph_encoders import GraphEncoder, EdgeAwareSAGEConv
from .context import structural_relations
from .composer import RelationComposer


class FeatureAdapter(nn.Module):
    """Private continuous projection OR explicit per-column category embeddings."""
    def __init__(self, input_dim, hidden_dim, cardinalities=None):
        super().__init__()
        self.input_dim = input_dim
        self.cardinalities = tuple(cardinalities or ())
        if self.cardinalities:
            if len(self.cardinalities) != input_dim or min(self.cardinalities) < 1:
                raise ValueError("One positive cardinality per categorical column required")
            self.embeddings = nn.ModuleList([nn.Embedding(c, hidden_dim) for c in self.cardinalities])
        else:
            self.projection = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError("Feature dimension differs from private schema")
        if not self.cardinalities:
            return self.projection(x.float())
        if x.dtype not in (torch.int32, torch.int64):
            raise ValueError("Categorical features must be integer indices")
        for j, cardinality in enumerate(self.cardinalities):
            if x[:, j].numel() and (x[:, j].min() < 0 or x[:, j].max() >= cardinality):
                raise ValueError("Category index outside explicit schema")
        return sum(emb(x[:, j].long()) for j, emb in enumerate(self.embeddings))


class EdgePassingGraphEncoder(GraphEncoder):
    """Additive override of Gefra GraphEncoder.encode, leaving original untouched.

    The original encode ignores edge_attr. This override sends projected edge
    features only to EdgeAwareSAGEConv; unsupported combinations fail explicitly.
    """
    def encode(self, x, edge_index, edge_attr=None):
        z = x
        for i, layer in enumerate(self.layers):
            if edge_attr is not None and not isinstance(layer, EdgeAwareSAGEConv):
                raise ValueError("Projected edge features require backbone=mysage")
            z = layer(z, edge_index, edge_attr) if isinstance(layer, EdgeAwareSAGEConv) else layer(z, edge_index)
            z = self.norms[i](z)
            if i < self.num_layers - 1:
                z = self.dropout(self.activation(z))
        return z


class SharedBackbone(nn.Module):
    """Shared schema: experts.* and composer.* only, no private adapters/predictor."""
    def __init__(self, config):
        super().__init__()
        # Same expert construction as Gefra EuclideanExperts, avoiding its geometry import.
        self.experts = nn.ModuleList([
            EdgePassingGraphEncoder(config.hidden_dim, config.hidden_dim, nn.ReLU,
                                    config.num_layers, backbone=config.backbone,
                                    normalize="layer", dropout=config.dropout)
            for _ in range(config.num_experts)
        ])
        self.composer = RelationComposer(config.hidden_dim, config.num_heads, config.num_experts,
                                         config.composer, config.relation_condition)

    def encode_experts(self, x, edge_index, edge_attr=None):
        return torch.stack([expert(x, edge_index, edge_attr) for expert in self.experts], dim=1)

    def forward(self, x, edge_index, edge_attr=None, relation_gate=None):
        h = self.encode_experts(x, edge_index, edge_attr)
        r = structural_relations(edge_index, x.shape[0], dtype=h.dtype)
        return self.composer(h, r, relation_gate=relation_gate)


class EncodingBranch(nn.Module):
    def __init__(self, shared: SharedBackbone, input_dim, edge_dim=0, node_cardinalities=None, edge_cardinalities=None):
        super().__init__()
        hidden = shared.composer.dim
        self.node_adapter = FeatureAdapter(input_dim, hidden, node_cardinalities)
        self.edge_adapter = FeatureAdapter(edge_dim, hidden, edge_cardinalities) if edge_dim else None
        self.shared = shared

    def forward(self, view):
        x = self.node_adapter(view.x)
        if self.edge_adapter is not None and view.edge_attr is None:
            raise ValueError("This client schema requires edge features")
        if self.edge_adapter is None and view.edge_attr is not None:
            raise ValueError("Unexpected edge features; do not silently discard them")
        edge = self.edge_adapter(view.edge_attr) if self.edge_adapter is not None else None
        return self.shared(x, view.edge_index, edge)


class LocalPredictionModel(nn.Module):
    """Predict the pooled prediction-region representation from visible context.

    Independently implemented EMA/stop-gradient mechanism based on I-JEPA (CVPR
    2023). No I-JEPA source copied. There is one region per sampled view, so no
    full-graph node ID/position encoding is passed to the predictor.
    """
    def __init__(self, config, shared, input_dim, edge_dim=0,
                 node_cardinalities=None, edge_cardinalities=None):
        super().__init__()
        self.config = config
        if edge_dim and config.backbone != "mysage":
            raise ValueError("Use mysage for edge features, or explicitly prepare topology-only inputs")
        self.online = EncodingBranch(shared, input_dim, edge_dim, node_cardinalities, edge_cardinalities)
        self.momentum_branch = None
        output_dim = config.hidden_dim
        if config.objective == "joint_embedding_prediction":
            self.momentum_branch = deepcopy(self.online).requires_grad_(False).eval()
        else:
            if node_cardinalities:
                raise ValueError("Masked-feature regression is continuous-only; categorical codes are not regression targets")
            output_dim = input_dim
        self.predictor = nn.Sequential(nn.Linear(config.hidden_dim, config.hidden_dim), nn.GELU(),
                                       nn.Linear(config.hidden_dim, output_dim))

    def train(self, mode=True):
        super().train(mode)
        if self.momentum_branch is not None:
            self.momentum_branch.eval()
        return self

    @torch.no_grad()
    def reset_momentum(self):
        if self.momentum_branch is not None:
            self.momentum_branch.load_state_dict(self.online.state_dict(), strict=True)
            self.momentum_branch.eval()

    @torch.no_grad()
    def update_momentum(self):
        if self.momentum_branch is None:
            return
        source = dict(self.online.named_parameters())
        for name, parameter in self.momentum_branch.named_parameters():
            parameter.mul_(self.config.momentum).add_(source[name], alpha=1 - self.config.momentum)
        # LayerNorm has no running statistics; explicit copies future-proof other buffers.
        buffers = dict(self.online.named_buffers())
        for name, buffer in self.momentum_branch.named_buffers():
            buffer.copy_(buffers[name])

    def forward(self, pair):
        context_z = self.online(pair.context)
        prediction = self.predictor(context_z.mean(dim=0, keepdim=True))
        with torch.no_grad():
            if self.momentum_branch is not None:
                full_z = self.momentum_branch(pair.full)
                reference = full_z[pair.prediction_indices].mean(dim=0, keepdim=True)
                reference = F.layer_norm(reference, (reference.shape[-1],))
            else:
                reference = pair.full.x[pair.prediction_indices].float().mean(dim=0, keepdim=True)
        # I-JEPA normalizes its latent reference, not both prediction and reference.
        loss = F.smooth_l1_loss(prediction, reference)
        return loss, {"prediction": prediction.detach(), "reference": reference.detach(),
                      "context_std": context_z.detach().std(dim=0, unbiased=False).mean()}
# ===== FedG2C ADD END A =====
