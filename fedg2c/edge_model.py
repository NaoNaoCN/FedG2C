# ===== FedG2C ADD BEGIN A： nodewise joint-embedding prediction, revision 2 =====
"""I-JEPA training mechanism with graph corruption; no upstream source copied."""
from copy import deepcopy
import torch
from torch import nn
from torch.nn import functional as F
from .model import EncodingBranch
from .node_predictor import NodeRepresentationPredictor
from .structural_encoding import random_walk_encoding


class MaskedEncodingBranch(EncodingBranch):
    """BGRL channel mask before continuous projection; after category embedding.

    Category IDs are never zeroed: zero can be a valid atom/bond category.
    The latter placement is our input-schema adaptation, not BGRL's original.
    """
    def forward(self, view, feature_keep=None):
        categorical = bool(self.node_adapter.cardinalities)
        raw = view.x
        if feature_keep is not None:
            channels = self.shared.composer.dim if categorical else raw.shape[1]
            if feature_keep.dtype != torch.bool or feature_keep.shape != (channels,):
                raise ValueError("Expected one boolean mask shared by all nodes")
            if not categorical:
                raw = raw * feature_keep.to(raw.dtype).unsqueeze(0)
        x = self.node_adapter(raw)
        if feature_keep is not None and categorical:
            x = x * feature_keep.to(x.dtype).unsqueeze(0)
        if self.edge_adapter is not None and view.edge_attr is None:
            raise ValueError("This client schema requires edge features")
        if self.edge_adapter is None and view.edge_attr is not None:
            raise ValueError("Unexpected edge features; specify the private schema")
        edge = self.edge_adapter(view.edge_attr) if self.edge_adapter is not None else None
        return self.shared(x, view.edge_index, edge)

    def forward_batch(self, view, feature_keep=None):
        categorical = bool(self.node_adapter.cardinalities)
        raw = view.x
        channels = self.shared.composer.dim if categorical else raw.shape[1]
        if feature_keep is not None and (feature_keep.dtype != torch.bool or
                                         feature_keep.shape != (view.num_graphs, channels)):
            raise ValueError('Expected one boolean channel mask per view')
        keep = None if feature_keep is None else feature_keep[view.node_batch]
        if keep is not None and not categorical:
            raw = raw * keep.to(raw.dtype)
        x = self.node_adapter(raw)
        if keep is not None and categorical:
            x = x * keep.to(x.dtype)
        if (self.edge_adapter is None) != (view.edge_attr is None):
            raise ValueError('Edge attributes must match the private feature schema')
        edge = self.edge_adapter(view.edge_attr) if self.edge_adapter is not None else None
        return self.shared.forward_batch(x, view, edge)


class NodewisePredictionModel(nn.Module):
    """Main branch learns; momentum branch supplies detached per-node references.

    Only main_branch.shared (experts + composer) is exported. Adapters, predictor,
    its position projection and the momentum copy are local training components.
    """
    def __init__(self, config, shared, input_dim, edge_dim=0,
                 node_cardinalities=None, edge_cardinalities=None):
        super().__init__()
        if edge_dim and config.backbone != "mysage":
            raise ValueError("Edge features require backbone=mysage")
        self.config = config
        self.main_branch = MaskedEncodingBranch(shared, input_dim, edge_dim,
                                                node_cardinalities, edge_cardinalities)
        self.momentum_branch = deepcopy(self.main_branch).requires_grad_(False).eval()
        # ===== FedG2C ADD BEGIN A： original constructor retained as comparison =====
        # Previous default: self.predictor = NodeRepresentationPredictor(config)
        if config.predictor_type == "graphsage":
            from .graph_predictor import GraphSAGEPredictor
            self.predictor = GraphSAGEPredictor(config)
        else:
            self.predictor = NodeRepresentationPredictor(config)
        # ===== FedG2C ADD END A =====

    def train(self, mode=True):
        super().train(mode)
        self.momentum_branch.eval()
        return self

    @torch.no_grad()
    def reset_momentum(self):
        self.momentum_branch.load_state_dict(self.main_branch.state_dict(), strict=True)
        self.momentum_branch.eval()

    @torch.no_grad()
    def update_momentum(self):
        source = dict(self.main_branch.named_parameters())
        for name, parameter in self.momentum_branch.named_parameters():
            parameter.mul_(self.config.momentum).add_(source[name], alpha=1-self.config.momentum)
        buffers = dict(self.main_branch.named_buffers())
        for name, buffer in self.momentum_branch.named_buffers():
            buffer.copy_(buffers[name])

    def forward(self, pair):
        if not torch.equal(pair.reference_view.original_node_id, pair.visible_view.original_node_id):
            raise ValueError("Both branches must retain the same node ordering")
        q = pair.prediction_indices
        if q.ndim != 1 or q.dtype != torch.long or not q.numel():
            raise ValueError("Need nonempty one-dimensional prediction indices")
        if q.min() < 0 or q.max() >= pair.visible_view.num_nodes:
            raise ValueError("Prediction index outside local view")
        main_z = self.main_branch(pair.visible_view, pair.feature_keep)
        # Recompute AFTER edge masking. No cached full-graph encoding is read.
        position = random_walk_encoding(pair.visible_view.edge_index, len(main_z),
                                        self.config.walk_steps, main_z.dtype)
        # ===== FedG2C ADD BEGIN A： GraphSAGE reads visible edges only =====
        # Previous call: prediction = self.predictor(main_z, position, q)
        if self.config.predictor_type == "graphsage":
            prediction = self.predictor(main_z, position, q, pair.visible_view.edge_index)
        else:
            prediction = self.predictor(main_z, position, q)
        # ===== FedG2C ADD END A =====
        with torch.no_grad():
            reference_z = self.momentum_branch(pair.reference_view)
            reference = F.layer_norm(reference_z, (reference_z.shape[-1],))[q]
        loss = F.smooth_l1_loss(prediction, reference)
        return loss, {"prediction": prediction.detach(), "reference": reference.detach(),
                      "context_std": main_z.detach().std(dim=0, unbiased=False).mean()}

    # ===== FedG2C ADD BEGIN A: same per-view objective in a physical batch =====
    def forward_batch(self, pair):
        if self.training and self.config.dropout:
            raise ValueError('Use microbatch_size=1 with dropout to preserve the scalar RNG stream')
        view, q = pair.visible_view, pair.prediction_indices
        main_z = self.main_branch.forward_batch(view, pair.feature_keep)
        # Both relation and random-walk matrices use THIS view's masked edges.
        position = view.walk_encoding(self.config.walk_steps, main_z.dtype)
        if self.config.predictor_type == 'graphsage':
            prediction = self.predictor(main_z, position, q, view.edge_index)
        else:
            prediction = self.predictor.forward_batch(main_z, position, pair)
        with torch.no_grad():
            reference_z = self.momentum_branch.forward_batch(pair.reference_view)
            reference = F.layer_norm(reference_z, (reference_z.shape[-1],))[q]
        node_losses = F.smooth_l1_loss(prediction, reference, reduction='none').mean(dim=-1)
        losses = node_losses.new_zeros(view.num_graphs).scatter_add(0, pair.query_batch, node_losses)
        losses = losses / pair.query_counts
        with torch.no_grad():
            padded = view.pad(main_z.detach())
            mask = view.valid_nodes.unsqueeze(-1)
            counts = mask.sum(dim=1)
            mean = (padded * mask).sum(dim=1) / counts
            variance = ((padded - mean.unsqueeze(1)).square() * mask).sum(dim=1) / counts
            context_std = variance.sqrt().mean(dim=-1)
        # Sum here; the client divides gradients once by the logical view count.
        return losses.sum(), {'losses': losses.detach(), 'prediction': prediction.detach(),
                              'reference': reference.detach(), 'context_std': context_std}
    # ===== FedG2C ADD END A =====
# ===== FedG2C ADD END A =====
