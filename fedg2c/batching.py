# ===== FedG2C ADD BEGIN A/C: independent bounded-graph physical batches =====
"""Pack message passing; pad only the per-graph composer and walk matrices.

No node or edge is added to an individual view. Padding has no message edges
and never participates in attention, loss, pooling, or normalization statistics.
Collation consumes no random numbers and preserves view/edge/query ordering.
"""
from dataclasses import dataclass, fields, replace
from functools import cached_property

import torch


@dataclass
class GraphBatch:
    x: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor | None
    original_node_id: torch.Tensor
    sizes: tuple
    offsets: torch.Tensor
    node_batch: torch.Tensor
    local_node: torch.Tensor
    dense_positions: torch.Tensor
    valid_nodes: torch.Tensor

    @classmethod
    def collate(cls, views):
        if not views:
            raise ValueError('Cannot batch an empty view list')
        if any(v.x.device.type != 'cpu' or v.edge_index.device.type != 'cpu' for v in views):
            raise ValueError('Collate CPU views before moving the physical batch')
        sizes = tuple(v.num_nodes for v in views)
        if min(sizes) < 1:
            raise ValueError('Each view must have at least one node')
        if any(v.x.shape[1:] != views[0].x.shape[1:] or
               (v.edge_attr is None) != (views[0].edge_attr is None) for v in views):
            raise ValueError('A physical batch must use one feature schema')
        offsets = torch.tensor([0, *sizes[:-1]], dtype=torch.long).cumsum(0)
        node_batch = torch.repeat_interleave(torch.arange(len(views)), torch.tensor(sizes))
        local_node = torch.cat([torch.arange(n) for n in sizes])
        return cls(
            torch.cat([v.x for v in views]),
            torch.cat([v.edge_index + offset for v, offset in zip(views, offsets)], dim=1),
            None if views[0].edge_attr is None else torch.cat([v.edge_attr for v in views]),
            torch.cat([v.original_node_id for v in views]), sizes, offsets, node_batch,
            local_node, node_batch * max(sizes) + local_node,
            torch.arange(max(sizes)).unsqueeze(0) < torch.tensor(sizes).unsqueeze(1))

    @property
    def num_graphs(self):
        return len(self.sizes)

    @property
    def num_nodes(self):
        return self.x.shape[0]

    @property
    def max_nodes(self):
        return max(self.sizes)

    def to(self, device, non_blocking=False):
        # Cached matrices are intentionally rebuilt on the destination device.
        return replace(self, **{f.name: getattr(self, f.name).to(device, non_blocking=non_blocking)
                                for f in fields(self) if isinstance(getattr(self, f.name), torch.Tensor)})

    def pad(self, value):
        shape = (self.num_graphs * self.max_nodes, *value.shape[1:])
        padded = value.new_zeros(shape).index_copy(0, self.dense_positions, value)
        return padded.reshape(self.num_graphs, self.max_nodes, *value.shape[1:])

    def unpad(self, value):
        return value.flatten(0, 1).index_select(0, self.dense_positions)

    @cached_property
    def adjacency(self):
        adjacency = self.x.new_zeros((self.num_graphs, self.max_nodes, self.max_nodes),
                                     dtype=torch.float32)
        u, v = self.edge_index
        adjacency[self.node_batch[u], self.local_node[u], self.local_node[v]] = 1
        return adjacency

    def relations(self, dtype):
        adjacency = self.adjacency.to(dtype)
        identity = torch.diag_embed(self.valid_nodes.to(dtype))
        two_hop = (adjacency @ adjacency > 0).to(dtype)
        return torch.stack((identity, adjacency, two_hop), dim=-1)

    def root_relations(self, roots, dtype):
        """One query row per view; two-hop paths still use the entire view."""
        rows = torch.arange(self.num_graphs, device=self.x.device)
        direct = self.adjacency[rows, roots]
        identity = roots[:, None] == torch.arange(self.max_nodes, device=roots.device)
        two_hop = (direct.unsqueeze(1) @ self.adjacency).squeeze(1) > 0
        return torch.stack((identity, direct.bool(), two_hop), dim=-1).to(dtype)

    def walk_encoding(self, steps, dtype):
        adjacency = self.adjacency.to(dtype)
        transition = adjacency / adjacency.sum(-1, keepdim=True).clamp_min(1)
        current = torch.diag_embed(self.valid_nodes.to(dtype))
        wanted, values = set(steps), {}
        for step in range(1, max(steps) + 1):
            current = current @ transition
            if step in wanted:
                values[step] = current.diagonal(dim1=-2, dim2=-1)
        return self.unpad(torch.stack([values[k] for k in steps], dim=-1))


@dataclass
class EdgePredictionBatch:
    reference_view: GraphBatch
    visible_view: GraphBatch
    prediction_indices: torch.Tensor
    query_batch: torch.Tensor
    query_local: torch.Tensor
    query_counts: torch.Tensor
    feature_keep: torch.Tensor

    @classmethod
    def collate(cls, pairs):
        # These checks run before transfer, avoiding per-view CUDA scalar reads.
        for pair in pairs:
            if not torch.equal(pair.reference_view.original_node_id, pair.visible_view.original_node_id):
                raise ValueError('Both branches must retain the same node ordering')
            q = pair.prediction_indices
            if q.device.type != 'cpu' or q.ndim != 1 or q.dtype != torch.long or not q.numel():
                raise ValueError('Need nonempty one-dimensional CPU prediction indices')
            if q.min() < 0 or q.max() >= pair.visible_view.num_nodes:
                raise ValueError('Prediction index outside local view')
        reference = GraphBatch.collate([p.reference_view for p in pairs])
        visible = GraphBatch.collate([p.visible_view for p in pairs])
        counts = torch.tensor([p.prediction_indices.numel() for p in pairs])
        return cls(reference, visible,
                   torch.cat([p.prediction_indices + offset for p, offset in zip(pairs, visible.offsets)]),
                   torch.repeat_interleave(torch.arange(len(pairs)), counts),
                   torch.cat([torch.arange(n) for n in counts.tolist()]), counts,
                   torch.stack([p.feature_keep for p in pairs]))

    def to(self, device, non_blocking=False):
        return replace(self, **{f.name: getattr(self, f.name).to(device, non_blocking=non_blocking)
                                for f in fields(self)})
# ===== FedG2C ADD END A/C =====
