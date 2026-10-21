# ===== FedG2C ADD BEGIN A： local neighborhoods, paired edge removal, channel masking =====
"""BGRL input corruption principles; independent, tensor-only implementation.

Reference: nerdslab/bgrl@60f9f19, bgrl/transforms.py (Apache-2.0).
Adaptations: local generators; undirected edge groups; bounded neighborhoods;
category-safe feature masking is applied by the private encoding branch.
No full-graph positional encoding or class/factor labels enter the main branch.
"""
from dataclasses import dataclass
import torch
from .context import GraphView, ContextSampler, induced_view


@dataclass
class EdgePredictionPair:
    reference_view: GraphView
    visible_view: GraphView
    prediction_indices: torch.Tensor
    feature_keep: torch.Tensor  # one channel mask shared by every node
    root_index: int
    removed_edges: int

    def to(self, device):
        return EdgePredictionPair(self.reference_view.to(device), self.visible_view.to(device),
                                  self.prediction_indices.to(device), self.feature_keep.to(device),
                                  self.root_index, self.removed_edges)


def mask_edges(edge_index, edge_attr, rate, num_nodes, undirected, generator):
    """Never reconstruct missing reverse edges; retain exact surviving rows/attrs.

    For undirected inputs, sample once per unordered pair. This prevents a kept
    reverse arc or duplicate edge from revealing a supposedly removed bond.
    Directed mode groups duplicates of the same ordered pair only.
    """
    if not 0 <= rate <= 1:
        raise ValueError("Edge mask rate must be in [0,1]")
    if edge_index.device.type != "cpu":
        raise ValueError("Sample on CPU, then move the bounded view")
    u, v = edge_index
    ordered = u * num_nodes + v
    if undirected and not torch.equal(torch.unique(ordered), torch.unique(v * num_nodes + u)):
        raise ValueError("undirected=true requires reverse arcs; set false for directed input")
    codes = torch.minimum(u, v) * num_nodes + torch.maximum(u, v) if undirected else ordered
    groups, inverse = torch.unique(codes, return_inverse=True)
    keep_groups = torch.rand(len(groups), generator=generator) >= rate
    keep = keep_groups[inverse]
    return edge_index[:, keep], None if edge_attr is None else edge_attr[keep], keep


def sample_feature_keep(channels, rate, generator):
    """BGRL-style per-feature-dimension mask, NOT elementwise or node dropping."""
    if channels < 1 or not 0 <= rate <= 1:
        raise ValueError("Invalid feature mask parameters")
    return torch.rand(channels, generator=generator) >= rate


class NeighborhoodEdgeSampler(ContextSampler):
    """k-hop neighborhood with a per-view cap, never fill from other components.

    Undirected adjacency is used ONLY to choose a region. Message passing and
    random walks retain the actual edge orientation. Feature values remain raw.
    """
    @staticmethod
    def neighborhood(neighbors, root, hops, cap, generator=None):
        chosen, frontier = {root}, [root]
        for _ in range(hops):
            candidates = sorted(set().union(*(neighbors[u] for u in frontier)) - chosen) if frontier else []
            if not candidates:
                break
            if generator is not None:
                order = torch.randperm(len(candidates), generator=generator).tolist()
                candidates = [candidates[j] for j in order]
            frontier = candidates[:max(0, cap - len(chosen))]
            chosen.update(frontier)
            if len(chosen) >= cap:
                break
        return torch.tensor(sorted(chosen), dtype=torch.long)

    def sample_edge_view(self, config, feature_channels, generator):
        roots = [i for i, neighbors in enumerate(self.neighbors) if neighbors - {i}]
        if not roots:
            raise ValueError("No non-self edge in source graph; cannot construct an edge-masking view")
        root = roots[int(torch.randint(len(roots), (1,), generator=generator))]
        nodes = self.neighborhood(self.neighbors, root, config.context_hops, config.max_nodes, generator)
        reference = induced_view(self.graph, nodes)
        root_index = int((nodes == root).nonzero().item())
        edges, attrs, keep = mask_edges(reference.edge_index, reference.edge_attr,
                                       config.edge_mask_rate, len(nodes), config.undirected, generator)
        visible = GraphView(reference.x, edges, attrs, reference.original_node_id)
        neighbors = [set() for _ in range(len(nodes))]
        for u, v in edges.t().tolist():
            neighbors[u].add(v)
            neighbors[v].add(u)
        # Choose prediction positions on the VISIBLE graph; hidden edges cannot
        # affect the query-neighborhood descriptor through this operation.
        prediction = self.neighborhood(neighbors, root_index, config.prediction_hops, len(nodes))
        return EdgePredictionPair(reference, visible, prediction,
                                  sample_feature_keep(feature_channels, config.feature_mask_rate, generator),
                                  root_index, int((~keep).sum()))
# ===== FedG2C ADD END A =====
