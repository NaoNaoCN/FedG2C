# ===== FedG2C ADD BEGIN A： crop before encoding, explicit original-node mapping =====
"""Thin PyG subgraph interface. No Graph-JEPA implementation is copied."""
from dataclasses import dataclass
import torch
from torch_geometric.utils import subgraph


@dataclass
class GraphView:
    x: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor | None
    original_node_id: torch.Tensor

    @property
    def num_nodes(self):
        return self.x.shape[0]

    def to(self, device):
        return GraphView(self.x.to(device), self.edge_index.to(device),
                         None if self.edge_attr is None else self.edge_attr.to(device),
                         self.original_node_id.to(device))


@dataclass
class ContextPair:
    full: GraphView
    context: GraphView
    prediction_indices: torch.Tensor  # indices in full, never predictor inputs

    def to(self, device):
        return ContextPair(self.full.to(device), self.context.to(device),
                           self.prediction_indices.to(device))


def validate_graph(graph):
    x, edges = graph.x, graph.edge_index
    if x.ndim != 2 or x.shape[0] < 2 or not torch.isfinite(x).all():
        raise ValueError("Graph requires finite [N,F] features and at least two nodes")
    if edges.dtype != torch.long or edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError("edge_index must be int64 [2,E]")
    if edges.numel() and (edges.min() < 0 or edges.max() >= x.shape[0]):
        raise ValueError("Edge endpoint outside graph")
    if graph.edge_attr is not None:
        if graph.edge_attr.ndim != 2 or graph.edge_attr.shape[0] != edges.shape[1]:
            raise ValueError("edge_attr must be [E,F_edge]")
        if not torch.isfinite(graph.edge_attr).all():
            raise ValueError("Nonfinite edge attributes")
    if graph.original_node_id.shape != (x.shape[0],):
        raise ValueError("original_node_id must be [N]")
    if torch.unique(graph.original_node_id).numel() != x.shape[0]:
        raise ValueError("original_node_id must be unique within a graph")


def induced_view(graph, nodes):
    edges, attrs = subgraph(nodes, graph.edge_index, graph.edge_attr,
                           relabel_nodes=True, num_nodes=graph.num_nodes)
    return GraphView(graph.x[nodes], edges, attrs, graph.original_node_id[nodes])


class ContextSampler:
    """CPU sampling cache for one permitted graph, including large node graphs.

    Adjacency is undirected ONLY for selecting a bounded region; induced
    edge_index retains the original directed edges. Never consult labels/PE.
    """
    def __init__(self, graph):
        validate_graph(graph)
        if graph.x.device.type != "cpu":
            raise ValueError("Keep source graph on CPU; move sampled views afterwards")
        self.graph = graph
        self.neighbors = [set() for _ in range(graph.num_nodes)]
        for u, v in graph.edge_index.t().tolist():
            self.neighbors[u].add(v)
            self.neighbors[v].add(u)

    @staticmethod
    def region(neighbors, count, generator):
        n = len(neighbors)
        chosen, seen = [], set()
        # Restart at another component when a component has too few nodes.
        for seed in torch.randperm(n, generator=generator).tolist():
            if seed in seen:
                continue
            queue = [seed]
            seen.add(seed)
            while queue and len(chosen) < count:
                u = queue.pop(0)
                chosen.append(u)
                available = sorted(neighbors[u] - seen)
                if available:
                    order = torch.randperm(len(available), generator=generator).tolist()
                    for j in order:
                        v = available[j]
                        seen.add(v)
                        queue.append(v)
            if len(chosen) == count:
                break
        return torch.tensor(sorted(chosen), dtype=torch.long)

    def sample(self, max_nodes, prediction_ratio, generator):
        if max_nodes < 2 or not 0 < prediction_ratio < 1:
            raise ValueError("Invalid context sampling configuration")
        count = min(max_nodes, self.graph.num_nodes)
        nodes = self.region(self.neighbors, count, generator)
        full = induced_view(self.graph, nodes)
        local_neighbors = [set() for _ in range(count)]
        for u, v in full.edge_index.t().tolist():
            local_neighbors[u].add(v)
            local_neighbors[v].add(u)
        hidden_count = min(count - 1, max(1, round(count * prediction_ratio)))
        prediction = self.region(local_neighbors, hidden_count, generator)
        visible = torch.ones(count, dtype=torch.bool)
        visible[prediction] = False
        context = induced_view(full, visible.nonzero().flatten())
        return ContextPair(full, context, prediction)


def structural_relations(edge_index, num_nodes, dtype=torch.float32):
    """Visible-only [P,P,3]: self, direct directed edge, directed two-hop reachability.

    Topological descriptors work for both molecules and nonmolecular graphs.
    No original-node ID, full-graph PE, class label or factor label is encoded.
    """
    adjacency = torch.zeros(num_nodes, num_nodes, device=edge_index.device, dtype=dtype)
    adjacency[edge_index[0], edge_index[1]] = 1
    identity = torch.eye(num_nodes, device=edge_index.device, dtype=dtype)
    two_hop = (adjacency @ adjacency > 0).to(dtype)
    return torch.stack([identity, adjacency, two_hop], dim=-1)
# ===== FedG2C ADD END A =====
