"""CPU adjacency index for bounded views without scanning the full graph/root.

The two permutations retain original edge rows, duplicates and attributes.
Large node targets are cropped lazily. No list of all neighborhoods is stored.
"""
import numpy as np
import torch

from .context import GraphView, validate_graph


class GraphNeighborhoodIndex:
    def __init__(self, graph):
        validate_graph(graph)
        if graph.x.device.type != 'cpu':
            raise ValueError('Index a CPU graph, then move bounded views')
        self.graph = graph
        row, col = graph.edge_index.numpy()
        self.row, self.col = row, col
        self.out_order = np.argsort(row, kind='stable')
        self.in_order = np.argsort(col, kind='stable')
        self.out_ptr = np.r_[0, np.bincount(row, minlength=graph.num_nodes).cumsum()]
        self.in_ptr = np.r_[0, np.bincount(col, minlength=graph.num_nodes).cumsum()]
        edge_nodes = np.unique(graph.edge_index[:, graph.edge_index[0] != graph.edge_index[1]].numpy())
        self.eligible_roots = torch.from_numpy(edge_nodes).long()
        # Existing samplers use neighbors[root] -> set, materialized on demand.
        self.neighbors = self

    def __len__(self):
        return self.graph.num_nodes

    def __getitem__(self, root):
        outgoing = self.out_order[self.out_ptr[root]:self.out_ptr[root+1]]
        incoming = self.in_order[self.in_ptr[root]:self.in_ptr[root+1]]
        return set(self.col[outgoing].tolist()) | set(self.row[incoming].tolist())

    def induced_edges(self, nodes):
        ids = nodes.tolist()
        mapping = {node: i for i, node in enumerate(ids)}
        selected = []
        for node in ids:
            arcs = self.out_order[self.out_ptr[node]:self.out_ptr[node+1]]
            # Preserve exact arc membership and row order, including duplicate arcs.
            selected.extend(int(arc) for arc in arcs if int(self.col[arc]) in mapping)
        selected = torch.tensor(sorted(selected), dtype=torch.long)
        original = self.graph.edge_index[:, selected]
        # neighborhood() returns sorted IDs, so searchsorted remaps only the
        # selected arcs without a fresh O(full_graph_nodes) allocation per root.
        edge_index = torch.searchsorted(nodes, original)
        return edge_index, selected

    def induce(self, nodes, topology=None):
        edge_index, selected = self.induced_edges(nodes) if topology is None else topology
        edge_attr = None if self.graph.edge_attr is None else self.graph.edge_attr[selected]
        return GraphView(self.graph.x[nodes], edge_index, edge_attr,
                         self.graph.original_node_id[nodes])
