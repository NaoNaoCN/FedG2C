# ===== FedG2C ADD BEGIN A： local-context GraphSAGE predictor =====
"""One standard PyG SAGEConv followed by activation and latent projection.

GraphSAGE: Hamilton et al., NeurIPS 2017. Library: torch_geometric.nn.SAGEConv.
GraphMAE (KDD 2022) motivates a shallow graph decoder, but reconstructs features.
Using GraphSAGE to predict momentum latent representations is our adaptation.
"""
from torch import nn
from torch_geometric.nn import SAGEConv


class GraphSAGEPredictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config.predictor_dim
        self.embedding = nn.Linear(config.hidden_dim, d)
        self.position = nn.Linear(len(config.walk_steps), d) if config.position_encoding == 'random_walk' else None
        self.conv = SAGEConv(d, d, aggr='mean', root_weight=True, normalize=False)
        self.activation = nn.ReLU()
        self.output = nn.Linear(d, config.hidden_dim)

    def forward(self, main_repr, visible_position, prediction_indices, edge_index):
        h = self.embedding(main_repr)
        if self.position is not None:
            h = h + self.position(visible_position)
        # Only visible edges; no edge attributes or complete-reference edges.
        # Root term retains node information even when masking isolates a node.
        prediction = self.output(self.activation(self.conv(h, edge_index)))
        return prediction[prediction_indices]
# ===== FedG2C ADD END A =====
