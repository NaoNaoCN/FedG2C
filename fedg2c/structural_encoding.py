# ===== FedG2C ADD BEGIN A： visible-graph random-walk structural encoding =====
"""Bounded dense specialization of GraphGPS get_rw_landing_probs.

Source: rampasek/GraphGPS@28015707, graphgps/transform/posenc_stats.py::get_rw_landing_probs.
MIT license retained in third_party_references/GraphGPS/LICENSE.
Same unweighted outgoing transition and isolated-node zero convention; no
space_dim rescaling. Duplicate arcs are treated as one structural edge.
"""
import torch


def random_walk_encoding(edge_index, num_nodes, steps, dtype=torch.float32):
    if num_nodes < 1 or not steps or any(k < 1 for k in steps):
        raise ValueError("Need nodes and positive walk lengths")
    adjacency = torch.zeros(num_nodes, num_nodes, dtype=dtype, device=edge_index.device)
    adjacency[edge_index[0], edge_index[1]] = 1
    transition = adjacency / adjacency.sum(-1, keepdim=True).clamp_min(1)
    # Dense powers are allowed only inside the sampler's bounded local view.
    values, current = {}, torch.eye(num_nodes, dtype=dtype, device=edge_index.device)
    wanted = set(steps)
    for step in range(1, max(steps) + 1):
        current = current @ transition
        if step in wanted:
            values[step] = current.diagonal()
    return torch.stack([values[k] for k in steps], dim=-1)
# ===== FedG2C ADD END A =====
