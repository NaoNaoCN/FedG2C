# ===== FedG2C ADD BEGIN A： explicit allowed-source tensor manifest =====
"""Read tensor dictionaries only. Source/factor filtering must precede this loader."""
import json
from pathlib import Path
import torch
from .context import GraphView, validate_graph


def read_manifest(path):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != 1 or not manifest.get("protocol_id"):
        raise ValueError("Need schema_version=1 and an explicit protocol_id")
    clients = manifest.get("clients", [])
    if not clients or len({str(c["id"]) for c in clients}) != len(clients):
        raise ValueError("Need nonempty clients with unique IDs")
    for c in clients:
        if not c.get("graphs") or "input_dim" not in c.get("feature_schema", {}):
            raise ValueError("Each client needs graph paths and a feature_schema")
        c["graphs"] = [str((path.parent / item).resolve()) for item in c["graphs"]]
    return manifest


def iter_graphs(paths):
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        graphs = payload if isinstance(payload, list) else [payload]
        for graph in graphs:
            x, edge_index = graph["x"], graph["edge_index"]
            view = GraphView(x, edge_index, graph.get("edge_attr"),
                             graph.get("original_node_id", torch.arange(x.shape[0])))
            validate_graph(view)
            # Deliberately do not pass y, PE, split masks, factor labels or caches.
            yield view


def write_smoke_manifest(directory):
    """Small generic graphs only; these are NOT the future server SBM probes."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    gen = torch.Generator().manual_seed(7)
    records = []
    for i in range(2):
        graphs = []
        for j in range(3):
            n = 10 + j
            u = torch.arange(n)
            v = (u + 1) % n
            edges = torch.cat([torch.stack([u, v]), torch.stack([v, u])], dim=1)
            graphs.append({"x": torch.randn(n, 5 + i, generator=gen), "edge_index": edges})
        filename = f"smoke_client_{i}.pt"
        torch.save(graphs, directory / filename)
        records.append({"id": str(i), "feature_schema": {"input_dim": 5 + i}, "graphs": [filename]})
    path = directory / "source_manifest.json"
    path.write_text(json.dumps({"schema_version": 1, "protocol_id": "synthetic_smoke_only",
                                "clients": records}, indent=2))
    return path
# ===== FedG2C ADD END A =====
